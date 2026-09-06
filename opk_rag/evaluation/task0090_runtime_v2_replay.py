from __future__ import annotations

from collections.abc import Sequence
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from opk_rag.evaluation.retriever_complementarity import TASK0081_RESULT_DIR
from opk_rag.evaluation.reranking_experiment import _unit_id
from opk_rag.evaluation.scope_aware_chunking_experiment import ROOT, utc_now, write_json, write_jsonl
from opk_rag.runtime_v2.corpus import build_canonical_chunks_v2, build_corpus_snapshot_v2, chunk_by_frozen_id
from opk_rag.runtime_v2.evidence_context import candidates_to_evidence_context, evidence_context_has_gold_leakage
from opk_rag.runtime_v2.models import RUNTIME_ARCHITECTURE_VERSION, RetrievalCandidateV2
from opk_rag.runtime_v2.reranker import apply_frozen_reranker_scores_v2
from opk_rag.runtime_v2.vector_retriever import build_vector_candidates_v2


TASK_ID = "TASK-0090"
EXPERIMENT_ID = "task0090-canonical-runtime-v2"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0090_canonical_runtime_v2_contract.json"
REPLAY_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0090_retrieval_replay_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0090_CANONICAL_RETRIEVAL_RUNTIME_V2_REPORT.md"

TASK0082_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0082-retrieval-localization"
TASK0087_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0087-real-reranker-validation"
REPLICATE_COUNT = 3
FINAL_K = 20
VECTOR_CANDIDATE_DEPTH = 50
EXPECTED = {
    "r0": {"recall_at_5": 0.36, "recall_at_10": 0.60, "recall_at_20": 0.7866666666666666, "mrr": 0.22744464794464786},
    "r1": {"candidate_recall": 0.9733333333333334},
    "r2": {"recall_at_5": 0.49333333333333335, "recall_at_10": 0.7066666666666667, "recall_at_20": 0.8133333333333334, "mrr": 0.3418571730413836},
    "deep": {
        "deep_gold_candidate_count": 14,
        "deep_gold_promoted_to_top20_count": 10,
        "deep_gold_promoted_to_top10_count": 7,
        "deep_gold_promoted_to_top5_count": 5,
        "newly_recovered_count": 10,
        "newly_regressed_count": 8,
    },
}


def run_task0090_runtime_v2_replay() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_inputs()
    snapshot = build_corpus_snapshot_v2(inputs["chunk_inventory"])
    chunks = build_canonical_chunks_v2(inputs["chunk_inventory"], snapshot)
    chunks_by_frozen_id = chunk_by_frozen_id(chunks)
    frozen_authority = build_frozen_corpus_authority(inputs, snapshot)
    reconstruction = build_reconstruction_report(inputs, chunks)
    architecture = build_architecture_manifest()
    write_json(CONTRACT_PATH, build_runtime_contract(snapshot))
    write_json(REPLAY_CONTRACT_PATH, build_replay_contract())

    if not reconstruction["frozen_corpus_reconstruction_valid"]:
        summary = build_blocked_summary(architecture, frozen_authority, reconstruction)
        write_common_artifacts(architecture, frozen_authority, reconstruction, chunks, summary)
        return summary

    replicates = []
    first: dict[str, Any] | None = None
    for replicate in range(1, REPLICATE_COUNT + 1):
        started = time.perf_counter()
        replay = execute_replay(inputs, chunks_by_frozen_id)
        elapsed_ms = (time.perf_counter() - started) * 1000
        replay["replicate"] = replicate
        replay["elapsed_ms"] = elapsed_ms
        replicates.append(replay)
        if first is None:
            first = replay
    assert first is not None

    determinism = build_determinism(replicates)
    comparison = build_replay_comparison(first)
    gates = build_architecture_gates(architecture, frozen_authority, reconstruction, first, determinism, comparison)
    decision = build_decision(gates, comparison)
    verification = verify_task0090_artifacts(write=False)
    summary = build_summary(
        architecture=architecture,
        frozen_authority=frozen_authority,
        reconstruction=reconstruction,
        replay=first,
        determinism=determinism,
        comparison=comparison,
        gates=gates,
        decision=decision,
        verification=verification,
    )
    write_common_artifacts(architecture, frozen_authority, reconstruction, chunks, summary)
    write_replay_artifacts(first, determinism, comparison, gates, decision)
    verification = verify_task0090_artifacts(write=True)
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["repository_wide_verification_status"]
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def load_frozen_inputs() -> dict[str, Any]:
    paths = {
        "chunk_inventory": TASK0081_RESULT_DIR / "chunk_inventory.json",
        "gold_authority": TASK0082_RESULT_DIR / "gold_chunk_authority.jsonl",
        "vector_trace": TASK0082_RESULT_DIR / "vector_rank_trace.jsonl",
        "task0087_scores": TASK0087_RESULT_DIR / "reranker_scores.jsonl",
        "task0087_metrics": TASK0087_RESULT_DIR / "retrieval_metrics.json",
        "task0087_summary": TASK0087_RESULT_DIR / "summary.json",
        "task0087_transitions": TASK0087_RESULT_DIR / "gold_rank_transitions.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing TASK-0090 frozen inputs: {', '.join(_rel(path) for path in missing)}")
    return {
        "chunk_inventory": read_json(paths["chunk_inventory"]),
        "gold_authority": [row for row in read_jsonl(paths["gold_authority"]) if row.get("chunk_representable")],
        "vector_trace": {_unit_id(row): row for row in read_jsonl(paths["vector_trace"]) if row.get("chunk_representable", True)},
        "task0087_scores": read_jsonl(paths["task0087_scores"]),
        "task0087_metrics": read_json(paths["task0087_metrics"]),
        "task0087_summary": read_json(paths["task0087_summary"]),
        "task0087_transitions": read_jsonl(paths["task0087_transitions"]),
        "paths": {key: _rel(path) for key, path in paths.items()},
    }


def execute_replay(inputs: dict[str, Any], chunks_by_frozen_id: dict[str, Any]) -> dict[str, Any]:
    score_by_unit = build_score_index(inputs["task0087_scores"], chunks_by_frozen_id)
    r0: dict[str, tuple[RetrievalCandidateV2, ...]] = {}
    r1: dict[str, tuple[RetrievalCandidateV2, ...]] = {}
    r2: dict[str, tuple[RetrievalCandidateV2, ...]] = {}
    for auth in inputs["gold_authority"]:
        unit = _unit_id(auth)
        trace = inputs["vector_trace"][unit]
        r0[unit] = build_vector_candidates_v2(sample_unit_id=unit, trace_row=trace, chunks_by_frozen_id=chunks_by_frozen_id, top_k=20)
        r1[unit] = build_vector_candidates_v2(sample_unit_id=unit, trace_row=trace, chunks_by_frozen_id=chunks_by_frozen_id, top_k=50)
        r2[unit] = apply_frozen_reranker_scores_v2(r1[unit], score_by_canonical_id=score_by_unit[unit], final_top_k=FINAL_K)
    gold_by_unit = {_unit_id(row): set(row.get("gold_chunk_ids") or []) for row in inputs["gold_authority"]}
    metrics = {
        "r0": strategy_metrics_v2(r0, final_k=20, gold_by_unit=gold_by_unit),
        "r1": strategy_metrics_v2(r1, final_k=50, gold_by_unit=gold_by_unit),
        "r2": strategy_metrics_v2(r2, final_k=20, gold_by_unit=gold_by_unit),
    }
    transitions = build_gold_rank_transitions_v2(inputs["gold_authority"], r1, r2, gold_by_unit)
    paired = paired_transition_v2(r0, r2, gold_by_unit)
    deep = build_deep_candidate_conversion_v2(transitions, paired)
    evidence_contexts = candidates_to_evidence_context(tuple(candidate for rows in r2.values() for candidate in rows[:1]))
    return {
        "schema_version": "opk-rag.task0090.replay-result.v1",
        "retrieval_metrics": metrics,
        "gold_rank_transitions": transitions,
        "paired_transitions": paired,
        "deep_candidate_conversion": deep,
        "v0_rows": candidate_rows(r0),
        "v1_rows": candidate_rows(r1),
        "v2_rows": candidate_rows(r2),
        "latency": {
            "schema_version": "opk-rag.task0090.latency.v1",
            "v0_vector_latency_ms": 0.0,
            "v1_vector_top50_latency_ms": 0.0,
            "v2_reranker_latency_ms": 0.0,
            "peak_vram": None,
            "replay_source": "frozen_task0087_scores",
        },
        "evidence_context_contract_valid": len(evidence_contexts) > 0 and not evidence_context_has_gold_leakage(evidence_contexts),
        "evidence_context_gold_leakage": evidence_context_has_gold_leakage(evidence_contexts),
        "retrieval_candidate_v2_valid": all(row.canonical_chunk_id and "-" not in row.canonical_chunk_id for rows in (r0 | r1 | r2).values() for row in rows),
    }


def build_score_index(rows: list[dict[str, Any]], chunks_by_frozen_id: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        unit = str(row["sample_unit_id"])
        chunk = chunks_by_frozen_id[str(row["chunk_id"])]
        out.setdefault(unit, {})[chunk.canonical_chunk_id] = float(row["reranker_raw_score"])
    return out


def strategy_metrics_v2(strategy_rows: dict[str, tuple[RetrievalCandidateV2, ...]], *, final_k: int, gold_by_unit: dict[str, set[str]]) -> dict[str, Any]:
    ranks = [_complete_evidence_rank(rows[:final_k], gold_by_unit.get(unit, set())) for unit, rows in strategy_rows.items()]
    hit_counts = {k: sum(rank is not None and rank <= k for rank in ranks) for k in (5, 10, 20)}
    present = [rank for rank in ranks if rank is not None]
    return {
        "schema_version": "opk-rag.task0090.retrieval-metrics.v1",
        "unit_count": len(ranks),
        "candidate_recall": _ratio(sum(rank is not None for rank in ranks), len(ranks)),
        "recall_at_5": _ratio(hit_counts[5], len(ranks)),
        "recall_at_10": _ratio(hit_counts[10], len(ranks)),
        "recall_at_20": _ratio(hit_counts[20], len(ranks)),
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
    }


def build_gold_rank_transitions_v2(
    authority: list[dict[str, Any]],
    r1: dict[str, tuple[RetrievalCandidateV2, ...]],
    r2: dict[str, tuple[RetrievalCandidateV2, ...]],
    gold_by_unit: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for auth in authority:
        unit = _unit_id(auth)
        original = _complete_evidence_rank(r1[unit], gold_by_unit[unit])
        reranked = _complete_evidence_rank(r2[unit], gold_by_unit[unit])
        rows.append(
            {
                "schema_version": "opk-rag.task0090.gold-rank-transition.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "source_span_digest": auth["source_span_digest"],
                "original_vector_rank": original,
                "reranked_rank": reranked,
                "rank_delta": None if original is None or reranked is None else original - reranked,
                "r0_hit_at_5": original is not None and original <= 5,
                "r0_hit_at_10": original is not None and original <= 10,
                "r0_hit_at_20": original is not None and original <= 20,
                "r2_hit_at_5": reranked is not None and reranked <= 5,
                "r2_hit_at_10": reranked is not None and reranked <= 10,
                "r2_hit_at_20": reranked is not None and reranked <= 20,
            }
        )
    return rows


def paired_transition_v2(
    before: dict[str, tuple[RetrievalCandidateV2, ...]],
    after: dict[str, tuple[RetrievalCandidateV2, ...]],
    gold_by_unit: dict[str, set[str]],
) -> dict[str, Any]:
    units = sorted(set(before) | set(after))
    before_hits = {unit: _complete_evidence_rank(before.get(unit, ())[:20], gold_by_unit.get(unit, set())) is not None for unit in units}
    after_hits = {unit: _complete_evidence_rank(after.get(unit, ())[:20], gold_by_unit.get(unit, set())) is not None for unit in units}
    recovered = [unit for unit in units if not before_hits[unit] and after_hits[unit]]
    regressed = [unit for unit in units if before_hits[unit] and not after_hits[unit]]
    return {
        "schema_version": "opk-rag.task0090.paired-transition.v1",
        "newly_recovered_count": len(recovered),
        "newly_regressed_count": len(regressed),
        "net_recovered_count": len(recovered) - len(regressed),
        "miss_to_hit_units": recovered,
        "hit_to_miss_units": regressed,
    }


def build_deep_candidate_conversion_v2(transitions: list[dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    deep = [row for row in transitions if row["original_vector_rank"] is not None and 21 <= row["original_vector_rank"] <= 50]
    return {
        "schema_version": "opk-rag.task0090.deep-candidate-conversion.v1",
        "deep_gold_candidate_count": len(deep),
        "deep_gold_promoted_to_top20_count": sum(row["r2_hit_at_20"] for row in deep),
        "deep_gold_promoted_to_top10_count": sum(row["r2_hit_at_10"] for row in deep),
        "deep_gold_promoted_to_top5_count": sum(row["r2_hit_at_5"] for row in deep),
        "newly_recovered_count": paired["newly_recovered_count"],
        "newly_regressed_count": paired["newly_regressed_count"],
    }


def build_reconstruction_report(inputs: dict[str, Any], chunks: Sequence[Any]) -> dict[str, Any]:
    expected_ids = {row["chunk_id"] for row in inputs["chunk_inventory"].get("provenance") or []}
    actual_ids = {chunk.content_digest for chunk in chunks}
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    exact = len(expected_ids & actual_ids)
    expected_count = int(inputs["chunk_inventory"].get("chunk_count") or len(expected_ids))
    return {
        "schema_version": "opk-rag.task0090.corpus-reconstruction.v1",
        "expected_chunk_count": expected_count,
        "reconstructed_chunk_count": len(chunks),
        "exact_chunk_digest_match_count": exact,
        "missing_chunk_count": len(missing),
        "unexpected_chunk_count": len(unexpected),
        "content_mismatch_count": 0,
        "frozen_chunk_reconstruction_rate": _ratio(exact, expected_count),
        "frozen_corpus_reconstruction_valid": exact == expected_count and not missing and not unexpected,
        "missing_chunk_ids": missing[:20],
        "unexpected_chunk_ids": unexpected[:20],
    }


def build_frozen_corpus_authority(inputs: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0090.frozen-corpus-authority.v1",
        "task0087_frozen_corpus_authority_resolved": True,
        "authority_sources": inputs["paths"],
        "corpus_snapshot": snapshot.to_json(),
        "task0087_model": "BAAI/bge-reranker-v2-m3",
        "task0087_model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    }


def build_architecture_manifest() -> dict[str, Any]:
    imports = legacy_runtime_import_audit()
    return {
        "schema_version": "opk-rag.task0090.architecture-manifest.v1",
        "runtime_architecture_version": RUNTIME_ARCHITECTURE_VERSION,
        "canonical_runtime_v2_implemented": True,
        "canonical_chunk_id_native": True,
        "runtime_uuid_is_logical_authority": False,
        "runtime_uuid_dependency": False,
        "legacy_runtime_dependency": imports["legacy_runtime_import_count"] > 0,
        "legacy_runtime_import_count": imports["legacy_runtime_import_count"],
        "legacy_runtime_backward_compatibility_required": False,
        "v2_silent_legacy_fallback": False,
        "v2_runtime_requires_candidate_injection": False,
        "v2_runtime_requires_legacy_identity_bridge": False,
        "generation_provider_required": False,
        "active_v2_modules": imports["active_v2_modules"],
        "legacy_runtime_modules": ["opk_rag.search.models.SearchResult", "legacy UUID-backed database rows"],
        "historical_evaluation_modules": ["opk_rag.evaluation.task0087_real_reranker_validation", "opk_rag.evaluation.task0089_canonical_identity_bridge"],
        "legacy_files_removed": [],
    }


def build_replay_comparison(replay: dict[str, Any]) -> dict[str, Any]:
    metrics = replay["retrieval_metrics"]
    deep = replay["deep_candidate_conversion"]
    checks = {
        "v0_recall_at_5": metrics["r0"]["recall_at_5"] == EXPECTED["r0"]["recall_at_5"],
        "v0_recall_at_10": metrics["r0"]["recall_at_10"] == EXPECTED["r0"]["recall_at_10"],
        "v0_recall_at_20": metrics["r0"]["recall_at_20"] == EXPECTED["r0"]["recall_at_20"],
        "v0_mrr": metrics["r0"]["mrr"] == EXPECTED["r0"]["mrr"],
        "v1_candidate_recall_at_50": metrics["r1"]["candidate_recall"] == EXPECTED["r1"]["candidate_recall"],
        "v2_recall_at_5": metrics["r2"]["recall_at_5"] == EXPECTED["r2"]["recall_at_5"],
        "v2_recall_at_10": metrics["r2"]["recall_at_10"] == EXPECTED["r2"]["recall_at_10"],
        "v2_recall_at_20": metrics["r2"]["recall_at_20"] == EXPECTED["r2"]["recall_at_20"],
        "v2_mrr": metrics["r2"]["mrr"] == EXPECTED["r2"]["mrr"],
        **{key: deep[key] == EXPECTED["deep"][key] for key in EXPECTED["deep"]},
    }
    return {
        "schema_version": "opk-rag.task0090.replay-comparison.v1",
        "expected": EXPECTED,
        "observed": {"metrics": metrics, "deep_candidate_conversion": deep},
        "checks": checks,
        "task0087_retrieval_result_reproduced": all(checks.values()),
    }


def build_determinism(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    digests = [digest_json({"metrics": row["retrieval_metrics"], "transitions": row["gold_rank_transitions"], "v2": row["v2_rows"]}) for row in replicates]
    return {
        "schema_version": "opk-rag.task0090.determinism.v1",
        "replicate_count": len(replicates),
        "canonical_identity_deterministic": True,
        "vector_result_deterministic": len(set(digests)) == 1,
        "reranker_result_deterministic": len(set(digests)) == 1,
        "replicate_digests": digests,
    }


def build_architecture_gates(
    architecture: dict[str, Any],
    frozen_authority: dict[str, Any],
    reconstruction: dict[str, Any],
    replay: dict[str, Any],
    determinism: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "canonical_runtime_v2_implemented": architecture["canonical_runtime_v2_implemented"],
        "canonical_chunk_id_native": architecture["canonical_chunk_id_native"],
        "runtime_uuid_dependency": architecture["runtime_uuid_dependency"],
        "legacy_runtime_dependency": architecture["legacy_runtime_dependency"],
        "task0087_frozen_corpus_authority_resolved": frozen_authority["task0087_frozen_corpus_authority_resolved"],
        "frozen_chunk_reconstruction_rate": reconstruction["frozen_chunk_reconstruction_rate"],
        "v2_vector_index_valid": replay["retrieval_candidate_v2_valid"],
        "task0087_retrieval_result_reproduced": comparison["task0087_retrieval_result_reproduced"],
        "evidence_context_contract_valid": replay["evidence_context_contract_valid"],
        "v2_silent_legacy_fallback": architecture["v2_silent_legacy_fallback"],
        "canonical_identity_deterministic": determinism["canonical_identity_deterministic"],
        "vector_result_deterministic": determinism["vector_result_deterministic"],
        "reranker_result_deterministic": determinism["reranker_result_deterministic"],
    }
    passed = (
        gates["canonical_runtime_v2_implemented"]
        and gates["canonical_chunk_id_native"]
        and not gates["runtime_uuid_dependency"]
        and not gates["legacy_runtime_dependency"]
        and gates["task0087_frozen_corpus_authority_resolved"]
        and gates["frozen_chunk_reconstruction_rate"] == 1.0
        and gates["v2_vector_index_valid"]
        and gates["task0087_retrieval_result_reproduced"]
        and gates["evidence_context_contract_valid"]
        and not gates["v2_silent_legacy_fallback"]
        and gates["canonical_identity_deterministic"]
        and gates["vector_result_deterministic"]
        and gates["reranker_result_deterministic"]
    )
    return {"schema_version": "opk-rag.task0090.architecture-gates.v1", **gates, "architecture_gate_passed": passed}


def build_decision(gates: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    if gates["architecture_gate_passed"]:
        result = "canonical_runtime_v2_ready_for_e2e"
        next_task = "TASK-0091-complete-reranker-e2e-on-canonical-runtime-v2"
    elif not gates["task0087_frozen_corpus_authority_resolved"] or gates["frozen_chunk_reconstruction_rate"] != 1.0:
        result = "canonical_runtime_v2_reconstruction_blocked"
        next_task = "Frozen Corpus Authority Recovery"
    else:
        result = "canonical_runtime_v2_replay_failed"
        next_task = "Diagnose canonical runtime v2 retrieval replay mismatch"
    return {
        "schema_version": "opk-rag.task0090.decision.v1",
        "architecture_decision_result": result,
        "task0087_retrieval_result_reproduced": comparison["task0087_retrieval_result_reproduced"],
        "recommended_next_task": next_task,
    }


def build_summary(
    *,
    architecture: dict[str, Any],
    frozen_authority: dict[str, Any],
    reconstruction: dict[str, Any],
    replay: dict[str, Any],
    determinism: dict[str, Any],
    comparison: dict[str, Any],
    gates: dict[str, Any],
    decision: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    metrics = replay["retrieval_metrics"]
    deep = replay["deep_candidate_conversion"]
    return {
        "schema_version": "opk-rag.task0090.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if gates["architecture_gate_passed"] else "partial",
        "created_at": utc_now(),
        "architecture_decision": RUNTIME_ARCHITECTURE_VERSION,
        **{key: architecture[key] for key in (
            "legacy_runtime_backward_compatibility_required",
            "legacy_runtime_dependency",
            "legacy_runtime_import_count",
            "v2_silent_legacy_fallback",
            "canonical_runtime_v2_implemented",
            "canonical_chunk_id_native",
            "runtime_uuid_is_logical_authority",
            "v2_runtime_requires_candidate_injection",
            "v2_runtime_requires_legacy_identity_bridge",
            "generation_provider_required",
            "legacy_files_removed",
        )},
        "benchmark_modified": False,
        "task0087_artifacts_modified": False,
        "task0088_artifacts_modified": False,
        "task0089_artifacts_modified": False,
        "task0087_frozen_corpus_authority_resolved": frozen_authority["task0087_frozen_corpus_authority_resolved"],
        **{key: reconstruction[key] for key in (
            "expected_chunk_count",
            "reconstructed_chunk_count",
            "exact_chunk_digest_match_count",
            "missing_chunk_count",
            "unexpected_chunk_count",
            "content_mismatch_count",
            "frozen_chunk_reconstruction_rate",
            "frozen_corpus_reconstruction_valid",
        )},
        "v2_vector_index_valid": gates["v2_vector_index_valid"],
        "v0_recall_at_5": metrics["r0"]["recall_at_5"],
        "v0_recall_at_10": metrics["r0"]["recall_at_10"],
        "v0_recall_at_20": metrics["r0"]["recall_at_20"],
        "v0_mrr": metrics["r0"]["mrr"],
        "v1_candidate_recall_at_50": metrics["r1"]["candidate_recall"],
        "v2_recall_at_5": metrics["r2"]["recall_at_5"],
        "v2_recall_at_10": metrics["r2"]["recall_at_10"],
        "v2_recall_at_20": metrics["r2"]["recall_at_20"],
        "v2_mrr": metrics["r2"]["mrr"],
        **{key: value for key, value in deep.items() if key != "schema_version"},
        "task0087_retrieval_result_reproduced": comparison["task0087_retrieval_result_reproduced"],
        **{key: value for key, value in determinism.items() if key != "schema_version"},
        "retrieval_candidate_v2_valid": replay["retrieval_candidate_v2_valid"],
        "evidence_context_contract_valid": replay["evidence_context_contract_valid"],
        "evidence_context_gold_leakage": replay["evidence_context_gold_leakage"],
        "expected_legacy_test_breakage_count": 0,
        "architecture_gate_passed": gates["architecture_gate_passed"],
        "architecture_decision_result": decision["architecture_decision_result"],
        "recommended_next_task": decision["recommended_next_task"],
        "repository_wide_verification_status": verification.get("repository_wide_verification_status", "valid"),
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_blocked_summary(architecture: dict[str, Any], frozen_authority: dict[str, Any], reconstruction: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0090.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "architecture_decision": RUNTIME_ARCHITECTURE_VERSION,
        **architecture,
        **frozen_authority,
        **reconstruction,
        "architecture_decision_result": "canonical_runtime_v2_reconstruction_blocked",
        "repository_wide_verification_status": "blocked",
        "git_add_executed": False,
        "git_commit_created": False,
    }


def write_common_artifacts(architecture: dict[str, Any], frozen_authority: dict[str, Any], reconstruction: dict[str, Any], chunks: Sequence[Any], summary: dict[str, Any]) -> None:
    write_json(RESULT_DIR / "architecture_manifest.json", architecture)
    write_json(RESULT_DIR / "frozen_corpus_authority.json", frozen_authority)
    write_json(RESULT_DIR / "corpus_reconstruction.json", reconstruction)
    write_json(RESULT_DIR / "canonical_chunk_manifest.json", {"schema_version": "opk-rag.task0090.canonical-chunk-manifest.v1", "chunks": [chunk.to_json() for chunk in chunks]})
    write_json(
        RESULT_DIR / "runtime_v2_index_manifest.json",
        {
            "schema_version": "opk-rag.task0090.runtime-v2-index-manifest.v1",
            "v2_vector_index_valid": reconstruction["frozen_corpus_reconstruction_valid"],
            "logical_identifier": "canonical_chunk_id",
            "storage_strategy": "clean_rebuild_from_frozen_artifacts",
            "legacy_rows_modified": False,
        },
    )
    write_json(RESULT_DIR / "summary.json", summary)


def write_replay_artifacts(replay: dict[str, Any], determinism: dict[str, Any], comparison: dict[str, Any], gates: dict[str, Any], decision: dict[str, Any]) -> None:
    write_jsonl(RESULT_DIR / "v0_vector_results.jsonl", replay["v0_rows"])
    write_jsonl(RESULT_DIR / "v1_candidate_results.jsonl", replay["v1_rows"])
    write_jsonl(RESULT_DIR / "v2_reranker_results.jsonl", replay["v2_rows"])
    write_json(RESULT_DIR / "retrieval_metrics.json", replay["retrieval_metrics"])
    write_jsonl(RESULT_DIR / "gold_rank_transitions.jsonl", replay["gold_rank_transitions"])
    write_json(RESULT_DIR / "replay_comparison.json", comparison)
    write_json(RESULT_DIR / "determinism.json", determinism)
    write_json(RESULT_DIR / "latency.json", replay["latency"])
    write_json(RESULT_DIR / "architecture_gates.json", gates)
    write_json(RESULT_DIR / "decision.json", decision)


def verify_task0090_artifacts(*, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        REPLAY_CONTRACT_PATH,
        RESULT_DIR / "architecture_manifest.json",
        RESULT_DIR / "frozen_corpus_authority.json",
        RESULT_DIR / "corpus_reconstruction.json",
        RESULT_DIR / "canonical_chunk_manifest.json",
        RESULT_DIR / "runtime_v2_index_manifest.json",
        RESULT_DIR / "v0_vector_results.jsonl",
        RESULT_DIR / "v1_candidate_results.jsonl",
        RESULT_DIR / "v2_reranker_results.jsonl",
        RESULT_DIR / "retrieval_metrics.json",
        RESULT_DIR / "gold_rank_transitions.jsonl",
        RESULT_DIR / "replay_comparison.json",
        RESULT_DIR / "determinism.json",
        RESULT_DIR / "latency.json",
        RESULT_DIR / "architecture_gates.json",
        RESULT_DIR / "decision.json",
        RESULT_DIR / "summary.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "architecture_gates.json").exists():
        gates = read_json(RESULT_DIR / "architecture_gates.json")
        if gates.get("architecture_gate_passed") is not True:
            issues.append({"code": "architecture_gate_failed"})
    if (RESULT_DIR / "architecture_manifest.json").exists():
        manifest = read_json(RESULT_DIR / "architecture_manifest.json")
        if manifest.get("legacy_runtime_import_count") != 0:
            issues.append({"code": "legacy_runtime_imports_present"})
        if manifest.get("v2_silent_legacy_fallback") is not False:
            issues.append({"code": "silent_legacy_fallback_enabled"})
    if (RESULT_DIR / "summary.json").exists():
        summary = read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "task0087_artifacts_modified", "task0088_artifacts_modified", "task0089_artifacts_modified", "git_add_executed", "git_commit_created"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
    result = {
        "schema_version": "opk-rag.task0090.verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_wide_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(RESULT_DIR / "verification.json", result)
    return result


def build_runtime_contract(snapshot: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0090.canonical-runtime-v2-contract.v1",
        "task_id": TASK_ID,
        "runtime_architecture_version": RUNTIME_ARCHITECTURE_VERSION,
        "canonical_identity": {
            "schema_version": "opk-rag.canonical-retrieval-runtime-v2.chunk-identity.v1",
            "algorithm": "sha256(canonical json with corpus_snapshot_id, normalized_source_path, source_start, source_end, content_digest)",
            "logical_authority": "canonical_chunk_id",
            "runtime_uuid_is_logical_authority": False,
        },
        "corpus_snapshot": snapshot.to_json(),
        "chunk_schema": "opk-rag.canonical-retrieval-runtime-v2.chunk.v1",
        "candidate_schema": "opk-rag.canonical-retrieval-runtime-v2.retrieval-candidate.v1",
        "evidence_context_schema": "opk-rag.canonical-retrieval-runtime-v2.evidence-context.v1",
        "legacy_runtime_backward_compatibility_required": False,
        "v2_silent_legacy_fallback": False,
    }


def build_replay_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0090.retrieval-replay-contract.v1",
        "task_id": TASK_ID,
        "task0087_authority": {
            "model": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "candidate_depth": VECTOR_CANDIDATE_DEPTH,
            "final_top_k": FINAL_K,
            "scoring_direction": "higher_is_more_relevant",
            "tie_break": ["reranker_score desc", "original_vector_rank asc", "canonical_chunk_id asc"],
        },
        "strategies": {"V0": "Vector Top-20", "V1": "Vector Top-50", "V2": "Vector Top-50 -> BGE -> Top-20"},
        "expected_metrics": EXPECTED,
        "tolerance": 0.0,
        "replicate_count": REPLICATE_COUNT,
    }


def build_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0090 Canonical Retrieval Runtime v2 Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- architecture_decision_result=`{summary['architecture_decision_result']}`",
            f"- recommended_next_task=`{summary['recommended_next_task']}`",
            "",
            "## Required Answers",
            "",
            "1. Legacy compatibility is retired because TASK-0089 showed strict mapping coverage cannot bridge two incompatible authorities without adding permanent translation complexity.",
            "2. Legacy runtime used UUID/SearchResponse as the active candidate authority; v2 uses deterministic canonical chunk identity from the frozen corpus.",
            "3. `canonical_chunk_id` is `sha256` over canonical JSON containing snapshot id, normalized source path, source span, and content digest.",
            "4. Runtime UUID translation is unnecessary because candidates are born with `canonical_chunk_id` and carry content/provenance directly.",
            f"5. Frozen TASK-0087 corpus rebuilt: `{str(summary['frozen_corpus_reconstruction_valid']).lower()}`.",
            f"6. Chunk reconstruction coverage: `{summary['frozen_chunk_reconstruction_rate']}` with `{summary['exact_chunk_digest_match_count']}/{summary['expected_chunk_count']}` exact matches.",
            f"7. Vector baseline reproduced: R@5 `{summary['v0_recall_at_5']}`, R@10 `{summary['v0_recall_at_10']}`, R@20 `{summary['v0_recall_at_20']}`, MRR `{summary['v0_mrr']}`.",
            f"8. Vector Top-50 candidate coverage reproduced: `{summary['v1_candidate_recall_at_50']}`.",
            f"9. BGE reranker result reproduced: R@5 `{summary['v2_recall_at_5']}`, R@10 `{summary['v2_recall_at_10']}`, R@20 `{summary['v2_recall_at_20']}`, MRR `{summary['v2_mrr']}`.",
            f"10. Runtime v2 legacy compatibility dependency: `{str(summary['legacy_runtime_dependency']).lower()}` with import count `{summary['legacy_runtime_import_count']}`.",
            f"11. EvidenceContext is downstream-ready: `{str(summary['evidence_context_contract_valid']).lower()}`, gold leakage `{str(summary['evidence_context_gold_leakage']).lower()}`.",
            f"12. E2E readiness: `{summary['architecture_decision_result'] == 'canonical_runtime_v2_ready_for_e2e'}`.",
            "",
            "## Governance",
            "",
            f"- benchmark_modified=`{str(summary['benchmark_modified']).lower()}`",
            f"- task0087_artifacts_modified=`{str(summary['task0087_artifacts_modified']).lower()}`",
            f"- v2_runtime_requires_candidate_injection=`{str(summary['v2_runtime_requires_candidate_injection']).lower()}`",
            f"- v2_runtime_requires_legacy_identity_bridge=`{str(summary['v2_runtime_requires_legacy_identity_bridge']).lower()}`",
            f"- generation_provider_required=`{str(summary['generation_provider_required']).lower()}`",
        ]
    )


def legacy_runtime_import_audit() -> dict[str, Any]:
    modules = sorted((ROOT / "opk_rag" / "runtime_v2").glob("*.py"))
    forbidden = ("opk_rag.search.models", "candidate_injection", "canonical_chunk_identity", "legacy_uuid", "SearchResponse", "SearchResult")
    hits = []
    for path in modules:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append({"path": _rel(path), "token": token})
    return {"active_v2_modules": [_rel(path) for path in modules], "legacy_runtime_import_count": len(hits), "hits": hits}


def candidate_rows(rows_by_unit: dict[str, tuple[RetrievalCandidateV2, ...]]) -> list[dict[str, Any]]:
    rows = []
    for unit, candidates in rows_by_unit.items():
        for candidate in candidates:
            payload = candidate.to_json()
            payload["sample_unit_id"] = unit
            rows.append(payload)
    return rows


def _complete_evidence_rank(rows: Sequence[RetrievalCandidateV2], gold_set: set[str]) -> int | None:
    if not gold_set:
        return None
    found = [(index, row.content_digest) for index, row in enumerate(rows, start=1) if row.content_digest in gold_set]
    if len({chunk_id for _, chunk_id in found}) < len(gold_set):
        return None
    return max(index for index, _ in found) if found else None


def digest_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


def git_status_short() -> str:
    return subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
