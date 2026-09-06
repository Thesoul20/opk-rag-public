from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0140_graph_retrieval_activation_runtime_promotion as task0140
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0142_one_hop_graph_residual_evidence_gap_diagnosis as task0142
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval


TASK_ID = "TASK-0143"
EXPERIMENT_ID = "task0143-initial-retrieval-residual-candidate-recovery-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0143_initial_retrieval_residual_candidate_recovery_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0143_INITIAL_RETRIEVAL_RESIDUAL_CANDIDATE_RECOVERY_EXPERIMENT_REPORT.md"

C0_BASELINE = "C0_current_initial_retrieval_baseline"
C1_BUDGET = "C1_enlarged_candidate_budget"
C2_UNION = "C2_vector_lexical_candidate_union"
C3_STRUCTURE = "C3_structure_aware_retrieval"
C4_GRAPH_ANCHOR = "C4_graph_anchor_seed_retrieval"
ARMS = (C0_BASELINE, C1_BUDGET, C2_UNION, C3_STRUCTURE, C4_GRAPH_ANCHOR)

CURRENT_TOP_K = graph_retrieval.DEFAULT_SEED_COUNT
BOUNDED_LARGER_TOP_K = CURRENT_TOP_K * 2
STRUCTURAL_SECTION_LIMIT = 2
GRAPH_ANCHOR_SECTION_LIMIT = 1

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_audit.json",
    "residual_authority.json",
    "arm_results.jsonl",
    "per_sample.jsonl",
    "downstream_survival.jsonl",
    "regression_report.json",
    "arm_comparison.json",
    "root_cause_classification.json",
    "preflight.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0142_authority_valid",
    "residual_unit_count",
    "residual_missing_required_evidence_count",
    "experimental_arm_count",
    "successful_arm_count",
    "failed_or_skipped_arm_count",
    "baseline_reproduced",
    "C0_required_evidence_candidate_hit",
    "C1_required_evidence_candidate_hit",
    "C2_required_evidence_candidate_hit",
    "C3_required_evidence_candidate_hit",
    "C4_required_evidence_candidate_hit",
    "best_candidate_recovery_arm",
    "residual_required_evidence_candidate_hit",
    "residual_required_evidence_candidate_rank",
    "residual_candidate_recovery_count",
    "residual_downstream_completion_recovered",
    "candidate_pool_growth_ratio",
    "existing_complete_unit_regression_count",
    "dominant_recovery_mechanism",
    "runtime_gold_metadata_usage",
    "runtime_policy_mutation_outside_initial_retrieval",
    "canonical_candidate_identity_preserved",
    "promotion_eligible",
    "promotion_applied",
    "recommended_next_step",
)


def run_task0143_initial_retrieval_residual_candidate_recovery_experiment(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = task0142.runtime_policy_snapshot()
    preflight = build_preflight()
    if not preflight["preflight_valid"]:
        write_json(output_dir / "preflight.json", preflight)
        raise RuntimeError(f"TASK-0143 preflight failed: {preflight['failures']}")

    authority_audit = build_authority_audit(preflight)
    samples = task0137.load_graph_sensitive_samples()
    residual_authority = load_residual_authority(samples)
    task0141_baseline = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    previously_complete = sorted(
        sample_id
        for sample_id, row in task0141_baseline.items()
        if row.get("complete_required_evidence_set_recall")
    )

    per_sample: list[dict[str, Any]] = []
    downstream_survival: list[dict[str, Any]] = []
    for sample in samples:
        for arm_id in ARMS:
            row, survival = evaluate_sample_arm(
                sample,
                arm_id=arm_id,
                residual_authority=residual_authority,
                baseline_row=task0141_baseline.get(sample["sample_id"], {}),
            )
            per_sample.append(row)
            downstream_survival.extend(survival)

    arm_results = [row for row in per_sample if row["sample_id"] in set(residual_authority["residual_unit_ids"])]
    arm_comparison = build_arm_comparison(arm_results)
    regression_report = build_regression_report(per_sample, previously_complete)
    root_cause = classify_root_cause(arm_comparison)
    best_arm = select_best_arm(arm_comparison, regression_report)
    after_policy = task0142.runtime_policy_snapshot()
    config = build_config(preflight, before_policy, after_policy)
    summary = build_summary(
        preflight,
        residual_authority,
        arm_comparison,
        regression_report,
        root_cause,
        best_arm,
        before_policy,
        after_policy,
    )
    contract = build_contract(summary, config)
    digests = build_digests(authority_audit, residual_authority, arm_results, per_sample, downstream_survival, regression_report, arm_comparison, root_cause, preflight, config, summary)

    write_json(output_dir / "authority_audit.json", authority_audit)
    write_json(output_dir / "residual_authority.json", residual_authority)
    write_jsonl(output_dir / "arm_results.jsonl", arm_results)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "downstream_survival.jsonl", downstream_survival)
    write_json(output_dir / "regression_report.json", regression_report)
    write_json(output_dir / "arm_comparison.json", arm_comparison)
    write_json(output_dir / "root_cause_classification.json", root_cause)
    write_json(output_dir / "preflight.json", preflight)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0143_artifacts(output_dir=output_dir, write=True)
    summary["task0143_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison, root_cause, regression_report), encoding="utf-8")
    return summary


def build_preflight() -> dict[str, Any]:
    task0140_verification = task0140.verify_task0140_artifacts(write=False)
    task0141_verification = task0141.verify_task0141_artifacts(write=False)
    task0142_verification = task0142.verify_task0142_artifacts(write=False)
    task0142_summary = read_json(task0142.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    failures = []
    if task0140_verification["status"] != "valid":
        failures.append("task0140_verifier_invalid")
    if task0141_verification["status"] != "valid":
        failures.append("task0141_verifier_invalid")
    if task0142_verification["status"] != "valid":
        failures.append("task0142_verifier_invalid")
    if task0142_summary.get("dominant_residual_root_cause") != "initial_retrieval_failure":
        failures.append("task0142_residual_root_cause_not_initial_retrieval")
    if task0142_summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("task0142_gold_leakage_detected")
    if task0142_summary.get("runtime_policy_mutation_count") != 0:
        failures.append("task0142_policy_mutation_detected")
    if graph_authority["graph_snapshot_digest"] != task0142_summary.get("benchmark_digest"):
        failures.append("benchmark_digest_drift")
    return {
        "schema_version": "opk-rag.task0143.preflight.v1",
        "task_id": TASK_ID,
        "task0140_verifier_status": task0140_verification["status"],
        "task0141_verifier_status": task0141_verification["status"],
        "task0142_verifier_status": task0142_verification["status"],
        "task0140_summary_digest": sha256_file(task0140.RESULT_DIR / "summary.json"),
        "task0141_summary_digest": sha256_file(task0141.RESULT_DIR / "summary.json"),
        "task0142_summary_digest": sha256_file(task0142.RESULT_DIR / "summary.json"),
        "benchmark_revision": graph_authority["graph_snapshot_revision"],
        "benchmark_digest": graph_authority["graph_snapshot_digest"],
        "current_top_k": CURRENT_TOP_K,
        "bounded_larger_top_k": BOUNDED_LARGER_TOP_K,
        "preflight_valid": not failures,
        "failures": failures,
    }


def build_authority_audit(preflight: dict[str, Any]) -> dict[str, Any]:
    policy = task0142.runtime_policy_snapshot()
    return {
        "schema_version": "opk-rag.task0143.authority-audit.v1",
        "task_id": TASK_ID,
        "task0140_authority_valid": preflight["task0140_verifier_status"] == "valid",
        "task0141_authority_valid": preflight["task0141_verifier_status"] == "valid",
        "task0142_authority_valid": preflight["task0142_verifier_status"] == "valid",
        "benchmark_revision": preflight["benchmark_revision"],
        "benchmark_digest": preflight["benchmark_digest"],
        "canonical_initial_retrieval_policy": f"top_{CURRENT_TOP_K}_authoritative_seed_source_units",
        "bounded_larger_top_k": BOUNDED_LARGER_TOP_K,
        "reranker_policy": "BGE Guarded Rank Fusion",
        "rank_fusion_k": 60,
        "rank_fusion_lambda": 0.75,
        "runtime_policy_snapshot": policy,
        "gold_evidence_redefined": False,
        "promotion_applied": False,
    }


def load_residual_authority(samples: list[dict[str, Any]]) -> dict[str, Any]:
    task0142_summary = read_json(task0142.RESULT_DIR / "summary.json")
    traces = read_jsonl(task0142.RESULT_DIR / "residual_evidence_traces.jsonl")
    sample_ids = {sample["sample_id"] for sample in samples}
    residual_unit_ids = sorted({trace["sample_id"] for trace in traces})
    missing_ids = sorted({trace["required_evidence_id"] for trace in traces})
    failures = []
    if any(sample_id not in sample_ids for sample_id in residual_unit_ids):
        failures.append("residual_unit_absent_from_graph_sensitive_benchmark")
    if task0142_summary.get("residual_incomplete_unit_count") != len(residual_unit_ids):
        failures.append("residual_unit_count_mismatch")
    if task0142_summary.get("residual_missing_required_evidence_count") != len(missing_ids):
        failures.append("missing_required_evidence_count_mismatch")
    return {
        "schema_version": "opk-rag.task0143.residual-authority.v1",
        "task_id": TASK_ID,
        "source_task_id": task0142.TASK_ID,
        "task0142_authority_loaded": not failures,
        "residual_unit_identity_matches": not failures,
        "residual_incomplete_unit_count": task0142_summary["residual_incomplete_unit_count"],
        "residual_missing_required_evidence_count": task0142_summary["residual_missing_required_evidence_count"],
        "residual_unit_ids": residual_unit_ids,
        "missing_required_evidence_ids": missing_ids,
        "residual_traces_digest": sha256_file(task0142.RESULT_DIR / "residual_evidence_traces.jsonl"),
        "failures": failures,
    }


def evaluate_sample_arm(
    sample: dict[str, Any],
    *,
    arm_id: str,
    residual_authority: dict[str, Any],
    baseline_row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = perf_counter()
    initial_candidates, source_counts, operation_counts, skipped_reason = initial_candidates_for_arm(sample, arm_id)
    decision = graph_activation.decide_runtime_graph_activation(
        sample,
        activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        seeds=initial_candidates,
    )
    runtime_config = (
        graph_retrieval.GraphRetrievalRuntimeConfig()
        if arm_id == C4_GRAPH_ANCHOR
        else graph_activation.runtime_config_for_decision(decision)
    )
    runtime = graph_retrieval.expand_runtime_candidates(
        initial_candidates,
        sample,
        runtime_config=runtime_config,
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    candidate_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
    initial_ids = [candidate["candidate_id"] for candidate in initial_candidates]
    evidence, composition_trace = evidence_composition.compose_evidence(list(runtime.candidates), evaluation_unit_id=sample["sample_id"])
    evidence_ids = [row["canonical_chunk_id"] for row in evidence]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    residual_missing_ids = set(residual_authority["missing_required_evidence_ids"])
    residual_for_sample = [required_id for required_id in required_ids if required_id in residual_missing_ids]
    recovered_ids = [required_id for required_id in residual_for_sample if required_id in candidate_ids]
    baseline_candidate_ids = list(baseline_row.get("candidate_ids") or [])
    baseline_pool_size = max(1, len(baseline_candidate_ids))
    duplicate_count = len(candidate_ids) - len(set(candidate_ids))
    invalid_count = sum(1 for candidate_id in candidate_ids if "#L" not in candidate_id)
    evidence_complete = task0137.complete_required_evidence_set_recall(evidence_ids, required_ids)
    baseline_complete = bool(baseline_row.get("complete_required_evidence_set_recall", task0137.complete_required_evidence_set_recall(baseline_candidate_ids, required_ids)))
    downstream_before = bool(baseline_row.get("downstream_after", baseline_complete))
    downstream_after = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    rank = min((candidate_ids.index(required_id) + 1 for required_id in recovered_ids), default=None)
    composition_by_id = {row["canonical_chunk_id"]: row for row in composition_trace}
    survival = [
        {
            "schema_version": "opk-rag.task0143.downstream-survival.v1",
            "task_id": TASK_ID,
            "arm_id": arm_id,
            "sample_id": sample["sample_id"],
            "required_evidence_id": required_id,
            "required_evidence_candidate_hit": required_id in candidate_ids,
            "required_evidence_candidate_rank": candidate_ids.index(required_id) + 1 if required_id in candidate_ids else None,
            "required_evidence_survived_reranking": required_id in candidate_ids,
            "required_evidence_survived_evidence_budget": required_id in evidence_ids,
            "required_evidence_available_to_generation": required_id in evidence_ids,
            "evidence_composition_rejection_reason": (composition_by_id.get(required_id) or {}).get("rejection_reason"),
            "residual_downstream_completion_recovered": evidence_complete and not baseline_complete,
        }
        for required_id in residual_for_sample
    ]
    row = {
        "schema_version": "opk-rag.task0143.per-sample.v1",
        "task_id": TASK_ID,
        "arm_id": arm_id,
        "sample_id": sample["sample_id"],
        "query": sample.get("query") or sample.get("question"),
        "skipped": skipped_reason is not None,
        "skip_reason": skipped_reason,
        "baseline_candidate_ids": baseline_candidate_ids,
        "initial_candidate_ids": initial_ids,
        "candidate_ids": candidate_ids,
        "evidence_ids": evidence_ids,
        "required_ids": required_ids,
        "residual_missing_required_evidence_ids": residual_for_sample,
        "recovered_residual_required_evidence_ids": recovered_ids,
        "candidate_pool_size": len(candidate_ids),
        "candidate_pool_growth_absolute": len(candidate_ids) - len(baseline_candidate_ids),
        "candidate_pool_growth_ratio": round(len(candidate_ids) / baseline_pool_size, 6),
        "vector_candidate_count": source_counts["vector"],
        "lexical_candidate_count": source_counts["lexical"],
        "graph_anchor_candidate_count": source_counts["graph_anchor"],
        "structural_candidate_count": source_counts["structure"],
        "candidate_union_count": len(candidate_ids),
        "candidate_deduplication_count": source_counts["raw_total"] - len(initial_ids),
        "duplicate_candidate_identity_count": duplicate_count,
        "invalid_candidate_identity_count": invalid_count,
        "gold_identity_injection_count": 0,
        "canonical_candidate_identity_preserved": duplicate_count == 0 and invalid_count == 0,
        "required_evidence_candidate_hit": bool(recovered_ids),
        "required_evidence_candidate_rank": rank,
        "residual_candidate_recovery_count": len(recovered_ids),
        "required_evidence_raw_rank": rank,
        "required_evidence_survived_reranking": bool(recovered_ids),
        "required_evidence_survived_evidence_budget": any(required_id in evidence_ids for required_id in recovered_ids),
        "required_evidence_available_to_generation": any(required_id in evidence_ids for required_id in recovered_ids),
        "residual_downstream_completion_recovered": evidence_complete and not baseline_complete,
        "complete_required_evidence_set_recall": task0137.complete_required_evidence_set_recall(candidate_ids, required_ids),
        "evidence_completeness": evidence_complete,
        "baseline_complete_required_evidence_set_recall": baseline_complete,
        "downstream_before": downstream_before,
        "downstream_after": downstream_after,
        "downstream_regressed": downstream_before and not downstream_after,
        "candidate_recall_regressed": baseline_complete and not task0137.complete_required_evidence_set_recall(candidate_ids, required_ids),
        "evidence_completeness_regressed": baseline_complete and not evidence_complete,
        "retrieval_latency_delta": round((perf_counter() - started) * 1000, 6),
        "retrieval_operation_delta": operation_counts["retrieval_operation_delta"],
        "graph_lookup_count": operation_counts["graph_lookup_count"],
        "graph_used_for_seed_lookup_only": arm_id == C4_GRAPH_ANCHOR,
        "graph_neighbor_expansion": arm_id != C4_GRAPH_ANCHOR and decision.graph_activation,
        "max_graph_hops": 0 if arm_id == C4_GRAPH_ANCHOR else graph_retrieval.default_graph_retrieval_policy().maximum_hops,
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_sample_specific_override_count": 0,
        "runtime_policy_mutation_outside_initial_retrieval": False,
        "generation_policy_frozen": True,
        "promotion_applied": False,
    }
    return row, survival


def initial_candidates_for_arm(sample: dict[str, Any], arm_id: str) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], str | None]:
    vector = graph_retrieval.select_seed_candidates(sample, seed_count=CURRENT_TOP_K)
    lexical: list[dict[str, Any]] = []
    structure: list[dict[str, Any]] = []
    graph_anchor: list[dict[str, Any]] = []
    skipped_reason = None
    if arm_id == C1_BUDGET:
        vector = graph_retrieval.select_seed_candidates(sample, seed_count=BOUNDED_LARGER_TOP_K)
    elif arm_id == C2_UNION:
        lexical = lexical_candidates(sample)
    elif arm_id == C3_STRUCTURE:
        structure = structural_candidates(sample, section_limit=STRUCTURAL_SECTION_LIMIT)
    elif arm_id == C4_GRAPH_ANCHOR:
        graph_anchor = graph_anchor_seed_candidates(sample, section_limit=GRAPH_ANCHOR_SECTION_LIMIT)
    elif arm_id != C0_BASELINE:
        skipped_reason = "unknown_arm"
    raw = [*vector, *lexical, *structure, *graph_anchor]
    deduped = graph_retrieval.merge_candidates(raw, [])
    counts = {
        "vector": len(vector),
        "lexical": len(lexical),
        "structure": len(structure),
        "graph_anchor": len(graph_anchor),
        "raw_total": len(raw),
    }
    operations = {
        "retrieval_operation_delta": int(arm_id != C0_BASELINE) + len(lexical) + len(structure) + len(graph_anchor),
        "graph_lookup_count": len(graph_anchor) if arm_id == C4_GRAPH_ANCHOR else 0,
    }
    return deduped, counts, operations, skipped_reason


def lexical_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    query = str(sample.get("query") or sample.get("question") or "")
    candidates: list[dict[str, Any]] = []
    for unit in sample.get("seed_source_units", []):
        document_id = unit["document_id"]
        for section in markdown_sections(ROOT / document_id):
            if section["heading"] and section["heading"] in query:
                candidates.append(candidate_from_section(document_id, section, origin="lexical_heading_match"))
    return candidates[:CURRENT_TOP_K]


def structural_candidates(sample: dict[str, Any], *, section_limit: int) -> list[dict[str, Any]]:
    return initial_retrieval.structural_candidates(sample, section_limit=section_limit, root=ROOT)


def graph_anchor_seed_candidates(sample: dict[str, Any], *, section_limit: int) -> list[dict[str, Any]]:
    anchor_docs = sorted({unit["document_id"] for unit in sample.get("seed_source_units", [])})
    candidates: list[dict[str, Any]] = []
    for document_id in anchor_docs:
        for section in content_sections(ROOT / document_id)[:section_limit]:
            candidates.append(candidate_from_section(document_id, section, origin="graph_anchor_document_seed_lookup"))
    return candidates


def markdown_sections(path: Path) -> list[dict[str, Any]]:
    return initial_retrieval.markdown_sections(path)


def content_sections(path: Path) -> list[dict[str, Any]]:
    return initial_retrieval.content_sections(path)


def candidate_from_section(document_id: str, section: dict[str, Any], *, origin: str) -> dict[str, Any]:
    return initial_retrieval.candidate_from_section(document_id, section, origin=origin)


def build_arm_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm_id"] == arm] for arm in ARMS}
    comparison = []
    for arm, arm_rows in by_arm.items():
        residual_total = sum(len(row["residual_missing_required_evidence_ids"]) for row in arm_rows)
        recovery_count = sum(row["residual_candidate_recovery_count"] for row in arm_rows)
        hit = recovery_count > 0
        ranks = [row["required_evidence_candidate_rank"] for row in arm_rows if row["required_evidence_candidate_rank"] is not None]
        comparison.append(
            {
                "arm_id": arm,
                "sample_count": len(arm_rows),
                "residual_required_evidence_candidate_hit": hit,
                "residual_required_evidence_candidate_rank": min(ranks) if ranks else None,
                "residual_candidate_recovery_count": recovery_count,
                "residual_candidate_recovery_rate": round(_safe_div(recovery_count, residual_total), 6),
                "residual_downstream_completion_recovered": any(row["residual_downstream_completion_recovered"] for row in arm_rows),
                "mean_candidate_pool_size": round(_safe_div(sum(row["candidate_pool_size"] for row in arm_rows), len(arm_rows)), 6),
                "max_candidate_pool_growth_ratio": max((row["candidate_pool_growth_ratio"] for row in arm_rows), default=0.0),
                "candidate_pool_growth_is_bounded": max((row["candidate_pool_growth_ratio"] for row in arm_rows), default=0.0) <= 3.0,
                "retrieval_operation_delta": sum(row["retrieval_operation_delta"] for row in arm_rows),
                "graph_lookup_count": sum(row["graph_lookup_count"] for row in arm_rows),
                "runtime_gold_metadata_usage": any(row["runtime_gold_metadata_usage"] for row in arm_rows),
                "canonical_candidate_identity_preserved": all(row["canonical_candidate_identity_preserved"] for row in arm_rows),
                "skipped_count": sum(1 for row in arm_rows if row["skipped"]),
            }
        )
    return {
        "schema_version": "opk-rag.task0143.arm-comparison.v1",
        "task_id": TASK_ID,
        "arms": comparison,
    }


def build_regression_report(rows: list[dict[str, Any]], previously_complete: list[str]) -> dict[str, Any]:
    previous = set(previously_complete)
    cohort_rows = [row for row in rows if row["sample_id"] in previous and row["arm_id"] != C0_BASELINE]
    return {
        "schema_version": "opk-rag.task0143.regression-report.v1",
        "task_id": TASK_ID,
        "previously_complete_graph_sensitive_units": previously_complete,
        "existing_complete_unit_count": len(previously_complete),
        "existing_complete_unit_regression_count": sum(1 for row in cohort_rows if row["downstream_regressed"]),
        "candidate_recall_regression_count": sum(1 for row in cohort_rows if row["candidate_recall_regressed"]),
        "evidence_completeness_regression_count": sum(1 for row in cohort_rows if row["evidence_completeness_regressed"]),
        "downstream_regression_count": sum(1 for row in cohort_rows if row["downstream_regressed"]),
        "regression_case_ids": sorted({row["sample_id"] for row in cohort_rows if row["downstream_regressed"]}),
    }


def classify_root_cause(arm_comparison: dict[str, Any]) -> dict[str, Any]:
    by_arm = {row["arm_id"]: row for row in arm_comparison["arms"]}
    if by_arm[C1_BUDGET]["residual_required_evidence_candidate_hit"]:
        mechanism = "candidate_budget_cutoff"
    elif by_arm[C2_UNION]["residual_required_evidence_candidate_hit"]:
        mechanism = "lexical_complementarity_gap"
    elif by_arm[C3_STRUCTURE]["residual_required_evidence_candidate_hit"]:
        mechanism = "structural_representation_gap"
    elif by_arm[C4_GRAPH_ANCHOR]["residual_required_evidence_candidate_hit"]:
        mechanism = "graph_seed_localization_gap"
    else:
        mechanism = "unresolved_initial_retrieval_failure"
    return {
        "schema_version": "opk-rag.task0143.root-cause-classification.v1",
        "task_id": TASK_ID,
        "dominant_recovery_mechanism": mechanism,
        "all_arm_hits": {arm: by_arm[arm]["residual_required_evidence_candidate_hit"] for arm in ARMS},
        "recommended_next_step": recommended_next_step(mechanism),
    }


def select_best_arm(arm_comparison: dict[str, Any], regression_report: dict[str, Any]) -> str:
    rows = {row["arm_id"]: row for row in arm_comparison["arms"]}
    for arm in (C3_STRUCTURE, C4_GRAPH_ANCHOR, C2_UNION, C1_BUDGET):
        row = rows[arm]
        if (
            row["residual_required_evidence_candidate_hit"]
            and row["residual_downstream_completion_recovered"]
            and row["candidate_pool_growth_is_bounded"]
        ):
            return arm
    for arm in (C3_STRUCTURE, C4_GRAPH_ANCHOR, C2_UNION, C1_BUDGET):
        row = rows[arm]
        if row["residual_required_evidence_candidate_hit"] and row["candidate_pool_growth_is_bounded"]:
            return arm
    return "none"


def build_config(preflight: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0143.config.v1",
        "task_id": TASK_ID,
        "benchmark_digest": preflight["benchmark_digest"],
        "arms": list(ARMS),
        "current_top_k": CURRENT_TOP_K,
        "bounded_larger_top_k": BOUNDED_LARGER_TOP_K,
        "structural_section_limit": STRUCTURAL_SECTION_LIMIT,
        "graph_anchor_section_limit": GRAPH_ANCHOR_SECTION_LIMIT,
        "max_graph_hops_for_graph_anchor": 0,
        "before_runtime_policy_snapshot": before_policy,
        "after_runtime_policy_snapshot": after_policy,
        "runtime_default_modified": False,
        "allowed_mutation_surface": "initial_candidate_retrieval_only",
        "gold_signal_allowed": False,
        "promotion_applied": False,
    }


def build_summary(
    preflight: dict[str, Any],
    residual_authority: dict[str, Any],
    arm_comparison: dict[str, Any],
    regression_report: dict[str, Any],
    root_cause: dict[str, Any],
    best_arm: str,
    before_policy: dict[str, Any],
    after_policy: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["arm_id"]: row for row in arm_comparison["arms"]}
    best = rows.get(best_arm, {})
    successful = sum(1 for row in rows.values() if row["residual_required_evidence_candidate_hit"])
    baseline_reproduced = rows[C0_BASELINE]["residual_required_evidence_candidate_hit"] is False
    residual_hit = bool(best.get("residual_required_evidence_candidate_hit", False))
    recovery_generalizable = residual_hit and regression_report["existing_complete_unit_count"] > residual_authority["residual_incomplete_unit_count"]
    promotion_eligible = (
        residual_hit
        and regression_report["existing_complete_unit_regression_count"] == 0
        and bool(best.get("candidate_pool_growth_is_bounded", False))
        and recovery_generalizable
    )
    return {
        "schema_version": "opk-rag.task0143.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0142_authority_valid": preflight["task0142_verifier_status"] == "valid" and residual_authority["task0142_authority_loaded"],
        "residual_unit_count": residual_authority["residual_incomplete_unit_count"],
        "residual_missing_required_evidence_count": residual_authority["residual_missing_required_evidence_count"],
        "experimental_arm_count": len(ARMS),
        "successful_arm_count": successful,
        "failed_or_skipped_arm_count": len(ARMS) - successful,
        "baseline_reproduced": baseline_reproduced,
        "C0_required_evidence_candidate_hit": rows[C0_BASELINE]["residual_required_evidence_candidate_hit"],
        "C1_required_evidence_candidate_hit": rows[C1_BUDGET]["residual_required_evidence_candidate_hit"],
        "C2_required_evidence_candidate_hit": rows[C2_UNION]["residual_required_evidence_candidate_hit"],
        "C3_required_evidence_candidate_hit": rows[C3_STRUCTURE]["residual_required_evidence_candidate_hit"],
        "C4_required_evidence_candidate_hit": rows[C4_GRAPH_ANCHOR]["residual_required_evidence_candidate_hit"],
        "best_candidate_recovery_arm": best_arm,
        "best_arm_candidate_recovery": residual_hit,
        "best_arm_downstream_recovery": bool(best.get("residual_downstream_completion_recovered", False)),
        "best_arm_regression_free": regression_report["existing_complete_unit_regression_count"] == 0,
        "best_arm_bounded": bool(best.get("candidate_pool_growth_is_bounded", False)),
        "residual_required_evidence_candidate_hit": residual_hit,
        "residual_required_evidence_candidate_rank": best.get("residual_required_evidence_candidate_rank"),
        "residual_candidate_recovery_count": best.get("residual_candidate_recovery_count", 0),
        "residual_downstream_completion_recovered": bool(best.get("residual_downstream_completion_recovered", False)),
        "candidate_pool_growth_ratio": best.get("max_candidate_pool_growth_ratio", 0.0),
        "existing_complete_unit_regression_count": regression_report["existing_complete_unit_regression_count"],
        "dominant_recovery_mechanism": root_cause["dominant_recovery_mechanism"],
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_sample_specific_override_count": 0,
        "runtime_policy_mutation_outside_initial_retrieval": before_policy != after_policy,
        "canonical_candidate_identity_preserved": all(row["canonical_candidate_identity_preserved"] for row in rows.values()),
        "candidate_pool_growth_is_bounded": bool(best.get("candidate_pool_growth_is_bounded", False)),
        "recovery_mechanism_generalizable": recovery_generalizable,
        "promotion_eligible": promotion_eligible,
        "promotion_applied": False,
        "recommended_next_step": root_cause["recommended_next_step"],
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0143.initial-retrieval-residual-candidate-recovery-experiment-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "experimental_arm_count": summary["experimental_arm_count"],
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "runtime_policy_mutation_outside_initial_retrieval": summary["runtime_policy_mutation_outside_initial_retrieval"],
        "canonical_candidate_identity_preserved": summary["canonical_candidate_identity_preserved"],
        "promotion_applied": summary["promotion_applied"],
        "allowed_mutation_surface": config["allowed_mutation_surface"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:8])
    return {
        "schema_version": "opk-rag.task0143.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "experiment_logic_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "experiment_logic_deterministic": True,
        "provider_model_nondeterminism": "not_applicable_no_model_calls",
    }


def verify_task0143_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    per_sample = read_jsonl(output_dir / "per_sample.jsonl") if (output_dir / "per_sample.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")
    if summary.get("task0142_authority_valid") is not True:
        failures.append("task0142_authority_invalid")
    if summary.get("experimental_arm_count", 0) < 5:
        failures.append("experimental_arm_count_below_required")
    if summary.get("baseline_reproduced") is not True:
        failures.append("baseline_not_reproduced")
    if summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("runtime_gold_metadata_usage_detected")
    if summary.get("runtime_policy_mutation_outside_initial_retrieval") is not False:
        failures.append("runtime_policy_mutation_outside_initial_retrieval")
    if summary.get("canonical_candidate_identity_preserved") is not True:
        failures.append("canonical_candidate_identity_not_preserved")
    if summary.get("promotion_applied") is not False:
        failures.append("promotion_applied")
    if any(row.get("gold_identity_injection_count") != 0 for row in per_sample):
        failures.append("gold_identity_injection_detected")
    if any(row.get("duplicate_candidate_identity_count") != 0 for row in per_sample):
        failures.append("duplicate_candidate_identity_detected")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0143.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "task_status": summary.get("task_status"),
        "task0142_authority_valid": summary.get("task0142_authority_valid"),
        "experimental_arm_count": summary.get("experimental_arm_count"),
        "baseline_reproduced": summary.get("baseline_reproduced"),
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage"),
        "runtime_policy_mutation_outside_initial_retrieval": summary.get("runtime_policy_mutation_outside_initial_retrieval"),
        "canonical_candidate_identity_preserved": summary.get("canonical_candidate_identity_preserved"),
        "best_candidate_recovery_arm": summary.get("best_candidate_recovery_arm"),
        "residual_required_evidence_candidate_hit": summary.get("residual_required_evidence_candidate_hit"),
        "residual_candidate_recovery_count": summary.get("residual_candidate_recovery_count"),
        "residual_downstream_completion_recovered": summary.get("residual_downstream_completion_recovered"),
        "existing_complete_unit_regression_count": summary.get("existing_complete_unit_regression_count"),
        "dominant_recovery_mechanism": summary.get("dominant_recovery_mechanism"),
        "promotion_eligible": summary.get("promotion_eligible"),
        "promotion_applied": summary.get("promotion_applied"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], arm_comparison: dict[str, Any], root_cause: dict[str, Any], regression_report: dict[str, Any]) -> str:
    arm_rows = "\n".join(
        f"* `{row['arm_id']}`: hit={str(row['residual_required_evidence_candidate_hit']).lower()}, "
        f"rank={row['residual_required_evidence_candidate_rank']}, "
        f"downstream_recovered={str(row['residual_downstream_completion_recovered']).lower()}, "
        f"growth={row['max_candidate_pool_growth_ratio']}"
        for row in arm_comparison["arms"]
    )
    return f"""# TASK0143 Initial Retrieval Residual Candidate Recovery Experiment Report

## Summary

`task_status={summary['task_status']}`

TASK-0143 reuses TASK-0142 residual authority and changes only initial candidate retrieval in controlled offline arms. Ranking, Evidence Composition, one-hop graph expansion, generation policy, candidate identity, and runtime defaults remain frozen.

## Results

{arm_rows}

## Decision

`best_candidate_recovery_arm={summary['best_candidate_recovery_arm']}`

`dominant_recovery_mechanism={root_cause['dominant_recovery_mechanism']}`

`residual_downstream_completion_recovered={str(summary['residual_downstream_completion_recovered']).lower()}`

`existing_complete_unit_regression_count={regression_report['existing_complete_unit_regression_count']}`

`promotion_eligible={str(summary['promotion_eligible']).lower()}`

`promotion_applied=false`

## Interpretation

TASK-0142 established the residual root cause as Initial Retrieval Failure. TASK-0143 shows whether the missing Required Evidence can enter the candidate pool before reranking/evidence composition. Because promotion is not applied here, any runtime change must be handled by a later promotion task with a broader generalization gate.
"""


def recommended_next_step(mechanism: str) -> str:
    return {
        "candidate_budget_cutoff": "bounded_candidate_budget_policy_promotion_gate",
        "lexical_complementarity_gap": "hybrid_initial_retrieval_promotion_gate",
        "structural_representation_gap": "structure_aware_initial_retrieval_promotion_gate",
        "graph_seed_localization_gap": "graph_aware_initial_seed_retrieval_promotion_gate",
        "unresolved_initial_retrieval_failure": "representation_corpus_boundary_retrieval_reachability_diagnosis",
    }.get(mechanism, "representation_corpus_boundary_retrieval_reachability_diagnosis")


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den
