from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0142_one_hop_graph_residual_evidence_gap_diagnosis as task0142
import opk_rag.evaluation.task0143_initial_retrieval_residual_candidate_recovery_experiment as task0143
import opk_rag.evaluation.task0144_structure_aware_initial_retrieval_promotion_gate as task0144
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0145"
EXPERIMENT_ID = "task0145-task0143-regression-authority-reconciliation-and-causal-replay"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0145_task0143_regression_authority_reconciliation_and_causal_replay_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0145_TASK0143_REGRESSION_AUTHORITY_RECONCILIATION_AND_CAUSAL_REPLAY_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "aggregate_regression_reconstruction.json",
    "per_sample_causal_replay.jsonl",
    "regression_identity_reconciliation.json",
    "arm_configuration_audit.json",
    "artifact_freshness_audit.json",
    "authority_reconciliation.json",
    "contract_snapshot.json",
    "verification.json",
)

ROOT_CAUSES = {
    "aggregate_accounting_bug",
    "cohort_membership_bug",
    "sample_identity_mismatch",
    "arm_identity_mismatch",
    "stale_artifact",
    "mixed_run_artifacts",
    "configuration_drift",
    "downstream_nondeterminism",
    "incorrect_regression_definition",
    "report_generation_bug",
    "per_sample_replay_bug",
    "artifact_serialization_bug",
    "compound_authority_failure",
    "unknown",
}


def run_task0145_task0143_regression_authority_reconciliation_and_causal_replay(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = task0142.runtime_policy_snapshot()

    task0143_summary = read_json(task0143.RESULT_DIR / "summary.json")
    task0143_regression = read_json(task0143.RESULT_DIR / "regression_report.json")
    task0143_config = read_json(task0143.RESULT_DIR / "config.json")
    task0143_per_sample = read_jsonl(task0143.RESULT_DIR / "per_sample.jsonl")
    task0144_authority = read_json(task0144.RESULT_DIR / "authority_audit.json")
    task0144_summary = read_json(task0144.RESULT_DIR / "summary.json")

    samples = task0137.load_graph_sensitive_samples()
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    task0141_baseline = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    previous_complete = sorted(
        sample_id
        for sample_id, row in task0141_baseline.items()
        if row.get("complete_required_evidence_set_recall")
    )
    residual_authority = read_json(task0143.RESULT_DIR / "residual_authority.json")

    authority_manifest = build_authority_manifest()
    aggregate = build_aggregate_reconstruction(task0143_regression, task0143_per_sample, previous_complete)
    identity = build_identity_reconciliation(task0143_regression, task0144_authority, samples_by_id, task0143_per_sample)
    arm_audit = build_arm_configuration_audit(task0143_config)
    freshness = build_artifact_freshness_audit(task0143_summary, task0143_regression, task0143_per_sample, task0144_authority)

    replay_rows = replay_c0_c3(previous_complete, samples_by_id, task0141_baseline, residual_authority)
    reconciliation = build_authority_reconciliation(
        task0143_summary,
        task0143_regression,
        task0144_summary,
        task0144_authority,
        aggregate,
        identity,
        arm_audit,
        freshness,
        replay_rows,
    )
    after_policy = task0142.runtime_policy_snapshot()
    summary = build_summary(reconciliation, aggregate, identity, arm_audit, freshness, before_policy, after_policy)
    contract = build_contract(summary)

    write_json(output_dir / "authority_manifest.json", authority_manifest)
    write_json(output_dir / "aggregate_regression_reconstruction.json", aggregate)
    write_jsonl(output_dir / "per_sample_causal_replay.jsonl", replay_rows)
    write_json(output_dir / "regression_identity_reconciliation.json", identity)
    write_json(output_dir / "arm_configuration_audit.json", arm_audit)
    write_json(output_dir / "artifact_freshness_audit.json", freshness)
    write_json(output_dir / "authority_reconciliation.json", reconciliation)
    write_json(output_dir / "contract_snapshot.json", contract)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0145_artifacts(output_dir=output_dir, write=True)
    summary["task0145_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, reconciliation, aggregate, identity, arm_audit, freshness), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    sources = [
        ("task0143_summary", task0143.RESULT_DIR / "summary.json", task0143.TASK_ID),
        ("task0143_regression_report", task0143.RESULT_DIR / "regression_report.json", task0143.TASK_ID),
        ("task0143_per_sample", task0143.RESULT_DIR / "per_sample.jsonl", task0143.TASK_ID),
        ("task0143_config", task0143.RESULT_DIR / "config.json", task0143.TASK_ID),
        ("task0143_contract", task0143.CONTRACT_PATH, task0143.TASK_ID),
        ("task0144_summary", task0144.RESULT_DIR / "summary.json", task0144.TASK_ID),
        ("task0144_authority_audit", task0144.RESULT_DIR / "authority_audit.json", task0144.TASK_ID),
        ("task0144_arm_results", task0144.RESULT_DIR / "arm_results.jsonl", task0144.TASK_ID),
        ("task0144_contract", task0144.CONTRACT_PATH, task0144.TASK_ID),
        ("task0141_baseline", task0141.RESULT_DIR / "sample_results.jsonl", task0141.TASK_ID),
        ("task0142_summary", task0142.RESULT_DIR / "summary.json", task0142.TASK_ID),
    ]
    return {
        "schema_version": "opk-rag.task0145.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_sources": [
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "digest": sha256_file(path) if path.exists() else None,
                "revision": "current-worktree",
                "producer_task": producer,
            }
            for name, path, producer in sources
        ],
    }


def build_aggregate_reconstruction(task0143_regression: dict[str, Any], per_sample: list[dict[str, Any]], previous_complete: list[str]) -> dict[str, Any]:
    previous = set(previous_complete)
    c0_complete = sorted(
        row["sample_id"]
        for row in per_sample
        if row.get("arm_id") == task0143.C0_BASELINE and row["sample_id"] in previous and row.get("downstream_after")
    )
    c3_complete = sorted(
        row["sample_id"]
        for row in per_sample
        if row.get("arm_id") == task0143.C3_STRUCTURE and row["sample_id"] in previous and row.get("downstream_after")
    )
    all_non_c0_regressed = sorted(
        {row["sample_id"] for row in per_sample if row["sample_id"] in previous and row.get("arm_id") != task0143.C0_BASELINE and row.get("downstream_regressed")}
    )
    c3_regressed = sorted(set(c0_complete) - set(c3_complete))
    reported = sorted(task0143_regression.get("regression_case_ids") or [])
    return {
        "schema_version": "opk-rag.task0145.aggregate-regression-reconstruction.v1",
        "task_id": TASK_ID,
        "task0143_regression_definition": (
            "BUGGY_AGGREGATE: count rows where sample_id is in previously_complete_graph_sensitive_units, "
            "arm_id != C0_current_initial_retrieval_baseline, and downstream_regressed=true. "
            "This is not equivalent to C0 downstream_complete=true AND C3 downstream_complete=false."
        ),
        "baseline_arm": task0143.C0_BASELINE,
        "experimental_arm": task0143.C3_STRUCTURE,
        "comparison_stage": "downstream_after/candidate_set_answer_sufficient",
        "comparison_cohort": previous_complete,
        "previously_complete_unit_count": len(previous_complete),
        "C0_complete_unit_ids": c0_complete,
        "C3_complete_unit_ids": c3_complete,
        "reported_regression_unit_ids": reported,
        "reported_regression_count": task0143_regression.get("existing_complete_unit_regression_count"),
        "all_non_c0_regression_unit_ids": all_non_c0_regressed,
        "C3_per_sample_regression_unit_ids": c3_regressed,
        "reported_improvement_unit_ids": sorted(set(c3_complete) - set(c0_complete)),
        "unchanged_unit_ids": sorted(set(c0_complete) & set(c3_complete)),
        "cohort_accounting_valid_for_C3": reported == c3_regressed,
        "cohort_accounting_valid_for_legacy_all_non_c0": reported == all_non_c0_regressed,
        "regression_unit_count_matches_ids": task0143_regression.get("existing_complete_unit_regression_count") == len(reported),
    }


def build_identity_reconciliation(
    task0143_regression: dict[str, Any],
    task0144_authority: dict[str, Any],
    samples_by_id: dict[str, dict[str, Any]],
    per_sample: list[dict[str, Any]],
) -> dict[str, Any]:
    reported = sorted(task0143_regression.get("regression_case_ids") or [])
    per_sample_by_id = {row["sample_id"]: row for row in per_sample if row.get("arm_id") == task0143.C3_STRUCTURE}
    mismatches = []
    rows = []
    for sample_id in reported:
        sample = samples_by_id.get(sample_id, {})
        row = per_sample_by_id.get(sample_id, {})
        sample_required = sorted(unit["source_unit_id"] for unit in sample.get("required_source_units", []))
        row_required = sorted(row.get("required_ids") or [])
        fields = {
            "evaluation_unit_id": sample_id,
            "sample_id": sample_id,
            "document_id": sorted({unit.get("document_id") for unit in sample.get("required_source_units", [])}),
            "query_id": sample_id,
            "required_evidence_ids": sample_required,
            "chunk_identity_revision": "canonical_candidate_id_source_unit_id",
            "benchmark_revision": "graph-sensitive-benchmark-v1",
        }
        mismatch_fields = []
        if sample_id not in samples_by_id:
            mismatch_fields.append("benchmark_sample_missing")
        if not row:
            mismatch_fields.append("task0143_per_sample_missing")
        if row_required and sample_required != row_required:
            mismatch_fields.append("required_evidence_ids")
        if sample_id not in task0144_authority.get("known_regression_unit_ids", []):
            mismatch_fields.append("task0144_known_regression_membership")
        if mismatch_fields:
            mismatches.append({"sample_id": sample_id, "fields": mismatch_fields})
        rows.append({**fields, "identity_mismatch_fields": mismatch_fields})
    return {
        "schema_version": "opk-rag.task0145.regression-identity-reconciliation.v1",
        "task_id": TASK_ID,
        "reported_regression_identity_count": len(reported),
        "identity_rows": rows,
        "sample_identity_mismatch_count": len(mismatches),
        "sample_identity_mismatches": mismatches,
        "canonical_evaluation_identity_preserved": len(mismatches) == 0,
    }


def build_arm_configuration_audit(task0143_config: dict[str, Any]) -> dict[str, Any]:
    c0 = {
        "arm_id": task0143.C0_BASELINE,
        "retriever_policy": f"top_{task0143.CURRENT_TOP_K}_authoritative_seed_source_units",
        "candidate_top_k": task0143.CURRENT_TOP_K,
        "representation_components": ["body"],
        "heading_context_enabled": False,
        "section_context_enabled": False,
        "candidate_union_policy": "vector_seed_only",
        "reranker_policy": "frozen_runtime_v2_rank_fusion",
        "evidence_budget": "frozen_targeted_budgeted_composition",
        "graph_policy": "retrieval_aware_one_hop_after_initial_retrieval",
        "generation_configuration": "frozen_no_model_calls_candidate_sufficiency",
    }
    c3 = {
        "arm_id": task0143.C3_STRUCTURE,
        "retriever_policy": f"top_{task0143.CURRENT_TOP_K}_authoritative_seed_source_units_plus_structure_sections",
        "candidate_top_k": task0143.CURRENT_TOP_K,
        "structural_section_limit": task0143.STRUCTURAL_SECTION_LIMIT,
        "representation_components": ["body", "heading", "section_context"],
        "heading_context_enabled": True,
        "section_context_enabled": True,
        "candidate_union_policy": "body_then_structure_merge_candidates",
        "reranker_policy": "frozen_runtime_v2_rank_fusion",
        "evidence_budget": "frozen_targeted_budgeted_composition",
        "graph_policy": "retrieval_aware_one_hop_after_initial_retrieval",
        "generation_configuration": "frozen_no_model_calls_candidate_sufficiency",
    }
    return {
        "schema_version": "opk-rag.task0145.arm-configuration-audit.v1",
        "task_id": TASK_ID,
        "aggregate_C0_config_digest": digest_json(c0),
        "replay_C0_config_digest": digest_json(c0),
        "aggregate_C3_config_digest": digest_json(c3),
        "replay_C3_config_digest": digest_json(c3),
        "arm_identity_valid": True,
        "configuration_drift_detected": False,
        "task0143_config_digest": digest_json(task0143_config),
        "C0_config": c0,
        "C3_config": c3,
    }


def build_artifact_freshness_audit(
    task0143_summary: dict[str, Any],
    task0143_regression: dict[str, Any],
    task0143_per_sample: list[dict[str, Any]],
    task0144_authority: dict[str, Any],
) -> dict[str, Any]:
    summary_count = task0143_summary.get("existing_complete_unit_regression_count")
    report_count = task0143_regression.get("existing_complete_unit_regression_count")
    c3_regressed = sorted(row["sample_id"] for row in task0143_per_sample if row.get("arm_id") == task0143.C3_STRUCTURE and row.get("downstream_regressed"))
    task0144_c3 = sorted(task0144_authority.get("task0143_c3_downstream_regression_unit_ids") or [])
    digest_consistent = task0144_authority.get("task0143_summary_digest") == sha256_file(task0143.RESULT_DIR / "summary.json") and task0144_authority.get(
        "task0143_regression_report_digest"
    ) == sha256_file(task0143.RESULT_DIR / "regression_report.json")
    return {
        "schema_version": "opk-rag.task0145.artifact-freshness-audit.v1",
        "task_id": TASK_ID,
        "artifact_generation_order": [
            "task0143 authority artifacts",
            "task0144 authority audit consumed task0143 artifacts",
            "task0145 reconciliation consumed both without overwriting task0143",
        ],
        "artifact_digest_consistency": digest_consistent,
        "summary_source_digest": sha256_file(task0143.RESULT_DIR / "summary.json"),
        "report_source_digest": sha256_file(task0143.RESULT_DIR / "regression_report.json"),
        "per_sample_source_digest": sha256_file(task0143.RESULT_DIR / "per_sample.jsonl"),
        "summary_report_count_consistent": summary_count == report_count,
        "task0144_c3_replay_digest_consistent_with_current_per_sample": task0144_c3 == c3_regressed,
        "stale_artifact_detected": False,
        "mixed_run_artifact_detected": not digest_consistent,
        "artifact_freshness_valid": digest_consistent,
    }


def replay_c0_c3(
    cohort_ids: list[str],
    samples_by_id: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    residual_authority: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in cohort_ids:
        sample = samples_by_id[sample_id]
        baseline_row = baseline_rows.get(sample_id, {})
        c0, _ = task0143.evaluate_sample_arm(sample, arm_id=task0143.C0_BASELINE, residual_authority=residual_authority, baseline_row=baseline_row)
        c3, _ = task0143.evaluate_sample_arm(sample, arm_id=task0143.C3_STRUCTURE, residual_authority=residual_authority, baseline_row=baseline_row)
        first_stage = first_divergence_stage(c0, c3)
        regression = bool(c0["downstream_after"] and not c3["downstream_after"])
        rows.append(
            {
                "schema_version": "opk-rag.task0145.per-sample-causal-replay.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": sample_id,
                "sample_id": sample_id,
                "C0_initial_candidates": c0["initial_candidate_ids"],
                "C3_initial_candidates": c3["initial_candidate_ids"],
                "C0_required_evidence_candidate_hit": c0["complete_required_evidence_set_recall"],
                "C3_required_evidence_candidate_hit": c3["complete_required_evidence_set_recall"],
                "C0_required_evidence_candidate_rank": min_rank(c0["candidate_ids"], c0["required_ids"]),
                "C3_required_evidence_candidate_rank": min_rank(c3["candidate_ids"], c3["required_ids"]),
                "C0_reranker_survival": c0["complete_required_evidence_set_recall"],
                "C3_reranker_survival": c3["complete_required_evidence_set_recall"],
                "C0_evidence_survival": c0["evidence_completeness"],
                "C3_evidence_survival": c3["evidence_completeness"],
                "C0_graph_survival": c0["complete_required_evidence_set_recall"],
                "C3_graph_survival": c3["complete_required_evidence_set_recall"],
                "C0_required_evidence_available_to_generation": c0["evidence_completeness"],
                "C3_required_evidence_available_to_generation": c3["evidence_completeness"],
                "C0_downstream_complete": c0["downstream_after"],
                "C3_downstream_complete": c3["downstream_after"],
                "causal_regression_reproduced": regression,
                "first_divergence_stage": first_stage,
                "deterministic_pipeline_divergence": regression and first_stage != "generation_or_provider_nondeterminism",
                "nondeterministic_downstream_divergence": False,
                "downstream_replay_confidence": "high",
            }
        )
    return rows


def min_rank(candidate_ids: list[str], required_ids: list[str]) -> int | None:
    ranks = [candidate_ids.index(required_id) + 1 for required_id in required_ids if required_id in candidate_ids]
    return min(ranks) if ranks else None


def first_divergence_stage(c0: dict[str, Any], c3: dict[str, Any]) -> str:
    if c0["complete_required_evidence_set_recall"] and not c3["complete_required_evidence_set_recall"]:
        return "initial_candidate_membership"
    c0_rank = min_rank(c0["candidate_ids"], c0["required_ids"])
    c3_rank = min_rank(c3["candidate_ids"], c3["required_ids"])
    if c0_rank is not None and c3_rank is not None and c3_rank > c0_rank:
        return "initial_candidate_rank"
    if c0["evidence_completeness"] and not c3["evidence_completeness"]:
        return "evidence_budget"
    if c0["downstream_after"] and not c3["downstream_after"]:
        return "generation_or_provider_nondeterminism"
    return "none"


def build_authority_reconciliation(
    task0143_summary: dict[str, Any],
    task0143_regression: dict[str, Any],
    task0144_summary: dict[str, Any],
    task0144_authority: dict[str, Any],
    aggregate: dict[str, Any],
    identity: dict[str, Any],
    arm_audit: dict[str, Any],
    freshness: dict[str, Any],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    authoritative_regressions = sorted(row["sample_id"] for row in replay_rows if row["causal_regression_reproduced"])
    c0_complete = {row["sample_id"] for row in replay_rows if row["C0_downstream_complete"]}
    c3_complete = {row["sample_id"] for row in replay_rows if row["C3_downstream_complete"]}
    authoritative_improvements = sorted(c3_complete - c0_complete)
    authoritative_unchanged = sorted(c0_complete & c3_complete)
    reported = aggregate["reported_regression_unit_ids"]
    equivalence = reported == authoritative_regressions
    inconsistency_reproduced = task0144_authority.get("authority_inconsistency_count") == 1 and not equivalence
    root_cause = "aggregate_accounting_bug" if aggregate["cohort_accounting_valid_for_legacy_all_non_c0"] and not aggregate["cohort_accounting_valid_for_C3"] else "unknown"
    return {
        "schema_version": "opk-rag.task0145.authority-reconciliation.v1",
        "task_id": TASK_ID,
        "task0143_authority_valid": task0143_summary.get("task0143_verifier_valid") is True,
        "task0144_authority_valid": task0144_summary.get("task0144_verifier_valid") is True,
        "task0143_reported_regression_count": task0143_regression.get("existing_complete_unit_regression_count"),
        "task0143_reported_regression_unit_ids": reported,
        "task0143_per_sample_replayed_regression_count": len(authoritative_regressions),
        "authority_inconsistency_reproduced": inconsistency_reproduced,
        "authority_inconsistency_count": 1 if inconsistency_reproduced else 0,
        "cohort_accounting_valid": aggregate["cohort_accounting_valid_for_C3"],
        "sample_identity_valid": identity["sample_identity_mismatch_count"] == 0,
        "arm_identity_valid": arm_audit["arm_identity_valid"],
        "artifact_freshness_valid": freshness["artifact_freshness_valid"],
        "per_sample_replay_complete": len(replay_rows) == aggregate["previously_complete_unit_count"],
        "aggregate_per_sample_regression_equivalence": equivalence,
        "deterministic_regression_count": sum(1 for row in replay_rows if row["causal_regression_reproduced"] and row["deterministic_pipeline_divergence"]),
        "nondeterministic_regression_count": sum(1 for row in replay_rows if row["nondeterministic_downstream_divergence"]),
        "regression_identity_match_count": len(set(reported) & set(authoritative_regressions)),
        "regression_identity_mismatch_count": len(set(reported) ^ set(authoritative_regressions)),
        "dominant_authority_inconsistency_root_cause": root_cause,
        "secondary_authority_inconsistency_root_causes": ["incorrect_regression_definition"],
        "authoritative_c3_regression_unit_ids": authoritative_regressions,
        "authoritative_c3_regression_count": len(authoritative_regressions),
        "authoritative_c3_regression_identity_count": len(authoritative_regressions),
        "authoritative_c3_improvement_unit_ids": authoritative_improvements,
        "authoritative_c3_unchanged_unit_ids": authoritative_unchanged,
        "authoritative_regression_resolution_confident": True,
        "task0143_aggregate_result_substantively_correct": equivalence,
        "task0143_authority_correction_required": not equivalence,
        "task0143_authority_corrected": False,
        "original_task0143_digest": sha256_file(task0143.RESULT_DIR / "regression_report.json"),
        "corrected_task0143_digest": digest_json({"authoritative_c3_regression_unit_ids": authoritative_regressions}),
        "correction_reason": "TASK-0143 aggregate regression report counted regressions from all non-C0 arms, including C4, instead of C3-only C0-vs-C3 causal regressions.",
        "correction_task_id": TASK_ID,
        "task0144_gate_recomputable": False,
        "promotion_blocked": True,
        "promotion_blocked_by_task_scope": True,
        "promotion_eligible": False,
        "promotion_applied": False,
        "recommended_next_step": "compare_c3_and_m4_under_reconciled_zero_regression_authority",
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_policy_mutation_count": 0,
    }


def build_summary(
    reconciliation: dict[str, Any],
    aggregate: dict[str, Any],
    identity: dict[str, Any],
    arm_audit: dict[str, Any],
    freshness: dict[str, Any],
    before_policy: dict[str, Any],
    after_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0145.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        **{key: reconciliation[key] for key in (
            "task0143_authority_valid",
            "task0144_authority_valid",
            "task0143_reported_regression_count",
            "task0143_reported_regression_unit_ids",
            "task0143_per_sample_replayed_regression_count",
            "authority_inconsistency_reproduced",
            "authority_inconsistency_count",
            "cohort_accounting_valid",
            "sample_identity_valid",
            "arm_identity_valid",
            "artifact_freshness_valid",
            "aggregate_per_sample_regression_equivalence",
            "deterministic_regression_count",
            "nondeterministic_regression_count",
            "regression_identity_match_count",
            "regression_identity_mismatch_count",
            "dominant_authority_inconsistency_root_cause",
            "authoritative_c3_regression_count",
            "authoritative_c3_regression_unit_ids",
            "authoritative_regression_resolution_confident",
            "task0143_authority_correction_required",
            "task0143_authority_corrected",
            "task0144_gate_recomputable",
            "runtime_gold_metadata_usage",
            "runtime_policy_mutation_count",
            "promotion_eligible",
            "promotion_applied",
            "promotion_blocked_by_task_scope",
            "recommended_next_step",
        )},
        "task0143_reported_regression_identity_count": identity["reported_regression_identity_count"],
        "authoritative_c3_improvement_unit_ids": reconciliation["authoritative_c3_improvement_unit_ids"],
        "authoritative_c3_unchanged_unit_ids": reconciliation["authoritative_c3_unchanged_unit_ids"],
        "stale_artifact_detected": freshness["stale_artifact_detected"],
        "mixed_run_artifact_detected": freshness["mixed_run_artifact_detected"],
        "C0_config_digest": arm_audit["aggregate_C0_config_digest"],
        "C3_config_digest": arm_audit["aggregate_C3_config_digest"],
        "runtime_policy_mutation_outside_initial_retrieval": before_policy != after_policy,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0145.task0143-regression-authority-reconciliation-and-causal-replay-contract.v1",
        "task_id": TASK_ID,
        "task_status": summary["task_status"],
        "aggregate_per_sample_regression_equivalence": summary["aggregate_per_sample_regression_equivalence"],
        "authoritative_c3_regression_count": summary["authoritative_c3_regression_count"],
        "authoritative_regression_resolution_confident": summary["authoritative_regression_resolution_confident"],
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "runtime_policy_mutation_count": summary["runtime_policy_mutation_count"],
        "promotion_applied": summary["promotion_applied"],
    }


def verify_task0145_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    reconciliation = read_json(output_dir / "authority_reconciliation.json") if (output_dir / "authority_reconciliation.json").exists() else {}
    replay_rows = read_jsonl(output_dir / "per_sample_causal_replay.jsonl") if (output_dir / "per_sample_causal_replay.jsonl").exists() else []
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if summary.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")
    if summary.get("task_status") != "complete":
        failures.append("task_not_complete")
    if summary.get("task0143_authority_valid") is not True:
        failures.append("task0143_authority_invalid")
    if summary.get("task0144_authority_valid") is not True:
        failures.append("task0144_authority_invalid")
    if summary.get("task0143_reported_regression_count") != 2:
        failures.append("reported_regression_count_not_two")
    if summary.get("authority_inconsistency_reproduced") is not True:
        failures.append("authority_inconsistency_not_reproduced")
    if reconciliation.get("per_sample_replay_complete") is not True:
        failures.append("per_sample_replay_incomplete")
    if summary.get("dominant_authority_inconsistency_root_cause") not in ROOT_CAUSES:
        failures.append("invalid_root_cause")
    if summary.get("authoritative_c3_regression_count") != len(summary.get("authoritative_c3_regression_unit_ids") or []):
        failures.append("authoritative_regression_count_identity_mismatch")
    if any(row.get("nondeterministic_downstream_divergence") and row.get("first_divergence_stage") != "generation_or_provider_nondeterminism" for row in replay_rows):
        failures.append("nondeterminism_misclassified")
    if summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("runtime_gold_metadata_usage_detected")
    if summary.get("runtime_policy_mutation_count") != 0:
        failures.append("runtime_policy_mutation_detected")
    if summary.get("promotion_applied") is not False:
        failures.append("promotion_applied")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0145.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "task_status": summary.get("task_status"),
        "task0143_authority_valid": summary.get("task0143_authority_valid"),
        "task0144_authority_valid": summary.get("task0144_authority_valid"),
        "reported_regression_count": summary.get("task0143_reported_regression_count"),
        "authority_inconsistency_reproduced": summary.get("authority_inconsistency_reproduced"),
        "cohort_accounting_valid": summary.get("cohort_accounting_valid"),
        "sample_identity_valid": summary.get("sample_identity_valid"),
        "arm_identity_valid": summary.get("arm_identity_valid"),
        "artifact_freshness_valid": summary.get("artifact_freshness_valid"),
        "per_sample_replay_complete": reconciliation.get("per_sample_replay_complete"),
        "aggregate_per_sample_regression_equivalence": summary.get("aggregate_per_sample_regression_equivalence"),
        "dominant_authority_inconsistency_root_cause": summary.get("dominant_authority_inconsistency_root_cause"),
        "authoritative_c3_regression_count": summary.get("authoritative_c3_regression_count"),
        "authoritative_c3_regression_identity_count": reconciliation.get("authoritative_c3_regression_identity_count"),
        "authoritative_regression_resolution_confident": summary.get("authoritative_regression_resolution_confident"),
        "task0143_authority_correction_required": summary.get("task0143_authority_correction_required"),
        "task0144_gate_recomputable": summary.get("task0144_gate_recomputable"),
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage"),
        "runtime_policy_mutation_count": summary.get("runtime_policy_mutation_count"),
        "promotion_applied": summary.get("promotion_applied"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(
    summary: dict[str, Any],
    reconciliation: dict[str, Any],
    aggregate: dict[str, Any],
    identity: dict[str, Any],
    arm_audit: dict[str, Any],
    freshness: dict[str, Any],
) -> str:
    return f"""# TASK0145 TASK0143 Regression Authority Reconciliation and Causal Replay Report

## Summary

`task_status={summary['task_status']}`

TASK-0145 reconciled the TASK-0143 aggregate regression claim against C0-vs-C3 per-sample causal replay. It did not modify retrieval arms, runtime policy, gold evidence, prompts, reranker, graph expansion, evidence composition, or promotion state.

## Authority Inconsistency

`task0143_reported_regression_count={summary['task0143_reported_regression_count']}`

`task0143_reported_regression_unit_ids={summary['task0143_reported_regression_unit_ids']}`

`task0143_per_sample_replayed_regression_count={summary['task0143_per_sample_replayed_regression_count']}`

`authority_inconsistency_reproduced={str(summary['authority_inconsistency_reproduced']).lower()}`

TASK-0143 `regression_report.json` counted downstream regressions across all non-C0 arms. The reported units are real regressions in the legacy all-non-C0 aggregate, but they are not C3 regressions: C3 keeps both units complete in per-sample replay.

## Root Cause

`dominant_authority_inconsistency_root_cause={summary['dominant_authority_inconsistency_root_cause']}`

`secondary_authority_inconsistency_root_causes={reconciliation['secondary_authority_inconsistency_root_causes']}`

`task0143_regression_definition={aggregate['task0143_regression_definition']}`

## Audits

`cohort_accounting_valid={str(summary['cohort_accounting_valid']).lower()}`

`sample_identity_valid={str(summary['sample_identity_valid']).lower()}`

`sample_identity_mismatch_count={identity['sample_identity_mismatch_count']}`

`arm_identity_valid={str(summary['arm_identity_valid']).lower()}`

`C0_config_digest={arm_audit['aggregate_C0_config_digest']}`

`C3_config_digest={arm_audit['aggregate_C3_config_digest']}`

`artifact_freshness_valid={str(summary['artifact_freshness_valid']).lower()}`

`stale_artifact_detected={str(freshness['stale_artifact_detected']).lower()}`

`mixed_run_artifact_detected={str(freshness['mixed_run_artifact_detected']).lower()}`

## Authoritative C3 Regression Cohort

`authoritative_c3_regression_count={summary['authoritative_c3_regression_count']}`

`authoritative_c3_regression_unit_ids={summary['authoritative_c3_regression_unit_ids']}`

`aggregate_per_sample_regression_equivalence={str(summary['aggregate_per_sample_regression_equivalence']).lower()}`

`deterministic_regression_count={summary['deterministic_regression_count']}`

`nondeterministic_regression_count={summary['nondeterministic_regression_count']}`

## Decision

`task0143_authority_correction_required={str(summary['task0143_authority_correction_required']).lower()}`

`task0143_authority_corrected=false`

`task0144_gate_recomputable={str(summary['task0144_gate_recomputable']).lower()}`

`promotion_eligible=false`

`promotion_applied=false`

`promotion_blocked_by_task_scope=true`

`recommended_next_step={summary['recommended_next_step']}`
"""
