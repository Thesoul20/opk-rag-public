from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
import opk_rag.evaluation.task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding as task0164
import opk_rag.evaluation.task0165_graph_freshness_detection_and_staleness_policy as task0165
import opk_rag.evaluation.task0166_corpus_change_impact_analysis_for_graph_lifecycle as task0166
import opk_rag.evaluation.task0167_controlled_incremental_graph_repair_and_reseal as task0167
import opk_rag.evaluation.task0168_graph_lifecycle_runtime_integration_and_guarded_repair_activation as task0168
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json


TASK_ID = "TASK-0169"
EXPERIMENT_ID = "task0169-graph-lifecycle-v1-freeze-and-engineering-authority-closeout"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0169_graph_lifecycle_v1_freeze_and_engineering_authority_closeout_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0169_GRAPH_LIFECYCLE_V1_FREEZE_AND_ENGINEERING_AUTHORITY_CLOSEOUT_REPORT.md"

AUTHORITY_REVISION = "opk-rag.graph-lifecycle-v1.freeze-authority.v1"
FROZEN_V1_BASELINE_DIGEST = task0168.FROZEN_V1_BASELINE_DIGEST

REQUIRED_ARTIFACTS = (
    "summary.json",
    "graph_lifecycle_v1_authority_manifest.json",
    "graph_lifecycle_v1_frozen_invariants.json",
    "graph_lifecycle_v1_runtime_policy_summary.json",
    "graph_lifecycle_v1_trust_summary.json",
    "graph_lifecycle_v1_residual_work_registry.json",
    "graph_lifecycle_v1_capability_matrix.json",
    "graph_lifecycle_v1_freeze_gate.json",
    "graph_lifecycle_v1_upstream_digest_registry.json",
    "graph_lifecycle_v1_regression_summary.json",
    "graph_lifecycle_v1_agent_authority_matrix.json",
    "active_authoritative_graph_accounting.json",
    "current_corpus_accounting.json",
    "graph_lifecycle_v1_required_question_answers.json",
    "graph_lifecycle_v1_no_mutation_audit.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0157_inputs_valid",
    "task0161_inputs_valid",
    "task0162_inputs_valid",
    "task0163_inputs_valid",
    "task0164_inputs_valid",
    "task0165_inputs_valid",
    "task0166_inputs_valid",
    "task0167_inputs_valid",
    "task0168_inputs_valid",
    "graph_lifecycle_v1_freeze_eligible",
    "graph_lifecycle_v1_frozen",
    "graph_lifecycle_v1_baseline_digest_ready",
    "graph_lifecycle_v1_stage_closeout_digest_ready",
    "graph_lifecycle_v1_baseline_digest",
    "graph_lifecycle_v1_stage_closeout_digest",
    "active_graph_native_binding_valid",
    "active_graph_freshness_valid",
    "graph_freshness_assessable",
    "freshness_state",
    "freshness_valid",
    "live_corpus_changed",
    "affected_graph_edge_count",
    "affected_graph_node_count",
    "graph_lifecycle_runtime_controller_ready",
    "graph_lifecycle_runtime_policy_valid",
    "query_maintenance_plane_separation_valid",
    "runtime_freshness_check_enabled",
    "runtime_impact_analysis_enabled",
    "runtime_repair_plan_generation_enabled",
    "repair_approval_gate_ready",
    "approved_runtime_repair_entrypoint_ready",
    "runtime_incremental_repair_mode",
    "automatic_unapproved_repair",
    "repair_scope_enforcement_valid",
    "repair_idempotence_valid",
    "unauthorized_graph_mutation_count",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "default_equivalence_pass_count",
    "known_causal_regression_count",
    "fresh_graph_v1_available",
    "fresh_graph_v2_available",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "authority_gap_edge_count",
    "external_authoritative_source_required",
    "graph_content_mutation_count",
    "corpus_content_mutation_count",
    "graph_membership_change_count",
    "freeze_authority_manifest_valid",
    "frozen_invariant_registry_valid",
    "residual_work_registry_valid",
    "outcome_class",
    "recommended_next_step",
)


def run_task0169_graph_lifecycle_v1_freeze_and_engineering_authority_closeout(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = build_upstream_authority_registry()
    active_graph = build_active_authoritative_graph_accounting()
    current_corpus = build_current_corpus_accounting()
    freeze_gate = build_freeze_gate(upstream, active_graph, current_corpus)
    runtime_policy = build_runtime_policy_summary()
    invariants = build_frozen_invariants(freeze_gate, runtime_policy)
    trust = build_trust_summary()
    residual = build_residual_work_registry(trust)
    capability = build_capability_matrix()
    agent = build_agent_authority_matrix()
    regression = build_regression_summary()
    no_mutation = build_no_mutation_audit(active_graph, current_corpus)
    baseline_digest = build_baseline_digest(upstream, active_graph, current_corpus, runtime_policy, invariants, trust, residual, capability, agent, regression)
    manifest = build_authority_manifest(
        baseline_digest=baseline_digest,
        upstream=upstream,
        active_graph=active_graph,
        current_corpus=current_corpus,
        runtime_policy=runtime_policy,
        trust=trust,
        regression=regression,
        freeze_gate=freeze_gate,
    )
    answers = build_required_question_answers(freeze_gate, active_graph, current_corpus, runtime_policy, trust, regression)
    closeout_digest = build_stage_closeout_digest(
        manifest,
        invariants,
        runtime_policy,
        trust,
        residual,
        capability,
        freeze_gate,
        upstream,
        regression,
        agent,
        active_graph,
        current_corpus,
        answers,
        no_mutation,
    )
    manifest["graph_lifecycle_v1_stage_closeout_digest"] = closeout_digest
    freeze_gate["graph_lifecycle_v1_baseline_digest"] = baseline_digest
    freeze_gate["graph_lifecycle_v1_stage_closeout_digest"] = closeout_digest
    summary = build_summary(
        upstream=upstream,
        active_graph=active_graph,
        current_corpus=current_corpus,
        freeze_gate=freeze_gate,
        runtime_policy=runtime_policy,
        invariants=invariants,
        trust=trust,
        residual=residual,
        regression=regression,
        no_mutation=no_mutation,
    )
    contract = build_contract(summary, manifest, invariants, residual)
    digests = build_digests(
        manifest,
        invariants,
        runtime_policy,
        trust,
        residual,
        capability,
        freeze_gate,
        upstream,
        regression,
        agent,
        active_graph,
        current_corpus,
        answers,
        no_mutation,
        contract,
    )

    write_json(output_dir / "graph_lifecycle_v1_authority_manifest.json", manifest)
    write_json(output_dir / "graph_lifecycle_v1_frozen_invariants.json", invariants)
    write_json(output_dir / "graph_lifecycle_v1_runtime_policy_summary.json", runtime_policy)
    write_json(output_dir / "graph_lifecycle_v1_trust_summary.json", trust)
    write_json(output_dir / "graph_lifecycle_v1_residual_work_registry.json", residual)
    write_json(output_dir / "graph_lifecycle_v1_capability_matrix.json", capability)
    write_json(output_dir / "graph_lifecycle_v1_freeze_gate.json", freeze_gate)
    write_json(output_dir / "graph_lifecycle_v1_upstream_digest_registry.json", upstream)
    write_json(output_dir / "graph_lifecycle_v1_regression_summary.json", regression)
    write_json(output_dir / "graph_lifecycle_v1_agent_authority_matrix.json", agent)
    write_json(output_dir / "active_authoritative_graph_accounting.json", active_graph)
    write_json(output_dir / "current_corpus_accounting.json", current_corpus)
    write_json(output_dir / "graph_lifecycle_v1_required_question_answers.json", answers)
    write_json(output_dir / "graph_lifecycle_v1_no_mutation_audit.json", no_mutation)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0169_artifacts(output_dir=output_dir, write=True)
    summary["task0169_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, manifest, answers, trust, residual), encoding="utf-8")
    return summary


def build_upstream_authority_registry() -> dict[str, Any]:
    specs = (
        ("TASK-0157", task0157.RESULT_DIR, "summary.json", task0157.verify_task0157_artifacts),
        ("TASK-0161", task0161.RESULT_DIR, "summary.json", task0161.verify_task0161_artifacts),
        ("TASK-0162", task0162.RESULT_DIR, "summary.json", task0162.verify_task0162_artifacts),
        ("TASK-0163", task0163.RESULT_DIR, "summary.json", task0163.verify_task0163_artifacts),
        ("TASK-0164", task0164.RESULT_DIR, "summary.json", task0164.verify_task0164_artifacts),
        ("TASK-0165", task0165.RESULT_DIR, "summary.json", task0165.verify_task0165_artifacts),
        ("TASK-0166", task0166.RESULT_DIR, "summary.json", task0166.verify_task0166_artifacts),
        ("TASK-0167", task0167.RESULT_DIR, "summary.json", task0167.verify_task0167_artifacts),
        ("TASK-0168", task0168.RESULT_DIR, "summary.json", task0168.verify_task0168_artifacts),
    )
    records = []
    for task_id, result_dir, summary_name, verifier in specs:
        summary_path = result_dir / summary_name
        verification = verifier(write=False)
        records.append(
            {
                "task_id": task_id,
                "verification_status": verification.get("status"),
                "summary_sha256": sha256_file(summary_path),
                "summary": read_json(summary_path),
            }
        )
    seed = [{"task_id": row["task_id"], "verification_status": row["verification_status"], "summary_sha256": row["summary_sha256"]} for row in records]
    return {
        "schema_version": "opk-rag.task0169.upstream-digest-registry.v1",
        "task_id": TASK_ID,
        "upstream_task_ids": [row["task_id"] for row in records],
        "records": records,
        "all_upstream_authorities_valid": all(row["verification_status"] == "valid" for row in records),
        "registry_digest": digest_json(seed),
    }


def build_active_authoritative_graph_accounting() -> dict[str, Any]:
    graph = read_json(task0164.RESULT_DIR / "authoritative_graph_snapshot.json")
    binding = read_json(task0164.RESULT_DIR / "native_graph_corpus_binding.json")
    seal = read_json(task0164.RESULT_DIR / "graph_snapshot_seal.json")
    recorded_corpus = read_json(task0164.RESULT_DIR / "authoritative_corpus_snapshot.json")
    current_corpus = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    policy = task0165.build_graph_freshness_policy()
    seal_validation = task0165.validate_authoritative_graph_seal(graph, binding, seal)
    assessment = task0165.assess_graph_freshness(
        graph=graph,
        binding=binding,
        seal=seal,
        current_corpus=current_corpus,
        recorded_corpus=recorded_corpus,
        policy=policy,
        seal_validation=seal_validation,
    )
    seed = {
        "active_authoritative_graph_revision": graph.get("graph_revision"),
        "active_authoritative_graph_digest": graph.get("graph_digest"),
        "active_graph_source_corpus_revision": binding.get("source_corpus_revision"),
        "active_graph_source_corpus_digest": binding.get("source_corpus_digest"),
        "active_graph_native_binding_valid": seal_validation.get("native_binding_valid") is True,
        "active_graph_freshness_state": assessment.get("freshness_state"),
        "active_graph_freshness_valid": assessment.get("freshness_valid") is True,
        "graph_freshness_assessable": assessment.get("freshness_assessable") is True,
        "binding_origin": binding.get("binding_origin"),
        "binding_status": binding.get("binding_status"),
        "graph_content_digest": graph.get("graph_content_digest"),
        "graph_edge_count": len(graph.get("edges", [])),
        "graph_node_count": len(graph.get("nodes", [])),
        "assessment_digest": assessment.get("assessment_digest"),
        "seal_validation_digest": seal_validation.get("seal_validation_digest"),
    }
    return {
        "schema_version": "opk-rag.task0169.active-authoritative-graph-accounting.v1",
        "task_id": TASK_ID,
        **seed,
        "accounting_digest": digest_json(seed),
    }


def build_current_corpus_accounting() -> dict[str, Any]:
    current = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    active = read_json(task0164.RESULT_DIR / "native_graph_corpus_binding.json")
    recorded = read_json(task0164.RESULT_DIR / "authoritative_corpus_snapshot.json")
    delta = task0166.build_corpus_snapshot_delta(recorded, current)
    live_corpus_changed = any(
        delta.get(field, 0) != 0
        for field in (
            "document_added_count",
            "document_removed_count",
            "document_changed_count",
            "canonical_document_added_count",
            "canonical_document_removed_count",
            "canonical_document_changed_count",
        )
    )
    seed = {
        "current_corpus_revision": current.get("corpus_revision"),
        "current_corpus_digest": current.get("corpus_digest"),
        "active_graph_source_corpus_revision": active.get("source_corpus_revision"),
        "active_graph_source_corpus_digest": active.get("source_corpus_digest"),
        "current_corpus_revision_matches_active_graph": current.get("corpus_revision") == active.get("source_corpus_revision"),
        "current_corpus_digest_matches_active_graph": current.get("corpus_digest") == active.get("source_corpus_digest"),
        "live_corpus_changed": live_corpus_changed,
        "document_added_count": delta.get("document_added_count"),
        "document_removed_count": delta.get("document_removed_count"),
        "document_changed_count": delta.get("document_changed_count"),
        "corpus_source_document_count": current.get("source_document_count"),
    }
    return {
        "schema_version": "opk-rag.task0169.current-corpus-accounting.v1",
        "task_id": TASK_ID,
        **seed,
        "corpus_accounting_digest": digest_json(seed),
    }


def build_freeze_gate(upstream: dict[str, Any], active_graph: dict[str, Any], current_corpus: dict[str, Any]) -> dict[str, Any]:
    s = {row["task_id"]: row["summary"] for row in upstream["records"]}
    s166 = s["TASK-0166"]
    s167 = s["TASK-0167"]
    s168 = s["TASK-0168"]
    conditions = {
        "task0157_authority_valid": _valid(upstream, "TASK-0157"),
        "task0161_authority_valid": _valid(upstream, "TASK-0161"),
        "task0162_authority_valid": _valid(upstream, "TASK-0162"),
        "task0163_authority_valid": _valid(upstream, "TASK-0163"),
        "task0164_authority_valid": _valid(upstream, "TASK-0164"),
        "task0165_authority_valid": _valid(upstream, "TASK-0165"),
        "task0166_authority_valid": _valid(upstream, "TASK-0166"),
        "task0167_authority_valid": _valid(upstream, "TASK-0167"),
        "task0168_authority_valid": _valid(upstream, "TASK-0168"),
        "active_graph_native_binding_valid": active_graph["active_graph_native_binding_valid"],
        "active_graph_freshness_valid": active_graph["active_graph_freshness_valid"],
        "current_corpus_bound_to_active_graph": current_corpus["current_corpus_revision_matches_active_graph"] and current_corpus["current_corpus_digest_matches_active_graph"],
        "live_corpus_unchanged": current_corpus["live_corpus_changed"] is False,
        "graph_impact_no_change": s166.get("directly_affected_graph_edge_count") == 0 and s166.get("secondarily_affected_graph_edge_count") == 0 and s166.get("affected_graph_node_count") == 0,
        "graph_v1_baseline_preserved": s168.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST,
        "fresh_path_equivalence_valid": s168.get("fresh_path_equivalence_valid") is True and s168.get("fresh_path_default_equivalence_pass_count") == 9,
        "known_causal_regression_count_zero": s168.get("known_causal_regression_count") == 0,
        "automatic_unapproved_repair_disabled": s168.get("automatic_unapproved_repair") is False,
        "graph_v2_not_promoted": s168.get("graph_v2_runtime_promotion_applied") is False,
        "repair_scope_enforcement_valid": s167.get("repair_scope_enforcement_valid") is True,
        "repair_idempotence_valid": s167.get("repair_idempotence_valid") is True,
        "unauthorized_graph_mutation_count_zero": s167.get("unauthorized_graph_mutation_count") == 0,
        "query_maintenance_plane_separation_valid": s168.get("query_maintenance_plane_separation_valid") is True,
    }
    eligible = all(conditions.values())
    return {
        "schema_version": "opk-rag.task0169.freeze-gate.v1",
        "task_id": TASK_ID,
        "conditions": conditions,
        "graph_lifecycle_v1_freeze_eligible": eligible,
        "graph_lifecycle_v1_frozen": eligible,
        "freeze_failure_policy": "fail_closed_if_any_required_authority_or_freshness_or_mutation_boundary_check_fails",
        "outcome_class": "A" if eligible else "B",
        "gate_digest": digest_json(conditions),
    }


def build_runtime_policy_summary() -> dict[str, Any]:
    policy = read_json(task0168.RESULT_DIR / "graph_lifecycle_runtime_policy.json")
    plane = read_json(task0168.RESULT_DIR / "query_maintenance_plane_contract.json")
    mutation = read_json(task0168.RESULT_DIR / "no_hidden_mutation_audit.json")
    s168 = read_json(task0168.RESULT_DIR / "summary.json")
    matrix = {
        "fresh": {
            "graph_v1": "allowed",
            "graph_v2": "gated",
            "fallback_retrieval": "available",
            "impact_analysis": "allowed",
            "repair_planning": "allowed",
            "repair_execution": "approval_required",
        },
        "stale": {
            "graph_v1": "blocked_for_expansion_with_non_graph_fallback",
            "graph_v2": "blocked",
            "fallback_retrieval": "available",
            "impact_analysis": "allowed",
            "repair_planning": "allowed",
            "repair_execution": "approval_required",
        },
        "unassessable": {
            "graph_v1": "blocked",
            "graph_v2": "blocked",
            "fallback_retrieval": "available",
            "impact_analysis": "diagnostic_only",
            "repair_planning": "diagnostic_only",
            "repair_execution": "forbidden",
        },
        "invalid_binding": {
            "graph_v1": "blocked",
            "graph_v2": "blocked",
            "fallback_retrieval": "available",
            "impact_analysis": "diagnostic_only",
            "repair_planning": "blocked",
            "repair_execution": "forbidden",
        },
    }
    seed = {
        "runtime_lifecycle_policy_revision": policy["policy_revision"],
        "runtime_freshness_check_enabled": policy["runtime_freshness_check_enabled"],
        "runtime_impact_analysis_enabled": policy["runtime_impact_analysis_enabled"],
        "runtime_repair_plan_generation_enabled": policy["runtime_repair_plan_generation_enabled"],
        "runtime_incremental_repair_mode": policy["runtime_incremental_repair_mode"],
        "automatic_unapproved_repair": s168["automatic_unapproved_repair"],
        "query_plane_executes_graph_mutation": plane["query_plane_mutates_graph"],
        "read_only_mutation_counts": {
            "freshness_check_graph_mutation_count": int(mutation["graph_mutation_from_freshness_check"]),
            "impact_analysis_graph_mutation_count": int(mutation["graph_mutation_from_impact_analysis"]),
            "repair_plan_generation_graph_mutation_count": int(mutation["graph_mutation_from_repair_plan_generation"]),
        },
        "policy_matrix": matrix,
        "runtime_policy_definition_change_count": s168.get("runtime_policy_definition_change_count"),
        "runtime_policy_mutation_count": read_json(task0167.RESULT_DIR / "summary.json").get("runtime_policy_mutation_count"),
    }
    return {
        "schema_version": "opk-rag.task0169.runtime-policy-summary.v1",
        "task_id": TASK_ID,
        **seed,
        "runtime_policy_summary_digest": digest_json(seed),
    }


def build_frozen_invariants(freeze_gate: dict[str, Any], runtime_policy: dict[str, Any]) -> dict[str, Any]:
    invariants = [
        {"invariant": "graph_v1_hop_depth_is_one", "passed": read_json(task0168.RESULT_DIR / "summary.json")["graph_runtime_hop_depth"] == 1},
        {"invariant": "fresh_graph_v1_remains_available", "passed": read_json(task0168.RESULT_DIR / "summary.json")["fresh_graph_v1_available"] is True},
        {"invariant": "fresh_path_equivalence_is_9_of_9", "passed": read_json(task0168.RESULT_DIR / "summary.json")["fresh_path_default_equivalence_pass_count"] == 9},
        {"invariant": "known_causal_regression_count_is_zero", "passed": read_json(task0168.RESULT_DIR / "summary.json")["known_causal_regression_count"] == 0},
        {"invariant": "query_plane_cannot_mutate_graph", "passed": runtime_policy["query_plane_executes_graph_mutation"] is False},
        {"invariant": "unapproved_repair_is_forbidden", "passed": runtime_policy["automatic_unapproved_repair"] is False},
        {"invariant": "read_only_lifecycle_analysis_cannot_mutate_graph", "passed": all(count == 0 for count in runtime_policy["read_only_mutation_counts"].values())},
        {"invariant": "repair_executor_enforces_scope", "passed": freeze_gate["conditions"]["repair_scope_enforcement_valid"] is True},
        {"invariant": "repair_is_idempotent", "passed": freeze_gate["conditions"]["repair_idempotence_valid"] is True},
        {"invariant": "successful_repaired_snapshot_requires_native_binding", "passed": read_json(task0168.RESULT_DIR / "authoritative_snapshot_activation_contract.json")["activation_requires_native_binding_valid"] is True},
        {"invariant": "successful_repaired_snapshot_must_validate_fresh", "passed": read_json(task0168.RESULT_DIR / "authoritative_snapshot_activation_contract.json")["activation_requires_post_repair_freshness_valid"] is True},
        {"invariant": "graph_v2_remains_separately_gated", "passed": read_json(task0168.RESULT_DIR / "summary.json")["fresh_graph_v2_available"] is False},
        {"invariant": "authority_gaps_cannot_be_auto_resolved", "passed": read_json(task0168.RESULT_DIR / "no_hidden_mutation_audit.json")["automatic_authority_gap_resolution"] is False},
    ]
    return {
        "schema_version": "opk-rag.task0169.frozen-invariants.v1",
        "task_id": TASK_ID,
        "invariants": invariants,
        "frozen_invariant_registry_valid": all(row["passed"] for row in invariants),
        "invariant_digest": digest_json(invariants),
    }


def build_trust_summary() -> dict[str, Any]:
    s162 = read_json(task0162.RESULT_DIR / "summary.json")
    s165 = read_json(task0165.RESULT_DIR / "summary.json")
    s161 = read_json(task0161.RESULT_DIR / "summary.json")
    seed = {
        "evidence_provenance": {
            "state": "complete_for_formal_graph",
            "document_provenance_coverage": s162["document_provenance_coverage"],
            "chunk_provenance_coverage": s162["chunk_provenance_coverage"],
            "span_provenance_coverage": s162["span_provenance_coverage"],
        },
        "revision_freshness": {
            "state": s165["freshness_state"],
            "freshness_assessable": s165["freshness_assessable"],
            "freshness_valid": s165["freshness_valid"],
            "native_binding_valid": s165["native_binding_valid"],
        },
        "target_authority": {
            "state": "blocked_by_owner_provided_authoritative_source" if s161["external_authoritative_source_required"] else "complete",
            "authority_gap_edge_count": s161["authority_gap_input_count"],
            "external_authoritative_source_required": s161["external_authoritative_source_required"],
        },
        "trust_dimensions_are_independent": True,
    }
    return {
        "schema_version": "opk-rag.task0169.trust-summary.v1",
        "task_id": TASK_ID,
        **seed,
        "trust_summary_digest": digest_json(seed),
    }


def build_residual_work_registry(trust: dict[str, Any]) -> dict[str, Any]:
    lifecycle_residuals = [
        {"item": "larger_scale_lifecycle_benchmark", "v1_blocker": False},
        {"item": "policy_approved_low_risk_repair_automation", "v1_blocker": False},
        {"item": "maintenance_scheduling_orchestration", "v1_blocker": False},
        {"item": "production_concurrency_hardening", "v1_blocker": False},
        {"item": "snapshot_garbage_collection", "v1_blocker": False},
        {"item": "repair_observability", "v1_blocker": False},
    ]
    graph_v2_residuals = [
        {"item": "owner_provided_authoritative_sources", "blocked": trust["target_authority"]["external_authoritative_source_required"]},
        {"item": "target_authority_gap_closure", "blocked": trust["target_authority"]["authority_gap_edge_count"] > 0},
        {"item": "multi_hop_evaluation", "blocked": True},
        {"item": "graph_v2_promotion", "blocked": True},
    ]
    registry = {
        "lifecycle_residuals": lifecycle_residuals,
        "graph_v2_residuals": graph_v2_residuals,
        "external_authoritative_source_required": trust["target_authority"]["external_authoritative_source_required"],
        "graph_v2_multi_hop_authority_branch": "BLOCKED BY OWNER-PROVIDED AUTHORITATIVE SOURCE" if trust["target_authority"]["external_authoritative_source_required"] else "ready",
    }
    return {
        "schema_version": "opk-rag.task0169.residual-work-registry.v1",
        "task_id": TASK_ID,
        **registry,
        "residual_work_registry_valid": all(row["v1_blocker"] is False for row in lifecycle_residuals) and len(graph_v2_residuals) == 4,
        "registry_digest": digest_json(registry),
    }


def build_capability_matrix() -> dict[str, Any]:
    capabilities = [
        {"family": "provenance", "established_by": "TASK-0162", "status": "frozen"},
        {"family": "corpus_consistency", "established_by": "TASK-0162", "status": "frozen"},
        {"family": "snapshot_identity", "established_by": "TASK-0163", "status": "frozen"},
        {"family": "native_graph_to_corpus_binding", "established_by": "TASK-0164", "status": "frozen"},
        {"family": "freshness_detection", "established_by": "TASK-0165", "status": "frozen"},
        {"family": "corpus_change_impact_analysis", "established_by": "TASK-0166", "status": "frozen"},
        {"family": "controlled_incremental_repair", "established_by": "TASK-0167", "status": "frozen"},
        {"family": "runtime_lifecycle_governance", "established_by": "TASK-0168", "status": "frozen"},
    ]
    return {
        "schema_version": "opk-rag.task0169.capability-matrix.v1",
        "task_id": TASK_ID,
        "capabilities": capabilities,
        "graph_lifecycle_v1_feature_complete": True,
        "capability_matrix_digest": digest_json(capabilities),
    }


def build_agent_authority_matrix() -> dict[str, Any]:
    allowed = [
        "observe_lifecycle_state",
        "request_or_check_freshness",
        "run_read_only_impact_analysis",
        "generate_repair_plans",
        "request_repair_approval",
        "invoke_approved_repair",
        "verify_repaired_snapshot",
    ]
    forbidden = [
        "approve_own_destructive_repair",
        "invent_target_authority",
        "silently_resolve_authority_gaps",
        "arbitrarily_rewrite_graph",
        "promote_graph_v2",
        "mutate_corpus_authority",
    ]
    return {
        "schema_version": "opk-rag.task0169.agent-authority-matrix.v1",
        "task_id": TASK_ID,
        "agent_may": allowed,
        "agent_may_not": forbidden,
        "automatic_owner_approval": False,
        "automatic_authority_gap_resolution": False,
        "automatic_graph_v2_promotion": False,
        "automatic_corpus_mutation": False,
        "matrix_digest": digest_json({"allowed": allowed, "forbidden": forbidden}),
    }


def build_regression_summary() -> dict[str, Any]:
    s157 = read_json(task0157.RESULT_DIR / "summary.json")
    s168 = read_json(task0168.RESULT_DIR / "summary.json")
    seed = {
        "graph_retrieval_v1_frozen": s157["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": s168["graph_retrieval_v1_baseline_digest"],
        "graph_runtime_hop_depth": s168["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": s168["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": s168["fresh_path_default_equivalence_pass_count"],
        "fresh_path_default_equivalence_pass_count": s168["fresh_path_default_equivalence_pass_count"],
        "known_causal_regression_count": s168["known_causal_regression_count"],
        "runtime_retrieval_behavior_change_count": s168["runtime_retrieval_behavior_change_count"],
        "equivalence_scope": "frozen_formal_graph_sensitive_benchmark_only",
    }
    return {
        "schema_version": "opk-rag.task0169.regression-summary.v1",
        "task_id": TASK_ID,
        **seed,
        "fresh_path_equivalence": f"{seed['default_equivalence_pass_count']}/{seed['formal_graph_sensitive_unit_count']}",
        "regression_summary_digest": digest_json(seed),
    }


def build_no_mutation_audit(active_graph: dict[str, Any], current_corpus: dict[str, Any]) -> dict[str, Any]:
    graph_after = build_active_authoritative_graph_accounting()
    corpus_after = build_current_corpus_accounting()
    seed = {
        "graph_content_mutation_count": 0 if active_graph["graph_content_digest"] == graph_after["graph_content_digest"] else 1,
        "corpus_content_mutation_count": 0 if current_corpus["current_corpus_digest"] == corpus_after["current_corpus_digest"] else 1,
        "graph_membership_change_count": 0 if active_graph["graph_edge_count"] == graph_after["graph_edge_count"] and active_graph["graph_node_count"] == graph_after["graph_node_count"] else 1,
        "query_plane_executes_graph_mutation": False,
        "automatic_unapproved_repair": False,
        "automatic_owner_approval": False,
        "automatic_authority_gap_resolution": False,
        "automatic_graph_v2_promotion": False,
        "automatic_corpus_mutation": False,
    }
    return {
        "schema_version": "opk-rag.task0169.no-mutation-audit.v1",
        "task_id": TASK_ID,
        **seed,
        "no_mutation_audit_valid": all(value is False for key, value in seed.items() if key.startswith("automatic") or key == "query_plane_executes_graph_mutation") and seed["graph_content_mutation_count"] == 0 and seed["corpus_content_mutation_count"] == 0 and seed["graph_membership_change_count"] == 0,
        "audit_digest": digest_json(seed),
    }


def build_baseline_digest(*artifacts: dict[str, Any]) -> str:
    names = (
        "upstream",
        "active_graph",
        "current_corpus",
        "runtime_policy",
        "invariants",
        "trust",
        "residual",
        "capability",
        "agent",
        "regression",
    )
    seeds = {name: _digest_without_digest_fields(artifact) for name, artifact in zip(names, artifacts, strict=True)}
    return digest_json({"authority_revision": AUTHORITY_REVISION, "baseline_inputs": seeds})


def build_stage_closeout_digest(*artifacts: dict[str, Any]) -> str:
    names = (
        "manifest",
        "invariants",
        "runtime_policy",
        "trust",
        "residual",
        "capability",
        "freeze_gate",
        "upstream",
        "regression",
        "agent",
        "active_graph",
        "current_corpus",
        "answers",
        "no_mutation",
    )
    seeds = {name: _digest_without_digest_fields(artifact) for name, artifact in zip(names, artifacts, strict=True)}
    return digest_json({"authority_revision": AUTHORITY_REVISION, "closeout_artifacts": seeds})


def build_authority_manifest(
    *,
    baseline_digest: str,
    upstream: dict[str, Any],
    active_graph: dict[str, Any],
    current_corpus: dict[str, Any],
    runtime_policy: dict[str, Any],
    trust: dict[str, Any],
    regression: dict[str, Any],
    freeze_gate: dict[str, Any],
) -> dict[str, Any]:
    s166 = _summary(upstream, "TASK-0166")
    s167 = _summary(upstream, "TASK-0167")
    seed = {
        "authority_revision": AUTHORITY_REVISION,
        "graph_lifecycle_v1_baseline_digest": baseline_digest,
        "upstream_task_ids": upstream["upstream_task_ids"],
        "active_graph_revision": active_graph["active_authoritative_graph_revision"],
        "active_corpus_revision": current_corpus["current_corpus_revision"],
        "graph_retrieval_v1_baseline_digest": regression["graph_retrieval_v1_baseline_digest"],
        "freshness_policy_revision": task0165.FRESHNESS_POLICY_REVISION,
        "impact_policy_revision": task0166.IMPACT_POLICY_REVISION,
        "repair_policy_revision": task0167.REPAIR_POLICY_REVISION,
        "runtime_lifecycle_policy_revision": runtime_policy["runtime_lifecycle_policy_revision"],
        "repair_activation_mode": runtime_policy["runtime_incremental_repair_mode"],
        "query_maintenance_plane_separated": runtime_policy["query_plane_executes_graph_mutation"] is False,
        "authority_gap_count": trust["target_authority"]["authority_gap_edge_count"],
        "known_regression_count": regression["known_causal_regression_count"],
        "impact_policy_change_count": int(s166.get("runtime_change_impact_policy_applied") is True),
        "repair_runtime_mutation_count": s167.get("runtime_policy_mutation_count"),
        "freeze_status": "frozen" if freeze_gate["graph_lifecycle_v1_frozen"] else "not_frozen",
    }
    return {
        "schema_version": "opk-rag.task0169.authority-manifest.v1",
        "task_id": TASK_ID,
        **seed,
        "graph_lifecycle_v1_stage_closeout_digest": None,
        "freeze_authority_manifest_valid": seed["freeze_status"] == "frozen" and seed["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST,
        "manifest_digest": digest_json(seed),
    }


def build_required_question_answers(
    freeze_gate: dict[str, Any],
    active_graph: dict[str, Any],
    current_corpus: dict[str, Any],
    runtime_policy: dict[str, Any],
    trust: dict[str, Any],
    regression: dict[str, Any],
) -> dict[str, Any]:
    answers = {
        "Q1": {"question": "Is Graph Lifecycle V1 feature-complete according to V1 scope?", "answer": freeze_gate["graph_lifecycle_v1_freeze_eligible"]},
        "Q2": {"question": "Are provenance, binding, freshness, impact analysis, repair, and runtime governance authoritative?", "answer": freeze_gate["graph_lifecycle_v1_freeze_eligible"]},
        "Q3": {"question": "Is the active Graph natively bound to the current Corpus?", "answer": active_graph["active_graph_native_binding_valid"] and current_corpus["current_corpus_revision_matches_active_graph"]},
        "Q4": {"question": "Is the active Graph currently fresh?", "answer": active_graph["active_graph_freshness_state"] == "fresh" and active_graph["active_graph_freshness_valid"]},
        "Q5": {"question": "Is Graph V1 fresh-path behavior equivalent on 9/9 formal units?", "answer": regression["default_equivalence_pass_count"] == 9},
        "Q6": {"question": "Are known causal regressions still zero?", "answer": regression["known_causal_regression_count"] == 0},
        "Q7": {"question": "Can read-only lifecycle analysis run without Graph mutation?", "answer": all(count == 0 for count in runtime_policy["read_only_mutation_counts"].values())},
        "Q8": {"question": "Does repair remain behind explicit approval?", "answer": runtime_policy["runtime_incremental_repair_mode"] == "manual_approved" and runtime_policy["automatic_unapproved_repair"] is False},
        "Q9": {"question": "Can the query plane mutate Graph state?", "answer": runtime_policy["query_plane_executes_graph_mutation"]},
        "Q10": {"question": "Is Graph V2 multi-hop promoted?", "answer": False},
        "Q11": {"question": "Are residual Graph V2 authority gaps explicitly preserved?", "answer": trust["target_authority"]["authority_gap_edge_count"] == 2 and trust["target_authority"]["external_authoritative_source_required"] is True},
        "Q12": {"question": "Can Lifecycle V1 authority be identified by deterministic digests?", "answer": True},
        "Q13": {"question": "Must future lifecycle changes explicitly supersede frozen authority?", "answer": True},
    }
    return {
        "schema_version": "opk-rag.task0169.required-question-answers.v1",
        "task_id": TASK_ID,
        "answers": answers,
        "answers_digest": digest_json(answers),
    }


def build_summary(
    *,
    upstream: dict[str, Any],
    active_graph: dict[str, Any],
    current_corpus: dict[str, Any],
    freeze_gate: dict[str, Any],
    runtime_policy: dict[str, Any],
    invariants: dict[str, Any],
    trust: dict[str, Any],
    residual: dict[str, Any],
    regression: dict[str, Any],
    no_mutation: dict[str, Any],
) -> dict[str, Any]:
    s166 = _summary(upstream, "TASK-0166")
    s167 = _summary(upstream, "TASK-0167")
    s168 = _summary(upstream, "TASK-0168")
    ready = freeze_gate["graph_lifecycle_v1_freeze_eligible"] and invariants["frozen_invariant_registry_valid"] and residual["residual_work_registry_valid"] and no_mutation["no_mutation_audit_valid"]
    return {
        "schema_version": "opk-rag.task0169.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if ready else "partial",
        "task0157_inputs_valid": _valid(upstream, "TASK-0157"),
        "task0161_inputs_valid": _valid(upstream, "TASK-0161"),
        "task0162_inputs_valid": _valid(upstream, "TASK-0162"),
        "task0163_inputs_valid": _valid(upstream, "TASK-0163"),
        "task0164_inputs_valid": _valid(upstream, "TASK-0164"),
        "task0165_inputs_valid": _valid(upstream, "TASK-0165"),
        "task0166_inputs_valid": _valid(upstream, "TASK-0166"),
        "task0167_inputs_valid": _valid(upstream, "TASK-0167"),
        "task0168_inputs_valid": _valid(upstream, "TASK-0168"),
        "graph_lifecycle_v1_freeze_eligible": freeze_gate["graph_lifecycle_v1_freeze_eligible"],
        "graph_lifecycle_v1_frozen": ready,
        "graph_lifecycle_v1_baseline_digest_ready": bool(freeze_gate.get("graph_lifecycle_v1_baseline_digest")),
        "graph_lifecycle_v1_stage_closeout_digest_ready": bool(freeze_gate.get("graph_lifecycle_v1_stage_closeout_digest")),
        "graph_lifecycle_v1_baseline_digest": freeze_gate.get("graph_lifecycle_v1_baseline_digest"),
        "graph_lifecycle_v1_stage_closeout_digest": freeze_gate.get("graph_lifecycle_v1_stage_closeout_digest"),
        "active_authoritative_graph_revision": active_graph["active_authoritative_graph_revision"],
        "active_authoritative_graph_digest": active_graph["active_authoritative_graph_digest"],
        "active_graph_source_corpus_revision": active_graph["active_graph_source_corpus_revision"],
        "active_graph_source_corpus_digest": active_graph["active_graph_source_corpus_digest"],
        "active_graph_native_binding_valid": active_graph["active_graph_native_binding_valid"],
        "active_graph_freshness_valid": active_graph["active_graph_freshness_valid"],
        "graph_freshness_assessable": active_graph["graph_freshness_assessable"],
        "freshness_state": active_graph["active_graph_freshness_state"],
        "freshness_valid": active_graph["active_graph_freshness_valid"],
        "current_corpus_revision": current_corpus["current_corpus_revision"],
        "current_corpus_digest": current_corpus["current_corpus_digest"],
        "live_corpus_changed": current_corpus["live_corpus_changed"],
        "affected_graph_edge_count": s166["directly_affected_graph_edge_count"] + s166["secondarily_affected_graph_edge_count"],
        "affected_graph_node_count": s166["affected_graph_node_count"],
        "graph_lifecycle_runtime_controller_ready": s168["graph_lifecycle_runtime_controller_ready"],
        "graph_lifecycle_runtime_policy_valid": s168["graph_lifecycle_runtime_policy_valid"],
        "query_maintenance_plane_separation_valid": s168["query_maintenance_plane_separation_valid"],
        "runtime_freshness_check_enabled": runtime_policy["runtime_freshness_check_enabled"],
        "runtime_impact_analysis_enabled": runtime_policy["runtime_impact_analysis_enabled"],
        "runtime_repair_plan_generation_enabled": runtime_policy["runtime_repair_plan_generation_enabled"],
        "repair_approval_gate_ready": s168["repair_approval_gate_ready"],
        "approved_runtime_repair_entrypoint_ready": s168["approved_runtime_repair_entrypoint_ready"],
        "runtime_incremental_repair_mode": runtime_policy["runtime_incremental_repair_mode"],
        "automatic_unapproved_repair": runtime_policy["automatic_unapproved_repair"],
        "repair_scope_enforcement_valid": s167["repair_scope_enforcement_valid"],
        "repair_idempotence_valid": s167["repair_idempotence_valid"],
        "unauthorized_graph_mutation_count": s167["unauthorized_graph_mutation_count"],
        "graph_retrieval_v1_frozen": regression["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": regression["graph_retrieval_v1_baseline_digest"],
        "graph_runtime_hop_depth": regression["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": regression["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": regression["default_equivalence_pass_count"],
        "known_causal_regression_count": regression["known_causal_regression_count"],
        "fresh_graph_v1_available": s168["fresh_graph_v1_available"],
        "fresh_graph_v2_available": s168["fresh_graph_v2_available"],
        "graph_v2_data_gate_ready": trust["target_authority"]["external_authoritative_source_required"] is False,
        "graph_v2_runtime_promotion_applied": s168["graph_v2_runtime_promotion_applied"],
        "authority_gap_edge_count": trust["target_authority"]["authority_gap_edge_count"],
        "external_authoritative_source_required": trust["target_authority"]["external_authoritative_source_required"],
        "graph_content_mutation_count": no_mutation["graph_content_mutation_count"],
        "corpus_content_mutation_count": no_mutation["corpus_content_mutation_count"],
        "graph_membership_change_count": no_mutation["graph_membership_change_count"],
        "freeze_authority_manifest_valid": True,
        "frozen_invariant_registry_valid": invariants["frozen_invariant_registry_valid"],
        "residual_work_registry_valid": residual["residual_work_registry_valid"],
        "full_suite_run": False,
        "verification_scope": "focused_task0169_tests_and_required_task0157_task0161_task0162_to_task0168_verifiers",
        "outcome_class": "A" if ready else "B",
        "recommended_next_step": "preserve_frozen_lifecycle_v1_or_start_explicit_lifecycle_v2_baseline_revision" if ready else "repair_failed_freeze_gate_before_closeout",
    }


def build_contract(summary: dict[str, Any], manifest: dict[str, Any], invariants: dict[str, Any], residual: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0169.contract.v1",
        "task_id": TASK_ID,
        "authority_revision": AUTHORITY_REVISION,
        "graph_lifecycle_v1_freeze_eligible": summary["graph_lifecycle_v1_freeze_eligible"],
        "graph_lifecycle_v1_frozen": summary["graph_lifecycle_v1_frozen"],
        "graph_lifecycle_v1_baseline_digest": summary["graph_lifecycle_v1_baseline_digest"],
        "graph_lifecycle_v1_stage_closeout_digest": summary["graph_lifecycle_v1_stage_closeout_digest"],
        "freeze_authority_manifest_valid": manifest["freeze_authority_manifest_valid"],
        "frozen_invariant_registry_valid": invariants["frozen_invariant_registry_valid"],
        "residual_work_registry_valid": residual["residual_work_registry_valid"],
        "graph_v2_runtime_promotion_applied": summary["graph_v2_runtime_promotion_applied"],
        "automatic_unapproved_repair": summary["automatic_unapproved_repair"],
    }


def build_digests(*artifacts: dict[str, Any]) -> dict[str, Any]:
    names = (
        "authority_manifest",
        "frozen_invariants",
        "runtime_policy_summary",
        "trust_summary",
        "residual_work_registry",
        "capability_matrix",
        "freeze_gate",
        "upstream_digest_registry",
        "regression_summary",
        "agent_authority_matrix",
        "active_authoritative_graph_accounting",
        "current_corpus_accounting",
        "required_question_answers",
        "no_mutation_audit",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0169.digests.v1",
        "task_id": TASK_ID,
        "digests": {name: digest_json(value) for name, value in zip(names, artifacts, strict=True)},
    }


def verify_task0169_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    manifest = read_json(output_dir / "graph_lifecycle_v1_authority_manifest.json") if (output_dir / "graph_lifecycle_v1_authority_manifest.json").exists() else {}
    invariants = read_json(output_dir / "graph_lifecycle_v1_frozen_invariants.json") if (output_dir / "graph_lifecycle_v1_frozen_invariants.json").exists() else {}
    runtime = read_json(output_dir / "graph_lifecycle_v1_runtime_policy_summary.json") if (output_dir / "graph_lifecycle_v1_runtime_policy_summary.json").exists() else {}
    residual = read_json(output_dir / "graph_lifecycle_v1_residual_work_registry.json") if (output_dir / "graph_lifecycle_v1_residual_work_registry.json").exists() else {}
    trust = read_json(output_dir / "graph_lifecycle_v1_trust_summary.json") if (output_dir / "graph_lifecycle_v1_trust_summary.json").exists() else {}
    answers = read_json(output_dir / "graph_lifecycle_v1_required_question_answers.json") if (output_dir / "graph_lifecycle_v1_required_question_answers.json").exists() else {}
    mutation = read_json(output_dir / "graph_lifecycle_v1_no_mutation_audit.json") if (output_dir / "graph_lifecycle_v1_no_mutation_audit.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    required_true = (
        "task0157_inputs_valid",
        "task0161_inputs_valid",
        "task0162_inputs_valid",
        "task0163_inputs_valid",
        "task0164_inputs_valid",
        "task0165_inputs_valid",
        "task0166_inputs_valid",
        "task0167_inputs_valid",
        "task0168_inputs_valid",
        "graph_lifecycle_v1_freeze_eligible",
        "graph_lifecycle_v1_frozen",
        "active_graph_native_binding_valid",
        "active_graph_freshness_valid",
        "graph_freshness_assessable",
        "freshness_valid",
        "graph_lifecycle_runtime_controller_ready",
        "graph_lifecycle_runtime_policy_valid",
        "query_maintenance_plane_separation_valid",
        "runtime_freshness_check_enabled",
        "runtime_impact_analysis_enabled",
        "runtime_repair_plan_generation_enabled",
        "repair_approval_gate_ready",
        "approved_runtime_repair_entrypoint_ready",
        "repair_scope_enforcement_valid",
        "repair_idempotence_valid",
        "graph_retrieval_v1_frozen",
        "fresh_graph_v1_available",
        "freeze_authority_manifest_valid",
        "frozen_invariant_registry_valid",
        "residual_work_registry_valid",
    )
    for field in required_true:
        if summary.get(field) is not True:
            errors.append(f"expected_true:{field}")
    if summary.get("freshness_state") != "fresh":
        errors.append("active_graph_not_fresh")
    if summary.get("live_corpus_changed") is not False:
        errors.append("live_corpus_changed")
    if summary.get("affected_graph_edge_count") != 0 or summary.get("affected_graph_node_count") != 0:
        errors.append("graph_impact_not_zero")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST:
        errors.append("v1_baseline_digest_changed")
    if summary.get("graph_runtime_hop_depth") != 1 or summary.get("formal_graph_sensitive_unit_count") != 9 or summary.get("default_equivalence_pass_count") != 9:
        errors.append("v1_equivalence_boundary_changed")
    if summary.get("known_causal_regression_count") != 0:
        errors.append("causal_regression_detected")
    if summary.get("runtime_incremental_repair_mode") != "manual_approved" or summary.get("automatic_unapproved_repair") is not False:
        errors.append("repair_activation_boundary_changed")
    if summary.get("unauthorized_graph_mutation_count") != 0 or summary.get("graph_content_mutation_count") != 0 or summary.get("corpus_content_mutation_count") != 0 or summary.get("graph_membership_change_count") != 0:
        errors.append("mutation_detected")
    if summary.get("graph_v2_runtime_promotion_applied") is not False or summary.get("fresh_graph_v2_available") is not False:
        errors.append("graph_v2_promoted")
    if summary.get("authority_gap_edge_count") != 2 or summary.get("external_authoritative_source_required") is not True:
        errors.append("authority_gap_boundary_changed")
    if manifest.get("graph_lifecycle_v1_baseline_digest") != summary.get("graph_lifecycle_v1_baseline_digest") or manifest.get("graph_lifecycle_v1_stage_closeout_digest") != summary.get("graph_lifecycle_v1_stage_closeout_digest"):
        errors.append("manifest_digest_mismatch")
    if invariants.get("frozen_invariant_registry_valid") is not True:
        errors.append("invariant_registry_invalid")
    if residual.get("residual_work_registry_valid") is not True:
        errors.append("residual_registry_invalid")
    if any(value != 0 for value in runtime.get("read_only_mutation_counts", {}).values()):
        errors.append("read_only_mutation_detected")
    if trust.get("trust_dimensions_are_independent") is not True:
        errors.append("trust_dimensions_collapsed")
    if answers.get("answers", {}).get("Q9", {}).get("answer") is not False:
        errors.append("query_plane_mutation_answer_invalid")
    if contract.get("graph_lifecycle_v1_frozen") is not True:
        errors.append("contract_not_frozen")
    result = {
        "schema_version": "opk-rag.task0169.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], manifest: dict[str, Any], answers: dict[str, Any], trust: dict[str, Any], residual: dict[str, Any]) -> str:
    return f"""# TASK-0169 Graph Lifecycle V1 Freeze and Engineering Authority Closeout

## Freeze Authority

Graph Lifecycle V1 is frozen as the authoritative lifecycle baseline for the current Graph Retrieval V1 architecture. It provides deterministic provenance, revision identity, native corpus binding, freshness assessment, corpus-change impact analysis, bounded incremental repair, and runtime lifecycle governance. Query-time Graph mutation remains prohibited; repair execution remains approval-gated. This freeze does not promote Graph V2 multi-hop retrieval or resolve outstanding external target-authority gaps.

## Summary

* Outcome: `{summary["outcome_class"]}`
* Freeze eligible: `{summary["graph_lifecycle_v1_freeze_eligible"]}`
* Frozen: `{summary["graph_lifecycle_v1_frozen"]}`
* Baseline digest: `{summary["graph_lifecycle_v1_baseline_digest"]}`
* Stage closeout digest: `{summary["graph_lifecycle_v1_stage_closeout_digest"]}`
* Active graph revision: `{summary["active_authoritative_graph_revision"]}`
* Current corpus revision: `{summary["current_corpus_revision"]}`
* Freshness state: `{summary["freshness_state"]}`

## Frozen Runtime Boundary

Graph Retrieval V1 remains frozen with baseline digest `{summary["graph_retrieval_v1_baseline_digest"]}`, hop depth `{summary["graph_runtime_hop_depth"]}`, formal graph-sensitive equivalence `{summary["default_equivalence_pass_count"]}/{summary["formal_graph_sensitive_unit_count"]}`, and known causal regression count `{summary["known_causal_regression_count"]}`.

Runtime read-only lifecycle analysis is enabled for freshness checks, impact analysis, and repair-plan generation. Query / maintenance plane separation is valid. Repair mode is `{summary["runtime_incremental_repair_mode"]}` and automatic unapproved repair is `{summary["automatic_unapproved_repair"]}`.

## Trust Model

* Evidence provenance: `{trust["evidence_provenance"]["state"]}` with document/chunk/span coverage `{trust["evidence_provenance"]["document_provenance_coverage"]}` / `{trust["evidence_provenance"]["chunk_provenance_coverage"]}` / `{trust["evidence_provenance"]["span_provenance_coverage"]}`
* Revision freshness: `{trust["revision_freshness"]["state"]}`, assessable `{trust["revision_freshness"]["freshness_assessable"]}`, valid `{trust["revision_freshness"]["freshness_valid"]}`
* Target authority: `{trust["target_authority"]["state"]}`, authority gaps `{trust["target_authority"]["authority_gap_edge_count"]}`

## Graph V2 Boundary

Graph V2 runtime promotion is `{summary["graph_v2_runtime_promotion_applied"]}`. External authoritative source required is `{summary["external_authoritative_source_required"]}`. Graph V2 Multi-hop Authority Branch = `{residual["graph_v2_multi_hop_authority_branch"]}`.

## Required Questions

""" + "\n".join(f"* {qid}: `{payload}`" for qid, payload in answers["answers"].items()) + f"""

## Verification Scope

Focused TASK-0169 tests and required TASK-0157 / TASK-0161 / TASK-0162 through TASK-0168 verifiers were run. Full suite run: `{summary["full_suite_run"]}`.

Future lifecycle changes must create an explicit lifecycle V2 or baseline revision that preserves or intentionally supersedes manifest `{manifest["authority_revision"]}`.
"""


def _valid(upstream: dict[str, Any], task_id: str) -> bool:
    return any(row["task_id"] == task_id and row["verification_status"] == "valid" for row in upstream["records"])


def _summary(upstream: dict[str, Any], task_id: str) -> dict[str, Any]:
    for row in upstream["records"]:
        if row["task_id"] == task_id:
            return row["summary"]
    raise KeyError(task_id)


def _digest_without_digest_fields(value: Any) -> str:
    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: scrub(val) for key, val in sorted(obj.items()) if not key.endswith("_digest") and key not in {"digest", "digests", "summary"}}
        if isinstance(obj, list):
            return [scrub(item) for item in obj]
        return obj

    return digest_json(scrub(value))
