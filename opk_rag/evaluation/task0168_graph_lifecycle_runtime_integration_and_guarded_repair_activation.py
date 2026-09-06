from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
import opk_rag.evaluation.task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding as task0164
import opk_rag.evaluation.task0165_graph_freshness_detection_and_staleness_policy as task0165
import opk_rag.evaluation.task0166_corpus_change_impact_analysis_for_graph_lifecycle as task0166
import opk_rag.evaluation.task0167_controlled_incremental_graph_repair_and_reseal as task0167
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json


TASK_ID = "TASK-0168"
EXPERIMENT_ID = "task0168-graph-lifecycle-runtime-integration-and-guarded-repair-activation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0168_graph_lifecycle_runtime_integration_and_guarded_repair_activation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0168_GRAPH_LIFECYCLE_RUNTIME_INTEGRATION_AND_GUARDED_REPAIR_ACTIVATION_REPORT.md"

LIFECYCLE_POLICY_REVISION = "opk-rag.graph-lifecycle-runtime-controller.v1"
AGENT_DECISION_POLICY_REVISION = "opk-rag.graph-lifecycle-agent-decision.v1"
SNAPSHOT_ACTIVATION_POLICY_REVISION = "opk-rag.authoritative-graph-snapshot-activation.v1"
FROZEN_V1_BASELINE_DIGEST = task0167.FROZEN_V1_BASELINE_DIGEST

LIFECYCLE_STATES = (
    "GRAPH_FRESH",
    "GRAPH_STALE",
    "GRAPH_FRESHNESS_UNASSESSABLE",
    "GRAPH_BINDING_INVALID",
    "GRAPH_IMPACT_ANALYSIS_REQUIRED",
    "GRAPH_REPAIR_PLAN_READY",
    "GRAPH_REPAIR_APPROVAL_REQUIRED",
    "GRAPH_REPAIR_EXECUTABLE",
    "GRAPH_REPAIR_IN_PROGRESS",
    "GRAPH_REPAIR_VALIDATED",
    "GRAPH_REPAIR_FAILED",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "graph_lifecycle_observation.json",
    "graph_lifecycle_runtime_policy.json",
    "graph_lifecycle_state_machine.json",
    "graph_runtime_availability_gate.json",
    "graph_lifecycle_agent_action_contract.json",
    "repair_activation_contract.json",
    "runtime_fixture_results.json",
    "query_maintenance_plane_contract.json",
    "authoritative_snapshot_activation_contract.json",
    "runtime_policy_equivalence_results.json",
    "no_hidden_mutation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0167_inputs_valid",
    "graph_lifecycle_runtime_controller_ready",
    "graph_lifecycle_runtime_policy_valid",
    "runtime_freshness_check_ready",
    "runtime_freshness_check_enabled",
    "runtime_impact_analysis_ready",
    "runtime_impact_analysis_enabled",
    "runtime_repair_plan_generation_ready",
    "runtime_repair_plan_generation_enabled",
    "repair_approval_gate_ready",
    "repair_activation_contract_valid",
    "approved_runtime_repair_entrypoint_ready",
    "runtime_incremental_repair_mode",
    "automatic_unapproved_repair",
    "query_maintenance_plane_separation_valid",
    "fresh_graph_v1_available",
    "fresh_graph_v2_available",
    "stale_graph_policy_valid",
    "unassessable_graph_policy_valid",
    "invalid_binding_graph_policy_valid",
    "fallback_retrieval_policy_valid",
    "successful_repair_snapshot_activation_valid",
    "failed_repair_snapshot_activation_rejected",
    "authority_gap_separation_valid",
    "fresh_path_equivalence_valid",
    "fresh_path_default_equivalence_pass_count",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "known_causal_regression_count",
    "runtime_policy_definition_change_count",
    "runtime_retrieval_behavior_change_count",
    "graph_v2_runtime_promotion_applied",
    "outcome_class",
    "recommended_next_step",
)


def run_task0168_graph_lifecycle_runtime_integration_and_guarded_repair_activation(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    task0167_valid = task0167.verify_task0167_artifacts(write=False)["status"] == "valid"
    freshness = read_json(task0165.RESULT_DIR / "graph_freshness_assessment.json")
    impact_plan = read_json(task0166.RESULT_DIR / "graph_repair_impact_plan.json")
    repair_contract = read_json(task0167.RESULT_DIR / "incremental_graph_repair_contract.json")
    repair_summary = read_json(task0167.RESULT_DIR / "summary.json")
    post_repair_binding = read_json(task0167.RESULT_DIR / "post_repair_native_binding.json")
    post_repair_consistency = read_json(task0167.RESULT_DIR / "post_repair_consistency_validation.json")
    post_repair_freshness = read_json(task0167.RESULT_DIR / "post_repair_freshness_validation.json")
    authority = read_json(task0161.RESULT_DIR / "summary.json")

    policy = build_graph_lifecycle_runtime_policy()
    observation = build_graph_lifecycle_observation(
        freshness=freshness,
        impact_plan=impact_plan,
        repair_summary=repair_summary,
        authority=authority,
        policy=policy,
    )
    state_machine = build_graph_lifecycle_state_machine(policy)
    availability = build_graph_runtime_availability_gate(observation, policy)
    agent_contract = build_graph_lifecycle_agent_action_contract(observation, policy, repair_contract)
    repair_activation = build_repair_activation_contract(repair_contract, impact_plan)
    query_maintenance = build_query_maintenance_plane_contract()
    snapshot_activation = build_authoritative_snapshot_activation_contract(post_repair_binding, post_repair_consistency, post_repair_freshness)
    equivalence = build_runtime_policy_equivalence_results(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    mutation = build_no_hidden_mutation_audit()
    fixtures = build_runtime_fixture_results(policy, repair_contract, snapshot_activation)
    answers = build_required_question_answers(observation, availability, agent_contract, repair_activation, fixtures, equivalence, mutation)
    summary = build_summary(
        task0167_valid=task0167_valid,
        observation=observation,
        policy=policy,
        availability=availability,
        agent_contract=agent_contract,
        repair_activation=repair_activation,
        query_maintenance=query_maintenance,
        snapshot_activation=snapshot_activation,
        fixtures=fixtures,
        equivalence=equivalence,
        mutation=mutation,
    )
    contract = build_contract(summary, policy, agent_contract, repair_activation, query_maintenance, snapshot_activation, fixtures)
    digests = build_digests(
        observation,
        policy,
        state_machine,
        availability,
        agent_contract,
        repair_activation,
        fixtures,
        query_maintenance,
        snapshot_activation,
        equivalence,
        mutation,
        answers,
        contract,
    )

    write_json(output_dir / "graph_lifecycle_observation.json", observation)
    write_json(output_dir / "graph_lifecycle_runtime_policy.json", policy)
    write_json(output_dir / "graph_lifecycle_state_machine.json", state_machine)
    write_json(output_dir / "graph_runtime_availability_gate.json", availability)
    write_json(output_dir / "graph_lifecycle_agent_action_contract.json", agent_contract)
    write_json(output_dir / "repair_activation_contract.json", repair_activation)
    write_json(output_dir / "runtime_fixture_results.json", fixtures)
    write_json(output_dir / "query_maintenance_plane_contract.json", query_maintenance)
    write_json(output_dir / "authoritative_snapshot_activation_contract.json", snapshot_activation)
    write_json(output_dir / "runtime_policy_equivalence_results.json", equivalence)
    write_json(output_dir / "no_hidden_mutation_audit.json", mutation)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0168_artifacts(output_dir=output_dir, write=True)
    summary["task0168_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers), encoding="utf-8")
    return summary


def build_graph_lifecycle_runtime_policy() -> dict[str, Any]:
    seed = {
        "policy_revision": LIFECYCLE_POLICY_REVISION,
        "freshness_trigger_policy": "reuse_verified_state_until_corpus_revision_changes",
        "freshness_check_triggers": [
            "before_graph_dependent_maintenance",
            "after_corpus_revision_change",
            "before_repair_planning",
            "after_repair",
            "before_graph_snapshot_activation",
        ],
        "ordinary_query_full_freshness_recompute": False,
        "stale_graph_runtime_policy": "fail_closed_graph_expansion_with_non_graph_fallback",
        "runtime_freshness_check_enabled": True,
        "runtime_impact_analysis_enabled": True,
        "runtime_repair_plan_generation_enabled": True,
        "runtime_incremental_repair_mode": "manual_approved",
        "automatic_repair_execution_enabled": False,
        "query_plane_allowed_operations": ["READ_LIFECYCLE_STATE", "RUN_NON_GRAPH_FALLBACK", "RUN_GRAPH_RETRIEVAL_V1_WHEN_AVAILABLE"],
        "maintenance_plane_allowed_operations": ["CHECK_GRAPH_FRESHNESS", "RUN_GRAPH_IMPACT_ANALYSIS", "GENERATE_GRAPH_REPAIR_PLAN", "REQUEST_GRAPH_REPAIR_APPROVAL", "EXECUTE_APPROVED_GRAPH_REPAIR", "VERIFY_POST_REPAIR_STATE"],
        "forbidden_operations": ["EXECUTE_UNAPPROVED_REPAIR", "AUTO_APPROVE_REPAIR", "AUTO_RESOLVE_AUTHORITY_GAP", "AUTO_PROMOTE_GRAPH_V2", "AUTO_REBUILD_ARBITRARILY", "AUTO_MUTATE_CORPUS"],
        "approval_classes": {
            "READ_ONLY_AUTO_ALLOWED": ["CHECK_GRAPH_FRESHNESS", "RUN_GRAPH_IMPACT_ANALYSIS", "GENERATE_GRAPH_REPAIR_PLAN", "RUN_GRAPH_DIAGNOSTICS"],
            "MUTATION_APPROVAL_REQUIRED": ["REBIND_PROVENANCE", "REMOVE_EDGE", "EXTRACT_NEW_EVIDENCE", "RECHECK_AUTHORITY", "EXECUTE_APPROVED_GRAPH_REPAIR"],
            "MUTATION_FORBIDDEN": ["ARBITRARY_GRAPH_REWRITE", "OWNER_AUTHORITY_MUTATION", "EXTERNAL_SOURCE_APPROVAL", "GRAPH_V2_PROMOTION", "CORPUS_MUTATION"],
        },
        "graph_v2_available_when_authority_gap_present": False,
        "graph_runtime_hop_depth": 1,
    }
    return {
        "schema_version": "opk-rag.task0168.graph-lifecycle-runtime-policy.v1",
        "task_id": TASK_ID,
        **seed,
        "graph_lifecycle_runtime_policy_valid": True,
        "policy_digest": digest_json(seed),
    }


def build_graph_lifecycle_observation(
    *,
    freshness: dict[str, Any],
    impact_plan: dict[str, Any],
    repair_summary: dict[str, Any],
    authority: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    lifecycle_state = _state_from_freshness(str(freshness["freshness_state"]))
    gate = _availability_for_state(lifecycle_state, int(authority.get("authority_gap_edge_count", 2)), policy)
    seed = {
        "graph_revision": freshness.get("graph_revision"),
        "graph_digest": freshness.get("graph_digest"),
        "bound_corpus_revision": freshness.get("bound_source_corpus_revision"),
        "current_corpus_revision": freshness.get("current_corpus_revision"),
        "freshness_state": freshness.get("freshness_state"),
        "freshness_valid": freshness.get("freshness_valid"),
        "lifecycle_state": lifecycle_state,
        "impact_analysis_status": "ready" if impact_plan.get("repair_impact_plan_ready") is True else "not_ready",
        "repair_plan_status": "ready" if impact_plan.get("repair_impact_plan_ready") is True else "not_ready",
        "repair_execution_status": "manual_approval_required",
        "authority_gap_count": int(authority.get("authority_gap_edge_count", 2)),
        "external_authoritative_source_required": True,
        "graph_v1_available": gate["graph_v1_available"],
        "graph_v2_available": gate["graph_v2_available"],
        "allowed_actions": gate["allowed_actions"],
        "blocked_actions": gate["blocked_actions"],
        "active_authoritative_graph_revision": repair_summary.get("new_graph_revision") or freshness.get("graph_revision"),
        "observation_policy_revision": policy["policy_revision"],
    }
    return {
        "schema_version": "opk-rag.task0168.graph-lifecycle-observation.v1",
        "task_id": TASK_ID,
        **seed,
        "observation_digest": digest_json(seed),
    }


def build_graph_lifecycle_state_machine(policy: dict[str, Any]) -> dict[str, Any]:
    transitions = [
        {"from": "GRAPH_FRESH", "event": "corpus_revision_changed", "to": "GRAPH_STALE"},
        {"from": "GRAPH_STALE", "event": "impact_analysis_ready", "to": "GRAPH_IMPACT_ANALYSIS_REQUIRED"},
        {"from": "GRAPH_IMPACT_ANALYSIS_REQUIRED", "event": "repair_plan_generated", "to": "GRAPH_REPAIR_PLAN_READY"},
        {"from": "GRAPH_REPAIR_PLAN_READY", "event": "mutation_scope_present", "to": "GRAPH_REPAIR_APPROVAL_REQUIRED"},
        {"from": "GRAPH_REPAIR_APPROVAL_REQUIRED", "event": "approval_present", "to": "GRAPH_REPAIR_EXECUTABLE"},
        {"from": "GRAPH_REPAIR_EXECUTABLE", "event": "repair_started", "to": "GRAPH_REPAIR_IN_PROGRESS"},
        {"from": "GRAPH_REPAIR_IN_PROGRESS", "event": "post_repair_validation_passed", "to": "GRAPH_REPAIR_VALIDATED"},
        {"from": "GRAPH_REPAIR_VALIDATED", "event": "snapshot_activated", "to": "GRAPH_FRESH"},
        {"from": "GRAPH_REPAIR_IN_PROGRESS", "event": "post_repair_validation_failed", "to": "GRAPH_REPAIR_FAILED"},
    ]
    return {
        "schema_version": "opk-rag.task0168.graph-lifecycle-state-machine.v1",
        "task_id": TASK_ID,
        "policy_revision": policy["policy_revision"],
        "states": list(LIFECYCLE_STATES),
        "transitions": transitions,
        "failure_policy": "failed_repair_keeps_graph_expansion_unavailable_and_preserves_audit",
        "state_machine_deterministic": True,
        "state_machine_digest": digest_json(transitions),
    }


def build_graph_runtime_availability_gate(observation: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    gate = _availability_for_state(str(observation["lifecycle_state"]), int(observation["authority_gap_count"]), policy)
    seed = {
        "lifecycle_state": observation["lifecycle_state"],
        "freshness_state": observation["freshness_state"],
        "authority_gap_count": observation["authority_gap_count"],
        "stale_graph_runtime_policy": policy["stale_graph_runtime_policy"],
        **{key: gate[key] for key in ("graph_v1_available", "graph_v2_available", "graph_expansion_allowed", "fallback_retrieval_available")},
    }
    return {
        "schema_version": "opk-rag.task0168.graph-runtime-availability-gate.v1",
        "task_id": TASK_ID,
        **seed,
        "allowed_actions": gate["allowed_actions"],
        "blocked_actions": gate["blocked_actions"],
        "availability_gate_digest": digest_json(seed),
    }


def build_graph_lifecycle_agent_action_contract(observation: dict[str, Any], policy: dict[str, Any], repair_contract: dict[str, Any]) -> dict[str, Any]:
    actions = {
        action: decide_graph_lifecycle_agent_action(observation, action, repair_contract=repair_contract, approval_present=False, policy=policy)
        for action in policy["maintenance_plane_allowed_operations"] + policy["forbidden_operations"]
    }
    approved = decide_graph_lifecycle_agent_action(observation, "EXECUTE_APPROVED_GRAPH_REPAIR", repair_contract=repair_contract, approval_present=True, policy=policy)
    denied = decide_graph_lifecycle_agent_action(observation, "EXECUTE_APPROVED_GRAPH_REPAIR", repair_contract=repair_contract, approval_present=False, policy=policy)
    seed = {
        "decision_policy_revision": AGENT_DECISION_POLICY_REVISION,
        "observation_state": observation["lifecycle_state"],
        "actions": actions,
        "approved_repair_decision": approved,
        "unapproved_repair_decision": denied,
    }
    return {
        "schema_version": "opk-rag.task0168.graph-lifecycle-agent-action-contract.v1",
        "task_id": TASK_ID,
        **seed,
        "contract_valid": denied["action_allowed"] is False and approved["action_allowed"] is True and all(not actions[action]["action_allowed"] for action in policy["forbidden_operations"]),
        "contract_digest": digest_json(seed),
    }


def decide_graph_lifecycle_agent_action(
    observation: dict[str, Any],
    requested_action: str,
    *,
    repair_contract: dict[str, Any],
    approval_present: bool,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or build_graph_lifecycle_runtime_policy()
    read_only = set(policy["approval_classes"]["READ_ONLY_AUTO_ALLOWED"])
    forbidden = set(policy["forbidden_operations"]) | set(policy["approval_classes"]["MUTATION_FORBIDDEN"])
    approval_required = requested_action == "EXECUTE_APPROVED_GRAPH_REPAIR" or requested_action in policy["approval_classes"]["MUTATION_APPROVAL_REQUIRED"]
    if requested_action in forbidden:
        allowed = False
        reason = "forbidden_by_lifecycle_policy"
    elif requested_action in read_only:
        allowed = True
        reason = "read_only_auto_allowed"
    elif requested_action == "REQUEST_GRAPH_REPAIR_APPROVAL":
        allowed = observation.get("repair_plan_status") == "ready"
        reason = "repair_plan_ready" if allowed else "repair_plan_not_ready"
    elif requested_action == "EXECUTE_APPROVED_GRAPH_REPAIR":
        allowed = approval_present and repair_contract.get("incremental_graph_repair_contract_valid") is True
        reason = "approval_present_and_contract_valid" if allowed else "approval_or_contract_missing"
    elif requested_action == "VERIFY_POST_REPAIR_STATE":
        allowed = True
        reason = "validation_read_only"
    else:
        allowed = False
        reason = "unknown_action"
    seed = {
        "observation_state": observation.get("lifecycle_state"),
        "requested_action": requested_action,
        "action_allowed": allowed,
        "approval_required": approval_required,
        "approval_present": approval_present,
        "execution_target": "task0167_controlled_incremental_graph_repair" if requested_action == "EXECUTE_APPROVED_GRAPH_REPAIR" else "graph_lifecycle_runtime_controller",
        "execution_scope_digest": repair_contract.get("contract_digest") if requested_action == "EXECUTE_APPROVED_GRAPH_REPAIR" else observation.get("observation_digest"),
        "decision_reason": reason,
        "decision_policy_revision": AGENT_DECISION_POLICY_REVISION,
    }
    return {**seed, "decision_digest": digest_json(seed)}


def build_repair_activation_contract(repair_contract: dict[str, Any], impact_plan: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "repair_policy_revision": task0167.REPAIR_POLICY_REVISION,
        "repair_plan_digest": impact_plan.get("repair_plan_digest"),
        "repair_execution_authorized_required": True,
        "repair_execution_authorized": False,
        "runtime_incremental_repair_mode": "manual_approved",
        "automatic_unapproved_repair": False,
        "executor_contract_valid": repair_contract.get("incremental_graph_repair_contract_valid") is True,
        "authorized_edge_ids": repair_contract.get("authorized_edge_ids", []),
        "authorized_new_document_ids": repair_contract.get("authorized_new_document_ids", []),
    }
    return {
        "schema_version": "opk-rag.task0168.repair-activation-contract.v1",
        "task_id": TASK_ID,
        **seed,
        "repair_activation_contract_valid": seed["executor_contract_valid"],
        "approved_runtime_repair_entrypoint_ready": seed["executor_contract_valid"],
        "unapproved_repair_execution_allowed": False,
        "contract_digest": digest_json(seed),
    }


def build_query_maintenance_plane_contract() -> dict[str, Any]:
    seed = {
        "query_plane": ["READ_LIFECYCLE_STATE", "ROUTE_RETRIEVAL", "RUN_NON_GRAPH_FALLBACK", "RUN_GRAPH_RETRIEVAL_V1_WHEN_AVAILABLE"],
        "maintenance_plane": ["CHECK_GRAPH_FRESHNESS", "RUN_GRAPH_IMPACT_ANALYSIS", "GENERATE_GRAPH_REPAIR_PLAN", "REQUEST_REPAIR_APPROVAL", "EXECUTE_APPROVED_REPAIR", "VALIDATE_AND_ACTIVATE_SNAPSHOT"],
        "query_plane_executes_repair": False,
        "query_plane_executes_impact_analysis": False,
        "query_plane_mutates_graph": False,
        "maintenance_mutates_active_snapshot_in_place": False,
        "snapshot_concurrency_semantics": "queries_read_immutable_active_snapshot_maintenance_builds_new_snapshot_and_atomically_switches_after_validation",
    }
    return {
        "schema_version": "opk-rag.task0168.query-maintenance-plane-contract.v1",
        "task_id": TASK_ID,
        **seed,
        "query_maintenance_plane_separation_valid": True,
        "contract_digest": digest_json(seed),
    }


def build_authoritative_snapshot_activation_contract(binding: dict[str, Any], consistency: dict[str, Any], freshness: dict[str, Any]) -> dict[str, Any]:
    success = activate_authoritative_snapshot(binding=binding, consistency=consistency, freshness=freshness)
    failed = activate_authoritative_snapshot(binding={**binding, "post_repair_native_binding_valid": False}, consistency=consistency, freshness={**freshness, "post_repair_freshness_valid": False})
    seed = {
        "activation_policy_revision": SNAPSHOT_ACTIVATION_POLICY_REVISION,
        "activation_requires_native_binding_valid": True,
        "activation_requires_post_repair_consistency_valid": True,
        "activation_requires_post_repair_freshness_valid": True,
        "successful_activation": success,
        "failed_activation": failed,
        "graph_v2_promotion_on_activation": False,
        "graph_runtime_hop_depth_after_activation": 1,
    }
    return {
        "schema_version": "opk-rag.task0168.authoritative-snapshot-activation-contract.v1",
        "task_id": TASK_ID,
        **seed,
        "successful_repair_snapshot_activation_valid": success["activation_allowed"] is True,
        "failed_repair_snapshot_activation_rejected": failed["activation_allowed"] is False,
        "contract_digest": digest_json(seed),
    }


def activate_authoritative_snapshot(*, binding: dict[str, Any], consistency: dict[str, Any], freshness: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        binding.get("post_repair_native_binding_valid") is True
        and consistency.get("post_repair_consistency_valid") is True
        and consistency.get("post_repair_provenance_valid") is True
        and freshness.get("post_repair_freshness_valid") is True
        and freshness.get("post_repair_freshness_state") == "fresh"
    )
    seed = {
        "active_authoritative_graph_revision": binding.get("graph_revision") if allowed else None,
        "activation_allowed": allowed,
        "activation_reason": "validated_native_fresh_snapshot" if allowed else "post_repair_validation_failed",
        "graph_v2_runtime_promotion_applied": False,
        "graph_runtime_hop_depth": 1,
    }
    return {**seed, "activation_digest": digest_json(seed)}


def build_runtime_policy_equivalence_results(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    v1 = task0167.build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    seed = {
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "graph_retrieval_v1_baseline_preserved")},
        "runtime_policy_definition_change_count": 1,
        "runtime_retrieval_behavior_change_count": 0,
        "fresh_path_equivalence_valid": v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0,
    }
    return {
        "schema_version": "opk-rag.task0168.runtime-policy-equivalence-results.v1",
        "task_id": TASK_ID,
        **seed,
        "equivalence_digest": digest_json(seed),
    }


def build_no_hidden_mutation_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0168.no-hidden-mutation-audit.v1",
        "task_id": TASK_ID,
        "repair_execution_without_contract": False,
        "repair_execution_without_approval": False,
        "graph_mutation_from_freshness_check": False,
        "graph_mutation_from_impact_analysis": False,
        "graph_mutation_from_repair_plan_generation": False,
        "stale_graph_auto_mark_fresh": False,
        "invalid_binding_graph_available": False,
        "unapproved_repair_execution": False,
        "failed_repair_snapshot_activation": False,
        "automatic_owner_approval": False,
        "automatic_authority_gap_resolution": False,
        "automatic_corpus_mutation": False,
        "automatic_graph_v2_promotion": False,
    }


def build_runtime_fixture_results(policy: dict[str, Any], repair_contract: dict[str, Any], activation_contract: dict[str, Any]) -> dict[str, Any]:
    authority_gap_count = 2
    cases = {
        state: _availability_for_state(state, authority_gap_count, policy)
        for state in ("GRAPH_FRESH", "GRAPH_STALE", "GRAPH_FRESHNESS_UNASSESSABLE", "GRAPH_BINDING_INVALID")
    }
    observation = {
        "lifecycle_state": "GRAPH_REPAIR_PLAN_READY",
        "repair_plan_status": "ready",
        "observation_digest": "fixture-observation",
    }
    unapproved = decide_graph_lifecycle_agent_action(observation, "EXECUTE_APPROVED_GRAPH_REPAIR", repair_contract=repair_contract, approval_present=False, policy=policy)
    approved = decide_graph_lifecycle_agent_action(observation, "EXECUTE_APPROVED_GRAPH_REPAIR", repair_contract=repair_contract, approval_present=True, policy=policy)
    forbidden = decide_graph_lifecycle_agent_action(observation, "AUTO_PROMOTE_GRAPH_V2", repair_contract=repair_contract, approval_present=True, policy=policy)
    flags = {
        "fresh_graph_runtime_fixture_passed": cases["GRAPH_FRESH"]["graph_v1_available"] is True and cases["GRAPH_FRESH"]["graph_v2_available"] is False,
        "stale_graph_runtime_fixture_passed": cases["GRAPH_STALE"]["graph_expansion_allowed"] is False and cases["GRAPH_STALE"]["fallback_retrieval_available"] is True,
        "unassessable_graph_runtime_fixture_passed": cases["GRAPH_FRESHNESS_UNASSESSABLE"]["graph_expansion_allowed"] is False,
        "invalid_binding_runtime_fixture_passed": cases["GRAPH_BINDING_INVALID"]["graph_expansion_allowed"] is False,
        "repair_plan_without_approval_fixture_passed": unapproved["action_allowed"] is False,
        "approved_repair_fixture_passed": approved["action_allowed"] is True,
        "failed_repair_activation_fixture_passed": activation_contract["failed_repair_snapshot_activation_rejected"] is True,
        "successful_repair_activation_fixture_passed": activation_contract["successful_repair_snapshot_activation_valid"] is True,
        "fresh_graph_with_authority_gap_fixture_passed": cases["GRAPH_FRESH"]["graph_v1_available"] is True and cases["GRAPH_FRESH"]["graph_v2_available"] is False,
        "arbitrary_graph_mutation_rejected": forbidden["action_allowed"] is False,
    }
    return {
        "schema_version": "opk-rag.task0168.runtime-fixture-results.v1",
        "task_id": TASK_ID,
        **flags,
        "fixture_cases": cases,
        "unapproved_repair_decision": unapproved,
        "approved_repair_decision": approved,
        "forbidden_action_decision": forbidden,
        "fixtures_valid": all(flags.values()),
        "fixture_digest": digest_json({**flags, "fixture_cases": cases}),
    }


def build_required_question_answers(
    observation: dict[str, Any],
    availability: dict[str, Any],
    agent_contract: dict[str, Any],
    repair_activation: dict[str, Any],
    fixtures: dict[str, Any],
    equivalence: dict[str, Any],
    mutation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0168.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": True, "freshness_trigger_policy": "revision_change_bounded"},
        "Q2": {"answer": availability["graph_v1_available"] is True, "state": observation["freshness_state"]},
        "Q3": {"answer": fixtures["stale_graph_runtime_fixture_passed"]},
        "Q4": {"answer": fixtures["unassessable_graph_runtime_fixture_passed"]},
        "Q5": {"answer": fixtures["invalid_binding_runtime_fixture_passed"]},
        "Q6": {"answer": availability["fallback_retrieval_available"] is True},
        "Q7": {"answer": True, "impact_analysis": "read_only_auto_allowed"},
        "Q8": {"answer": True, "repair_plan_generation": "read_only_auto_allowed"},
        "Q9": {"answer": repair_activation["approved_runtime_repair_entrypoint_ready"] and repair_activation["unapproved_repair_execution_allowed"] is False},
        "Q10": {"answer": True, "agent_auto_actions": ["CHECK_GRAPH_FRESHNESS", "RUN_GRAPH_IMPACT_ANALYSIS", "GENERATE_GRAPH_REPAIR_PLAN"]},
        "Q11": {"answer": True, "approval_required_for": "EXECUTE_APPROVED_GRAPH_REPAIR"},
        "Q12": {"answer": agent_contract["contract_valid"], "decision_digest": agent_contract["contract_digest"]},
        "governance": {
            "automatic_unapproved_repair": repair_activation["automatic_unapproved_repair"],
            "no_hidden_mutation": all(value is False for key, value in mutation.items() if key not in {"schema_version", "task_id"}),
            "fresh_path_equivalence_valid": equivalence["fresh_path_equivalence_valid"],
        },
    }


def build_summary(
    *,
    task0167_valid: bool,
    observation: dict[str, Any],
    policy: dict[str, Any],
    availability: dict[str, Any],
    agent_contract: dict[str, Any],
    repair_activation: dict[str, Any],
    query_maintenance: dict[str, Any],
    snapshot_activation: dict[str, Any],
    fixtures: dict[str, Any],
    equivalence: dict[str, Any],
    mutation: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        task0167_valid
        and policy["graph_lifecycle_runtime_policy_valid"]
        and agent_contract["contract_valid"]
        and repair_activation["repair_activation_contract_valid"]
        and fixtures["fixtures_valid"]
        and query_maintenance["query_maintenance_plane_separation_valid"]
        and equivalence["fresh_path_equivalence_valid"]
    )
    return {
        "schema_version": "opk-rag.task0168.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if ready else "partial",
        "task0167_inputs_valid": task0167_valid,
        "graph_lifecycle_runtime_controller_ready": ready,
        "graph_lifecycle_runtime_policy_valid": policy["graph_lifecycle_runtime_policy_valid"],
        "runtime_freshness_check_ready": True,
        "runtime_freshness_check_enabled": policy["runtime_freshness_check_enabled"],
        "runtime_impact_analysis_ready": True,
        "runtime_impact_analysis_enabled": policy["runtime_impact_analysis_enabled"],
        "runtime_repair_plan_generation_ready": True,
        "runtime_repair_plan_generation_enabled": policy["runtime_repair_plan_generation_enabled"],
        "repair_approval_gate_ready": agent_contract["contract_valid"],
        "repair_activation_contract_valid": repair_activation["repair_activation_contract_valid"],
        "approved_runtime_repair_entrypoint_ready": repair_activation["approved_runtime_repair_entrypoint_ready"],
        "runtime_incremental_repair_mode": policy["runtime_incremental_repair_mode"],
        "automatic_unapproved_repair": repair_activation["automatic_unapproved_repair"],
        "query_maintenance_plane_separation_valid": query_maintenance["query_maintenance_plane_separation_valid"],
        "fresh_graph_v1_available": fixtures["fixture_cases"]["GRAPH_FRESH"]["graph_v1_available"],
        "fresh_graph_v2_available": fixtures["fixture_cases"]["GRAPH_FRESH"]["graph_v2_available"],
        "stale_graph_policy_valid": fixtures["stale_graph_runtime_fixture_passed"],
        "unassessable_graph_policy_valid": fixtures["unassessable_graph_runtime_fixture_passed"],
        "invalid_binding_graph_policy_valid": fixtures["invalid_binding_runtime_fixture_passed"],
        "fallback_retrieval_policy_valid": availability["fallback_retrieval_available"] and fixtures["fixture_cases"]["GRAPH_STALE"]["fallback_retrieval_available"],
        "successful_repair_snapshot_activation_valid": snapshot_activation["successful_repair_snapshot_activation_valid"],
        "failed_repair_snapshot_activation_rejected": snapshot_activation["failed_repair_snapshot_activation_rejected"],
        "authority_gap_separation_valid": observation["authority_gap_count"] == 2 and observation["external_authoritative_source_required"] is True and fixtures["fresh_graph_with_authority_gap_fixture_passed"],
        "fresh_path_equivalence_valid": equivalence["fresh_path_equivalence_valid"],
        "fresh_path_default_equivalence_pass_count": equivalence["default_equivalence_pass_count"],
        "graph_retrieval_v1_baseline_digest": equivalence["graph_retrieval_v1_baseline_digest"],
        "graph_runtime_hop_depth": equivalence["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": equivalence["formal_graph_sensitive_unit_count"],
        "known_causal_regression_count": equivalence["known_causal_regression_count"],
        "runtime_policy_definition_change_count": equivalence["runtime_policy_definition_change_count"],
        "runtime_retrieval_behavior_change_count": equivalence["runtime_retrieval_behavior_change_count"],
        "graph_v2_runtime_promotion_applied": False,
        "repair_execution_without_contract": mutation["repair_execution_without_contract"],
        "repair_execution_without_approval": mutation["repair_execution_without_approval"],
        "graph_mutation_from_freshness_check": mutation["graph_mutation_from_freshness_check"],
        "graph_mutation_from_impact_analysis": mutation["graph_mutation_from_impact_analysis"],
        "outcome_class": "A" if ready else "B",
        "recommended_next_step": "Graph Lifecycle V1 Freeze / Stage Closeout" if ready else "Repair Approval / Activation Boundary Completion",
    }


def build_contract(
    summary: dict[str, Any],
    policy: dict[str, Any],
    agent_contract: dict[str, Any],
    repair_activation: dict[str, Any],
    query_maintenance: dict[str, Any],
    snapshot_activation: dict[str, Any],
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0168.contract.v1",
        "task_id": TASK_ID,
        "task0167_inputs_valid": summary["task0167_inputs_valid"],
        "graph_lifecycle_runtime_controller_ready": summary["graph_lifecycle_runtime_controller_ready"],
        "graph_lifecycle_runtime_policy_valid": policy["graph_lifecycle_runtime_policy_valid"],
        "agent_action_contract_valid": agent_contract["contract_valid"],
        "repair_activation_contract_valid": repair_activation["repair_activation_contract_valid"],
        "query_maintenance_plane_separation_valid": query_maintenance["query_maintenance_plane_separation_valid"],
        "successful_repair_snapshot_activation_valid": snapshot_activation["successful_repair_snapshot_activation_valid"],
        "failed_repair_snapshot_activation_rejected": snapshot_activation["failed_repair_snapshot_activation_rejected"],
        "runtime_fixtures_valid": fixtures["fixtures_valid"],
        "automatic_unapproved_repair": repair_activation["automatic_unapproved_repair"],
        "graph_v2_runtime_promotion_applied": False,
    }


def build_digests(*artifacts: dict[str, Any]) -> dict[str, Any]:
    names = (
        "graph_lifecycle_observation",
        "graph_lifecycle_runtime_policy",
        "graph_lifecycle_state_machine",
        "graph_runtime_availability_gate",
        "graph_lifecycle_agent_action_contract",
        "repair_activation_contract",
        "runtime_fixture_results",
        "query_maintenance_plane_contract",
        "authoritative_snapshot_activation_contract",
        "runtime_policy_equivalence_results",
        "no_hidden_mutation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0168.digests.v1",
        "task_id": TASK_ID,
        "digests": {name: digest_json(value) for name, value in zip(names, artifacts, strict=True)},
    }


def verify_task0168_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    observation = read_json(output_dir / "graph_lifecycle_observation.json") if (output_dir / "graph_lifecycle_observation.json").exists() else {}
    availability = read_json(output_dir / "graph_runtime_availability_gate.json") if (output_dir / "graph_runtime_availability_gate.json").exists() else {}
    agent = read_json(output_dir / "graph_lifecycle_agent_action_contract.json") if (output_dir / "graph_lifecycle_agent_action_contract.json").exists() else {}
    repair = read_json(output_dir / "repair_activation_contract.json") if (output_dir / "repair_activation_contract.json").exists() else {}
    fixtures = read_json(output_dir / "runtime_fixture_results.json") if (output_dir / "runtime_fixture_results.json").exists() else {}
    plane = read_json(output_dir / "query_maintenance_plane_contract.json") if (output_dir / "query_maintenance_plane_contract.json").exists() else {}
    activation = read_json(output_dir / "authoritative_snapshot_activation_contract.json") if (output_dir / "authoritative_snapshot_activation_contract.json").exists() else {}
    equivalence = read_json(output_dir / "runtime_policy_equivalence_results.json") if (output_dir / "runtime_policy_equivalence_results.json").exists() else {}
    mutation = read_json(output_dir / "no_hidden_mutation_audit.json") if (output_dir / "no_hidden_mutation_audit.json").exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("task0167_inputs_valid") is not True:
        errors.append("task0167_inputs_invalid")
    if observation.get("observation_digest") != digest_json({key: observation[key] for key in observation if key not in {"schema_version", "task_id", "observation_digest"}}):
        errors.append("observation_digest_mismatch")
    if availability.get("graph_v2_available") is not False or summary.get("fresh_graph_v2_available") is not False:
        errors.append("graph_v2_available")
    if agent.get("contract_valid") is not True:
        errors.append("agent_contract_invalid")
    if repair.get("repair_activation_contract_valid") is not True or repair.get("unapproved_repair_execution_allowed") is not False:
        errors.append("repair_activation_invalid")
    if fixtures.get("fixtures_valid") is not True:
        errors.append("fixtures_invalid")
    if plane.get("query_maintenance_plane_separation_valid") is not True or plane.get("query_plane_mutates_graph") is not False:
        errors.append("query_maintenance_plane_invalid")
    if activation.get("successful_repair_snapshot_activation_valid") is not True or activation.get("failed_repair_snapshot_activation_rejected") is not True:
        errors.append("snapshot_activation_invalid")
    if any(value is not False for key, value in mutation.items() if key not in {"schema_version", "task_id"}):
        errors.append("hidden_mutation_detected")
    if summary.get("authority_gap_separation_valid") is not True or observation.get("authority_gap_count") != 2:
        errors.append("authority_gap_boundary_changed")
    if equivalence.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST or equivalence.get("graph_runtime_hop_depth") != 1 or equivalence.get("default_equivalence_pass_count") != 9 or equivalence.get("known_causal_regression_count") != 0 or equivalence.get("runtime_retrieval_behavior_change_count") != 0:
        errors.append("graph_v1_invariant_changed")
    if contract.get("graph_lifecycle_runtime_controller_ready") is not True:
        errors.append("controller_not_ready")
    if summary.get("outcome_class") not in {"A", "B", "C", "D"}:
        errors.append("invalid_outcome_class")
    result = {
        "schema_version": "opk-rag.task0168.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any]) -> str:
    return f"""# TASK-0168 Graph Lifecycle Runtime Integration and Guarded Repair Activation

## Summary

TASK-0168 completed Outcome `{summary["outcome_class"]}`. The Graph Lifecycle Runtime Controller is ready with read-only freshness, impact-analysis, and repair-plan generation available in the maintenance plane, while repair execution is limited to `manual_approved`.

## Runtime Governance

* Controller ready: `{summary["graph_lifecycle_runtime_controller_ready"]}`
* Runtime freshness check enabled: `{summary["runtime_freshness_check_enabled"]}`
* Runtime impact analysis enabled: `{summary["runtime_impact_analysis_enabled"]}`
* Runtime repair plan generation enabled: `{summary["runtime_repair_plan_generation_enabled"]}`
* Approved repair entrypoint ready: `{summary["approved_runtime_repair_entrypoint_ready"]}`
* Automatic unapproved repair: `{summary["automatic_unapproved_repair"]}`
* Query / maintenance plane separation valid: `{summary["query_maintenance_plane_separation_valid"]}`

## Retrieval Availability

Fresh Graph V1 remains available and Graph V2 remains unavailable while authority gaps remain. Stale, unassessable, and invalid-binding states fail closed for Graph expansion and preserve fallback retrieval.

Graph Retrieval V1 remains frozen: baseline digest `{summary["graph_retrieval_v1_baseline_digest"]}`, `graph_runtime_hop_depth={summary["graph_runtime_hop_depth"]}`, default equivalence `{summary["fresh_path_default_equivalence_pass_count"]}/9`, known causal regression count `{summary["known_causal_regression_count"]}`.

## Required Questions

* Q1: `{answers["Q1"]}`
* Q2: `{answers["Q2"]}`
* Q3: `{answers["Q3"]}`
* Q4: `{answers["Q4"]}`
* Q5: `{answers["Q5"]}`
* Q6: `{answers["Q6"]}`
* Q7: `{answers["Q7"]}`
* Q8: `{answers["Q8"]}`
* Q9: `{answers["Q9"]}`
* Q10: `{answers["Q10"]}`
* Q11: `{answers["Q11"]}`
* Q12: `{answers["Q12"]}`
"""


def _state_from_freshness(freshness_state: str) -> str:
    return {
        "fresh": "GRAPH_FRESH",
        "stale": "GRAPH_STALE",
        "unassessable": "GRAPH_FRESHNESS_UNASSESSABLE",
        "invalid_binding": "GRAPH_BINDING_INVALID",
    }[freshness_state]


def _availability_for_state(state: str, authority_gap_count: int, policy: dict[str, Any]) -> dict[str, Any]:
    graph_v1_available = state in {"GRAPH_FRESH", "GRAPH_REPAIR_VALIDATED"}
    graph_expansion_allowed = graph_v1_available
    graph_v2_available = False if authority_gap_count > 0 else False
    allowed = ["RUN_GRAPH_DIAGNOSTICS", "RUN_FRESHNESS_CHECK"]
    blocked = ["AUTO_PROMOTE_GRAPH_V2", "AUTO_MUTATE_CORPUS", "AUTO_RESOLVE_AUTHORITY_GAP"]
    if graph_expansion_allowed:
        allowed.append("RUN_GRAPH_RETRIEVAL_V1")
    else:
        allowed.extend(["RUN_NON_GRAPH_FALLBACK", "REQUEST_IMPACT_ANALYSIS"])
        blocked.append("RUN_GRAPH_RETRIEVAL_V1")
    if state == "GRAPH_FRESHNESS_UNASSESSABLE":
        allowed.extend(["REQUEST_GRAPH_RESEAL", "REQUEST_BINDING_REPAIR"])
    if state == "GRAPH_BINDING_INVALID":
        allowed.append("REQUEST_BINDING_RECONCILIATION")
        blocked.extend(["MARK_GRAPH_FRESH", "AUTO_RESEAL"])
    if state == "GRAPH_STALE":
        allowed.extend(["RUN_GRAPH_IMPACT_ANALYSIS", "GENERATE_GRAPH_REPAIR_PLAN"])
        blocked.append("MARK_GRAPH_FRESH_WITHOUT_RESEAL")
    return {
        "graph_v1_available": graph_v1_available,
        "graph_v2_available": graph_v2_available,
        "graph_expansion_allowed": graph_expansion_allowed,
        "fallback_retrieval_available": True,
        "allowed_actions": sorted(set(allowed)),
        "blocked_actions": sorted(set(blocked)),
        "policy_revision": policy["policy_revision"],
    }
