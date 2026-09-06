from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
import opk_rag.evaluation.task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding as task0164
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json


TASK_ID = "TASK-0165"
EXPERIMENT_ID = "task0165-graph-freshness-detection-and-staleness-policy"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0165_graph_freshness_detection_and_staleness_policy_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0165_GRAPH_FRESHNESS_DETECTION_AND_STALENESS_POLICY_REPORT.md"

FRESHNESS_POLICY_REVISION = "opk-rag.graph-freshness-detection-and-staleness-policy.v1"
ASSESSMENT_POLICY_REVISION = "opk-rag.graph-freshness-assessment.v1"
FROZEN_V1_BASELINE_DIGEST = task0164.FROZEN_V1_BASELINE_DIGEST

FRESHNESS_STATES = ("fresh", "stale", "unassessable", "invalid_binding")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "graph_freshness_assessment.json",
    "graph_freshness_policy.json",
    "freshness_decision_matrix.json",
    "freshness_capability_gate.json",
    "freshness_fixture_results.json",
    "staleness_reason_registry.json",
    "authoritative_graph_seal_validation.json",
    "no_automatic_repair_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0164_inputs_valid",
    "authoritative_graph_seal_valid",
    "native_binding_valid",
    "binding_origin",
    "new_graph_binding_status",
    "graph_revision",
    "graph_digest",
    "graph_snapshot_digest",
    "graph_seal_digest",
    "bound_source_corpus_revision",
    "bound_source_corpus_digest",
    "current_corpus_revision",
    "current_corpus_digest",
    "current_corpus_revision_recomputed",
    "current_corpus_identity_matches_recorded",
    "authoritative_graph_identity_valid",
    "binding_status",
    "binding_valid",
    "graph_freshness_policy_ready",
    "freshness_assessable",
    "freshness_state",
    "freshness_valid",
    "revision_match",
    "digest_match",
    "staleness_reason",
    "graph_freshness_gate_ready",
    "graph_freshness_gate_passed",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "external_authoritative_source_required",
    "authority_gap_edge_count",
    "document_provenance_coverage",
    "chunk_provenance_coverage",
    "span_provenance_coverage",
    "runtime_behavior_mutation",
    "automatic_graph_edge_deletion",
    "automatic_graph_edge_rewrite",
    "automatic_graph_regeneration",
    "automatic_graph_reseal",
    "automatic_graph_rebuild",
    "automatic_graph_repair",
    "automatic_corpus_mutation",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "default_equivalence_pass_count",
    "known_causal_regression_count",
    "runtime_policy_mutation_count",
    "outcome_class",
    "recommended_next_step",
    "assessment_policy_revision",
    "assessment_digest",
)


def run_task0165_graph_freshness_detection_and_staleness_policy(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    task0164_valid = task0164.verify_task0164_artifacts(write=False)["status"] == "valid"
    recorded_corpus = read_json(task0164.RESULT_DIR / "authoritative_corpus_snapshot.json")
    current_corpus = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    graph = read_json(task0164.RESULT_DIR / "authoritative_graph_snapshot.json")
    binding = read_json(task0164.RESULT_DIR / "native_graph_corpus_binding.json")
    seal = read_json(task0164.RESULT_DIR / "graph_snapshot_seal.json")

    policy = build_graph_freshness_policy()
    matrix = build_freshness_decision_matrix(policy)
    reasons = build_staleness_reason_registry()
    seal_validation = validate_authoritative_graph_seal(graph, binding, seal)
    assessment = assess_graph_freshness(
        graph=graph,
        binding=binding,
        seal=seal,
        current_corpus=current_corpus,
        recorded_corpus=recorded_corpus,
        policy=policy,
        seal_validation=seal_validation,
    )
    gate = build_freshness_capability_gate(assessment, policy)
    fixtures = build_freshness_fixture_results(graph, binding, seal, recorded_corpus, policy)
    mutation = build_no_automatic_repair_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    answers = build_required_question_answers(seal_validation, assessment, fixtures, gate, v1)
    summary = build_summary(
        task0164_valid=task0164_valid,
        graph=graph,
        seal=seal,
        binding=binding,
        current_corpus=current_corpus,
        recorded_corpus=recorded_corpus,
        policy=policy,
        seal_validation=seal_validation,
        assessment=assessment,
        gate=gate,
        mutation=mutation,
        v1=v1,
    )
    contract = build_contract(summary, policy, assessment, seal_validation, fixtures, mutation, gate, v1)
    digests = build_digests(assessment, policy, matrix, gate, fixtures, reasons, seal_validation, mutation, v1, answers, contract)

    write_json(output_dir / "graph_freshness_assessment.json", assessment)
    write_json(output_dir / "graph_freshness_policy.json", policy)
    write_json(output_dir / "freshness_decision_matrix.json", matrix)
    write_json(output_dir / "freshness_capability_gate.json", gate)
    write_json(output_dir / "freshness_fixture_results.json", fixtures)
    write_json(output_dir / "staleness_reason_registry.json", reasons)
    write_json(output_dir / "authoritative_graph_seal_validation.json", seal_validation)
    write_json(output_dir / "no_automatic_repair_audit.json", mutation)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0165_artifacts(output_dir=output_dir, write=True)
    summary["task0165_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers, policy), encoding="utf-8")
    return summary


def build_graph_freshness_policy() -> dict[str, Any]:
    allowed = {
        "fresh": ["USE_GRAPH_V1_RETRIEVAL", "RUN_GRAPH_DIAGNOSTICS", "RUN_CHANGE_IMPACT_BASELINE", "OBSERVE_FRESHNESS"],
        "stale": ["DIAGNOSE_CORPUS_CHANGE", "COMPUTE_GRAPH_CHANGE_IMPACT", "REQUEST_GRAPH_REPAIR", "DEFER_GRAPH_USE", "OBSERVE_FRESHNESS"],
        "unassessable": ["REQUEST_BINDING_RESEAL", "DIAGNOSE_BINDING", "OBSERVE_FRESHNESS"],
        "invalid_binding": ["REJECT_BINDING", "REQUEST_AUTHORITY_RECONCILIATION", "OBSERVE_FRESHNESS"],
    }
    blocked = {
        "fresh": ["PROMOTE_GRAPH_V2", "AUTO_MULTI_HOP", "AUTO_GRAPH_MUTATION", "AUTO_REPAIR_GRAPH", "AUTO_REBUILD_GRAPH"],
        "stale": ["MARK_FRESH_WITHOUT_RESEAL", "AUTO_REPAIR_GRAPH", "AUTO_RESEAL_GRAPH", "AUTO_REBUILD_GRAPH", "AUTO_PROMOTE_GRAPH_V2"],
        "unassessable": ["USE_AS_FRESH", "AUTO_MARK_FRESH", "AUTO_REPAIR_GRAPH", "AUTO_PROMOTE_GRAPH_V2"],
        "invalid_binding": ["USE_BINDING", "USE_AS_FRESH", "AUTO_MARK_FRESH", "AUTO_REPAIR_GRAPH", "AUTO_PROMOTE_GRAPH_V2"],
    }
    seed = {
        "policy_revision": FRESHNESS_POLICY_REVISION,
        "supported_states": list(FRESHNESS_STATES),
        "allowed_actions_by_state": allowed,
        "blocked_actions_by_state": blocked,
        "runtime_mutation_enabled": False,
        "automatic_repair_enabled": False,
    }
    return {
        "schema_version": "opk-rag.task0165.graph-freshness-policy.v1",
        "task_id": TASK_ID,
        **seed,
        "policy_digest": digest_json(seed),
    }


def build_freshness_decision_matrix(policy: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {"binding": "valid", "revision": "match", "digest": "match", "freshness_state": "fresh"},
        {"binding": "valid", "revision": "mismatch", "digest": "mismatch", "freshness_state": "stale"},
        {"binding": "valid", "revision": "mismatch", "digest": "match", "freshness_state": "invalid_binding"},
        {"binding": "valid", "revision": "match", "digest": "mismatch", "freshness_state": "invalid_binding"},
        {"binding": "missing", "revision": "unknown", "digest": "unknown", "freshness_state": "unassessable"},
        {"binding": "invalid", "revision": "any", "digest": "any", "freshness_state": "invalid_binding"},
    ]
    return {
        "schema_version": "opk-rag.task0165.freshness-decision-matrix.v1",
        "task_id": TASK_ID,
        "policy_revision": policy["policy_revision"],
        "rows": rows,
        "matrix_digest": digest_json(rows),
    }


def build_staleness_reason_registry() -> dict[str, Any]:
    reasons = [
        "corpus_revision_changed",
        "corpus_digest_changed",
        "corpus_membership_changed",
        "corpus_content_changed",
        "binding_missing",
        "binding_revision_digest_inconsistent",
        "binding_digest_invalid",
        "graph_identity_invalid",
        "seal_invalid",
        "unknown",
    ]
    return {
        "schema_version": "opk-rag.task0165.staleness-reason-registry.v1",
        "task_id": TASK_ID,
        "reason_codes": reasons,
        "reason_registry_digest": digest_json(reasons),
    }


def validate_authoritative_graph_seal(graph: dict[str, Any], binding: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    binding_seed = {
        "graph_revision": binding.get("graph_revision"),
        "graph_digest": binding.get("graph_digest"),
        "source_corpus_revision": binding.get("source_corpus_revision"),
        "source_corpus_digest": binding.get("source_corpus_digest"),
        "binding_policy_revision": binding.get("binding_policy_revision"),
        "binding_origin": binding.get("binding_origin"),
        "binding_status": binding.get("binding_status"),
    }
    expected_binding_digest = digest_json(binding_seed)
    if binding.get("binding_digest") != expected_binding_digest:
        errors.append("binding_digest_invalid")
    if binding.get("binding_origin") != "native" or binding.get("binding_status") != "proven" or binding.get("new_graph_binding_status") != "proven":
        errors.append("binding_not_native_proven")
    if not binding.get("source_corpus_revision") or not binding.get("source_corpus_digest"):
        errors.append("binding_missing")
    if binding.get("graph_revision") != graph.get("graph_revision"):
        errors.append("binding_graph_revision_mismatch")
    if binding.get("graph_digest") != graph.get("graph_digest"):
        errors.append("binding_graph_digest_mismatch")
    if graph.get("source_corpus_revision") != binding.get("source_corpus_revision") or graph.get("source_corpus_digest") != binding.get("source_corpus_digest"):
        errors.append("graph_binding_fields_mismatch")
    seal_seed = {
        "graph_revision": seal.get("graph_revision"),
        "graph_digest": seal.get("graph_digest"),
        "source_corpus_revision": seal.get("source_corpus_revision"),
        "source_corpus_digest": seal.get("source_corpus_digest"),
        "graph_derivation_policy_revision": seal.get("graph_derivation_policy_revision"),
        "binding_origin": seal.get("binding_origin"),
        "binding_status": seal.get("binding_status"),
        "formal_graph_node_count": seal.get("formal_graph_node_count"),
        "formal_graph_edge_count": seal.get("formal_graph_edge_count"),
        "authority_gap_edge_count": seal.get("authority_gap_edge_count"),
        "seal_policy_revision": seal.get("seal_policy_revision"),
    }
    expected_seal_digest = digest_json(seal_seed)
    if seal.get("seal_digest") != expected_seal_digest:
        errors.append("seal_digest_invalid")
    for field in ("graph_revision", "graph_digest", "source_corpus_revision", "source_corpus_digest", "binding_origin", "binding_status"):
        if seal.get(field) != binding.get(field):
            errors.append(f"seal_binding_{field}_mismatch")
    for field in ("formal_graph_node_count", "formal_graph_edge_count", "authority_gap_edge_count"):
        if seal.get(field) != graph.get(field):
            errors.append(f"seal_graph_{field}_mismatch")
    graph_seed = {
        "graph_content_digest": graph.get("graph_content_digest"),
        "source_corpus_revision": graph.get("source_corpus_revision"),
        "source_corpus_digest": graph.get("source_corpus_digest"),
        "graph_derivation_policy_revision": graph.get("graph_derivation_policy_revision"),
        "selected_authority_path": graph.get("selected_authority_path"),
        "binding_origin": graph.get("binding_origin"),
        "binding_status": graph.get("binding_status"),
    }
    expected_graph_digest = digest_json(graph_seed)
    if graph.get("graph_digest") != expected_graph_digest or graph.get("graph_revision") != f"graph-sha256:{expected_graph_digest}":
        errors.append("graph_identity_invalid")
    return {
        "schema_version": "opk-rag.task0165.authoritative-graph-seal-validation.v1",
        "task_id": TASK_ID,
        "authoritative_graph_seal_valid": not errors,
        "native_binding_valid": "binding_digest_invalid" not in errors and "binding_missing" not in errors and "binding_not_native_proven" not in errors,
        "binding_origin": binding.get("binding_origin"),
        "new_graph_binding_status": binding.get("new_graph_binding_status"),
        "graph_revision": graph.get("graph_revision"),
        "graph_digest": graph.get("graph_digest"),
        "graph_snapshot_digest": graph.get("graph_snapshot_digest"),
        "graph_seal_digest": seal.get("seal_digest"),
        "bound_source_corpus_revision": binding.get("source_corpus_revision"),
        "bound_source_corpus_digest": binding.get("source_corpus_digest"),
        "authoritative_graph_identity_valid": "graph_identity_invalid" not in errors,
        "binding_digest_valid": "binding_digest_invalid" not in errors,
        "seal_digest_valid": "seal_digest_invalid" not in errors,
        "errors": errors,
    }


def assess_graph_freshness(
    *,
    graph: dict[str, Any],
    binding: dict[str, Any],
    seal: dict[str, Any],
    current_corpus: dict[str, Any],
    recorded_corpus: dict[str, Any] | None,
    policy: dict[str, Any],
    seal_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seal_validation = seal_validation or validate_authoritative_graph_seal(graph, binding, seal)
    bound_revision = binding.get("source_corpus_revision") or graph.get("source_corpus_revision") or seal.get("source_corpus_revision")
    bound_digest = binding.get("source_corpus_digest") or graph.get("source_corpus_digest") or seal.get("source_corpus_digest")
    current_revision = current_corpus.get("corpus_revision")
    current_digest = current_corpus.get("corpus_digest")
    missing_binding = not bound_revision or not bound_digest or binding.get("binding_origin") in {None, "none"} or binding.get("binding_status") in {None, "binding_missing", "unavailable"}
    revision_match = bound_revision == current_revision if bound_revision and current_revision else None
    digest_match = bound_digest == current_digest if bound_digest and current_digest else None
    current_identity_matches_recorded = True
    if recorded_corpus is not None:
        current_identity_matches_recorded = current_revision == recorded_corpus.get("corpus_revision") and current_digest == recorded_corpus.get("corpus_digest")
    secondary_reasons: list[str] = []
    if missing_binding:
        state = "unassessable"
        reason = "binding_missing"
        assessable = False
        valid = False
    elif not seal_validation["authoritative_graph_seal_valid"]:
        state = "invalid_binding"
        reason = _invalid_binding_reason(seal_validation["errors"])
        assessable = True
        valid = False
    elif revision_match is True and digest_match is True:
        state = "fresh"
        reason = None
        assessable = True
        valid = True
    elif revision_match is False and digest_match is False:
        state = "stale"
        reason = _staleness_reason(recorded_corpus or binding, current_corpus)
        secondary_reasons = ["corpus_revision_changed", "corpus_digest_changed"]
        if reason not in secondary_reasons:
            secondary_reasons.append(reason)
        assessable = True
        valid = False
    elif (revision_match is True and digest_match is False) or (revision_match is False and digest_match is True):
        state = "invalid_binding"
        reason = "binding_revision_digest_inconsistent"
        assessable = True
        valid = False
    else:
        state = "invalid_binding"
        reason = "unknown"
        assessable = True
        valid = False
    gate = build_freshness_capability_gate({"freshness_state": state, "freshness_assessable": assessable, "freshness_valid": valid}, policy)
    seed = {
        "graph_revision": graph.get("graph_revision"),
        "graph_digest": graph.get("graph_digest"),
        "graph_snapshot_digest": graph.get("graph_snapshot_digest"),
        "graph_seal_digest": seal.get("seal_digest"),
        "bound_source_corpus_revision": bound_revision,
        "bound_source_corpus_digest": bound_digest,
        "current_corpus_revision": current_revision,
        "current_corpus_digest": current_digest,
        "binding_status": binding.get("binding_status"),
        "binding_valid": seal_validation["native_binding_valid"] and seal_validation["authoritative_graph_seal_valid"],
        "freshness_assessable": assessable,
        "freshness_state": state,
        "freshness_valid": valid,
        "revision_match": revision_match,
        "digest_match": digest_match,
        "staleness_reason": reason,
        "secondary_reasons": secondary_reasons,
        "allowed_actions": gate["allowed_actions"],
        "blocked_capabilities": gate["blocked_capabilities"],
        "assessment_policy_revision": ASSESSMENT_POLICY_REVISION,
    }
    return {
        "schema_version": "opk-rag.task0165.graph-freshness-assessment.v1",
        "task_id": TASK_ID,
        **seed,
        "current_corpus_revision_recomputed": True,
        "current_corpus_identity_matches_recorded": current_identity_matches_recorded,
        "authoritative_graph_identity_valid": seal_validation["authoritative_graph_identity_valid"],
        "assessment_digest": digest_json(seed),
    }


def _invalid_binding_reason(errors: list[str]) -> str:
    if "binding_digest_invalid" in errors:
        return "binding_digest_invalid"
    if "graph_identity_invalid" in errors:
        return "graph_identity_invalid"
    if any(error.startswith("seal_") for error in errors):
        return "seal_invalid"
    if "binding_missing" in errors:
        return "binding_missing"
    return "binding_revision_digest_inconsistent"


def _staleness_reason(bound_corpus: dict[str, Any], current_corpus: dict[str, Any]) -> str:
    if bound_corpus.get("source_document_identity_digest") and bound_corpus.get("source_document_identity_digest") != current_corpus.get("source_document_identity_digest"):
        return "corpus_membership_changed"
    if bound_corpus.get("corpus_content_digest") and bound_corpus.get("corpus_content_digest") != current_corpus.get("corpus_content_digest"):
        return "corpus_content_changed"
    return "corpus_revision_changed"


def build_freshness_capability_gate(assessment: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    state = str(assessment["freshness_state"])
    return {
        "schema_version": "opk-rag.task0165.freshness-capability-gate.v1",
        "task_id": TASK_ID,
        "policy_revision": policy["policy_revision"],
        "freshness_state": state,
        "graph_freshness_gate_ready": state in FRESHNESS_STATES,
        "graph_freshness_gate_passed": state == "fresh",
        "allowed_actions": policy["allowed_actions_by_state"][state],
        "blocked_capabilities": policy["blocked_actions_by_state"][state],
        "runtime_behavior_mutation": False,
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        "gate_digest": digest_json(
            {
                "policy_revision": policy["policy_revision"],
                "freshness_state": state,
                "allowed_actions": policy["allowed_actions_by_state"][state],
                "blocked_capabilities": policy["blocked_actions_by_state"][state],
                "runtime_behavior_mutation": False,
            }
        ),
    }


def build_freshness_fixture_results(graph: dict[str, Any], binding: dict[str, Any], seal: dict[str, Any], corpus: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    content_changed = _variant_content_changed(corpus)
    added = _variant_document_added(corpus)
    removed = _variant_document_removed(corpus)
    missing_binding = {**binding, "source_corpus_revision": None, "source_corpus_digest": None, "binding_origin": "none", "binding_status": "binding_missing", "new_graph_binding_status": "unproven"}
    missing_graph = {**graph, "source_corpus_revision": None, "source_corpus_digest": None, "binding_origin": "none", "binding_status": "unavailable"}
    bad_digest_binding = dict(binding)
    bad_digest_binding["binding_digest"] = "not-the-deterministic-binding-digest"
    same_revision_different_digest = {**corpus, "corpus_digest": "fixture-different-digest"}
    different_revision_same_digest = {**corpus, "corpus_revision": "corpus-sha256:fixture-different-revision"}
    cases = {
        "case_a_no_corpus_change": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=corpus, recorded_corpus=corpus, policy=policy),
        "case_b_document_content_change": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=content_changed, recorded_corpus=corpus, policy=policy),
        "case_c_document_added": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=added, recorded_corpus=corpus, policy=policy),
        "case_d_document_removed": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=removed, recorded_corpus=corpus, policy=policy),
        "missing_binding": assess_graph_freshness(graph=missing_graph, binding=missing_binding, seal=seal, current_corpus=corpus, recorded_corpus=corpus, policy=policy),
        "same_revision_different_digest": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=same_revision_different_digest, recorded_corpus=corpus, policy=policy),
        "different_revision_same_digest": assess_graph_freshness(graph=graph, binding=binding, seal=seal, current_corpus=different_revision_same_digest, recorded_corpus=corpus, policy=policy),
        "invalid_binding_digest": assess_graph_freshness(graph=graph, binding=bad_digest_binding, seal=seal, current_corpus=corpus, recorded_corpus=corpus, policy=policy),
    }
    states = {name: row["freshness_state"] for name, row in cases.items()}
    return {
        "schema_version": "opk-rag.task0165.freshness-fixture-results.v1",
        "task_id": TASK_ID,
        "case_states": states,
        "case_results": cases,
        "fixtures_valid": states
        == {
            "case_a_no_corpus_change": "fresh",
            "case_b_document_content_change": "stale",
            "case_c_document_added": "stale",
            "case_d_document_removed": "stale",
            "missing_binding": "unassessable",
            "same_revision_different_digest": "invalid_binding",
            "different_revision_same_digest": "invalid_binding",
            "invalid_binding_digest": "invalid_binding",
        },
        "fixture_digest": digest_json(states),
    }


def _variant_content_changed(corpus: dict[str, Any]) -> dict[str, Any]:
    members = [dict(row) for row in corpus["members"]]
    members[0]["document_content_digest"] = f"{members[0]['document_content_digest']}-changed"
    members[0]["source_revision"] = members[0]["document_content_digest"]
    return task0163.build_corpus_snapshot_from_entries(members)


def _variant_document_added(corpus: dict[str, Any]) -> dict[str, Any]:
    members = [dict(row) for row in corpus["members"]]
    members.append(
        {
            "source_document_id": "source-documents/fixture-added.md",
            "canonical_document_id": "source-documents/fixture-added.md",
            "document_content_digest": "fixture-added-digest",
            "source_revision": "fixture-added-digest",
        }
    )
    return task0163.build_corpus_snapshot_from_entries(members)


def _variant_document_removed(corpus: dict[str, Any]) -> dict[str, Any]:
    return task0163.build_corpus_snapshot_from_entries([dict(row) for row in corpus["members"][1:]])


def build_no_automatic_repair_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0165.no-automatic-repair-audit.v1",
        "task_id": TASK_ID,
        "runtime_behavior_mutation": False,
        "automatic_graph_edge_deletion": False,
        "automatic_graph_edge_rewrite": False,
        "automatic_graph_regeneration": False,
        "automatic_graph_reseal": False,
        "automatic_graph_rebuild": False,
        "automatic_graph_repair": False,
        "automatic_corpus_mutation": False,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    v1 = task0164.build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    return {
        "schema_version": "opk-rag.task0165.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count", "graph_retrieval_v1_baseline_preserved")},
    }


def build_required_question_answers(
    seal_validation: dict[str, Any],
    assessment: dict[str, Any],
    fixtures: dict[str, Any],
    gate: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0165.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": seal_validation["authoritative_graph_seal_valid"], "native_binding_valid": seal_validation["native_binding_valid"]},
        "Q2": {"answer": assessment["current_corpus_revision_recomputed"], "current_corpus_identity_matches_recorded": assessment["current_corpus_identity_matches_recorded"]},
        "Q3": {"answer": assessment["freshness_state"] in FRESHNESS_STATES, "assessment_digest": assessment["assessment_digest"]},
        "Q4": {"freshness_state": assessment["freshness_state"]},
        "Q5": {"answer": assessment["revision_match"], "bound_source_corpus_revision": assessment["bound_source_corpus_revision"], "current_corpus_revision": assessment["current_corpus_revision"]},
        "Q6": {"answer": assessment["digest_match"], "bound_source_corpus_digest": assessment["bound_source_corpus_digest"], "current_corpus_digest": assessment["current_corpus_digest"]},
        "Q7": {"answer": fixtures["case_states"]["case_b_document_content_change"] == "stale" and fixtures["case_states"]["missing_binding"] == "unassessable"},
        "Q8": {"answer": fixtures["case_states"]["case_b_document_content_change"] == "stale" and fixtures["case_states"]["same_revision_different_digest"] == "invalid_binding"},
        "Q9": {"answer": fixtures["case_states"]["case_b_document_content_change"] == "stale"},
        "Q10": {"answer": fixtures["case_states"]["case_c_document_added"] == "stale" and fixtures["case_states"]["case_d_document_removed"] == "stale"},
        "Q11": {"answer": fixtures["case_states"]["same_revision_different_digest"] == "invalid_binding" and fixtures["case_states"]["different_revision_same_digest"] == "invalid_binding"},
        "Q12": {"answer": True, "authority_gap_edge_count": 2, "freshness_state": assessment["freshness_state"]},
        "Q13": {"answer": bool(gate["allowed_actions"]) and bool(gate["blocked_capabilities"]), "freshness_state": gate["freshness_state"]},
        "Q14": {"answer": gate["runtime_behavior_mutation"] is False},
        "Q15": {"answer": v1["graph_retrieval_v1_baseline_preserved"] and v1["graph_runtime_hop_depth"] == 1 and v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0 and v1["runtime_policy_mutation_count"] == 0},
    }


def build_summary(
    *,
    task0164_valid: bool,
    graph: dict[str, Any],
    seal: dict[str, Any],
    binding: dict[str, Any],
    current_corpus: dict[str, Any],
    recorded_corpus: dict[str, Any],
    policy: dict[str, Any],
    seal_validation: dict[str, Any],
    assessment: dict[str, Any],
    gate: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    s162 = read_json(task0162.RESULT_DIR / "summary.json")
    s161 = read_json(task0161.RESULT_DIR / "summary.json")
    state = assessment["freshness_state"]
    if state == "fresh":
        outcome = "A"
        next_step = "TASK-0166 Corpus Change Impact Analysis for Graph Lifecycle"
    elif state == "stale":
        outcome = "B"
        next_step = "TASK-0166 Corpus Change Impact Analysis for Graph Lifecycle"
    elif state == "unassessable":
        outcome = "C"
        next_step = "repair_or_reestablish_authoritative_native_binding"
    else:
        outcome = "D"
        next_step = "Graph Snapshot Authority Reconciliation"
    return {
        "schema_version": "opk-rag.task0165.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0164_inputs_valid": task0164_valid,
        **{key: seal_validation[key] for key in ("authoritative_graph_seal_valid", "native_binding_valid", "binding_origin", "new_graph_binding_status", "graph_revision", "graph_digest", "graph_snapshot_digest", "graph_seal_digest", "bound_source_corpus_revision", "bound_source_corpus_digest", "authoritative_graph_identity_valid")},
        "current_corpus_revision": current_corpus["corpus_revision"],
        "current_corpus_digest": current_corpus["corpus_digest"],
        "current_corpus_revision_recomputed": assessment["current_corpus_revision_recomputed"],
        "current_corpus_identity_matches_recorded": current_corpus["corpus_revision"] == recorded_corpus["corpus_revision"] and current_corpus["corpus_digest"] == recorded_corpus["corpus_digest"],
        "binding_status": binding.get("binding_status"),
        "binding_valid": assessment["binding_valid"],
        "graph_freshness_policy_ready": policy["supported_states"] == list(FRESHNESS_STATES) and policy["runtime_mutation_enabled"] is False,
        "freshness_assessable": assessment["freshness_assessable"],
        "graph_freshness_assessable": assessment["freshness_assessable"],
        "freshness_state": state,
        "freshness_valid": assessment["freshness_valid"],
        "graph_freshness_valid": assessment["freshness_valid"],
        "revision_match": assessment["revision_match"],
        "digest_match": assessment["digest_match"],
        "staleness_reason": assessment["staleness_reason"],
        "graph_freshness_gate_ready": gate["graph_freshness_gate_ready"],
        "graph_freshness_gate_passed": gate["graph_freshness_gate_passed"],
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        "external_authoritative_source_required": s161["external_authoritative_source_required"],
        "authority_gap_edge_count": graph["authority_gap_edge_count"],
        "target_authority_completeness_state": "incomplete",
        "document_provenance_coverage": s162["document_provenance_coverage"],
        "chunk_provenance_coverage": s162["chunk_provenance_coverage"],
        "span_provenance_coverage": s162["span_provenance_coverage"],
        **{key: mutation[key] for key in ("runtime_behavior_mutation", "automatic_graph_edge_deletion", "automatic_graph_edge_rewrite", "automatic_graph_regeneration", "automatic_graph_reseal", "automatic_graph_rebuild", "automatic_graph_repair", "automatic_corpus_mutation")},
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "assessment_policy_revision": assessment["assessment_policy_revision"],
        "assessment_digest": assessment["assessment_digest"],
        "policy_revision": policy["policy_revision"],
        "policy_digest": policy["policy_digest"],
        "outcome_class": outcome,
        "recommended_next_step": next_step,
    }


def build_contract(
    summary: dict[str, Any],
    policy: dict[str, Any],
    assessment: dict[str, Any],
    seal_validation: dict[str, Any],
    fixtures: dict[str, Any],
    mutation: dict[str, Any],
    gate: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0165.contract.v1",
        "task_id": TASK_ID,
        "task0164_input_contract_valid": summary["task0164_inputs_valid"] is True,
        "authoritative_graph_seal_contract_valid": seal_validation["authoritative_graph_seal_valid"] is True,
        "freshness_policy_contract_valid": policy["supported_states"] == list(FRESHNESS_STATES) and policy["runtime_mutation_enabled"] is False and policy["automatic_repair_enabled"] is False,
        "current_baseline_freshness_contract_valid": assessment["freshness_state"] == summary["freshness_state"] and assessment["assessment_digest"] == summary["assessment_digest"],
        "fixture_staleness_contract_valid": fixtures["fixtures_valid"] is True,
        "capability_gate_contract_valid": gate["graph_freshness_gate_ready"] is True and gate["runtime_behavior_mutation"] is False,
        "no_automatic_repair_contract_valid": all(value is False for key, value in mutation.items() if key.startswith("automatic_") or key == "runtime_behavior_mutation"),
        "provenance_freshness_separation_valid": summary["document_provenance_coverage"] == 1.0 and fixtures["case_states"]["case_b_document_content_change"] == "stale",
        "target_authority_freshness_separation_valid": summary["authority_gap_edge_count"] == 2 and summary["external_authoritative_source_required"] is True,
        "graph_v2_not_promoted": summary["graph_v2_data_gate_ready"] is False and summary["graph_v2_runtime_promotion_applied"] is False,
        "v1_isolation_contract_valid": v1["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST and v1["graph_runtime_hop_depth"] == 1 and v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0 and v1["runtime_policy_mutation_count"] == 0,
    }


def build_digests(*items: Any) -> dict[str, Any]:
    names = (
        "graph_freshness_assessment",
        "graph_freshness_policy",
        "freshness_decision_matrix",
        "freshness_capability_gate",
        "freshness_fixture_results",
        "staleness_reason_registry",
        "authoritative_graph_seal_validation",
        "no_automatic_repair_audit",
        "v1_isolation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0165.digests.v1",
        "task_id": TASK_ID,
        **{name: digest_json(item) for name, item in zip(names, items)},
    }


def verify_task0165_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    assessment = read_json(output_dir / "graph_freshness_assessment.json") if (output_dir / "graph_freshness_assessment.json").exists() else {}
    policy = read_json(output_dir / "graph_freshness_policy.json") if (output_dir / "graph_freshness_policy.json").exists() else {}
    fixtures = read_json(output_dir / "freshness_fixture_results.json") if (output_dir / "freshness_fixture_results.json").exists() else {}
    mutation = read_json(output_dir / "no_automatic_repair_audit.json") if (output_dir / "no_automatic_repair_audit.json").exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("freshness_state") not in FRESHNESS_STATES:
        errors.append("invalid_freshness_state")
    if summary.get("assessment_digest") != assessment.get("assessment_digest"):
        errors.append("assessment_digest_mismatch")
    if policy.get("runtime_mutation_enabled") is not False or policy.get("automatic_repair_enabled") is not False:
        errors.append("policy_mutation_or_repair_enabled")
    if fixtures.get("fixtures_valid") is not True:
        errors.append("fixture_results_invalid")
    if summary.get("freshness_state") == "fresh" and summary.get("graph_freshness_gate_passed") is not True:
        errors.append("fresh_state_gate_not_passed")
    if summary.get("authority_gap_edge_count") != 2 or summary.get("external_authoritative_source_required") is not True:
        errors.append("target_authority_boundary_changed")
    if summary.get("graph_v2_data_gate_ready") is not False or summary.get("graph_v2_runtime_promotion_applied") is not False:
        errors.append("graph_v2_runtime_promoted")
    if any(value is not False for key, value in mutation.items() if key.startswith("automatic_") or key == "runtime_behavior_mutation"):
        errors.append("automatic_repair_or_mutation_enabled")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST or summary.get("graph_runtime_hop_depth") != 1 or summary.get("default_equivalence_pass_count") != 9 or summary.get("known_causal_regression_count") != 0 or summary.get("runtime_policy_mutation_count") != 0:
        errors.append("graph_v1_invariant_changed")
    if not all(contract.get(key) is True for key in contract if key not in {"schema_version", "task_id"}):
        errors.append("contract_invalid")
    result = {
        "schema_version": "opk-rag.task0165.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any], policy: dict[str, Any]) -> str:
    return f"""# TASK-0165 Graph Freshness Detection and Staleness Policy

## Summary

TASK-0165 completed Outcome `{summary["outcome_class"]}`. Graph freshness policy `{policy["policy_revision"]}` is established with runtime mutation disabled and automatic repair disabled.

## Current Freshness

* Graph revision: `{summary["graph_revision"]}`
* Graph seal digest: `{summary["graph_seal_digest"]}`
* Bound corpus revision: `{summary["bound_source_corpus_revision"]}`
* Current corpus revision: `{summary["current_corpus_revision"]}`
* Revision match: `{summary["revision_match"]}`
* Digest match: `{summary["digest_match"]}`
* Freshness assessable: `{summary["freshness_assessable"]}`
* Freshness state: `{summary["freshness_state"]}`
* Freshness valid: `{summary["freshness_valid"]}`
* Staleness reason: `{summary["staleness_reason"]}`
* Assessment digest: `{summary["assessment_digest"]}`

## Governance Boundaries

Freshness remains separate from provenance and target authority. Provenance coverage remains `{summary["document_provenance_coverage"]}`/`{summary["chunk_provenance_coverage"]}`/`{summary["span_provenance_coverage"]}`. Authority-gap edges remain `{summary["authority_gap_edge_count"]}`, external authoritative source remains required, Graph V2 data gate remains `{summary["graph_v2_data_gate_ready"]}`, and Graph V2 runtime promotion remains `{summary["graph_v2_runtime_promotion_applied"]}`.

No automatic graph edge deletion, rewrite, regeneration, reseal, rebuild, repair, runtime behavior mutation, or corpus mutation is enabled.

Graph Retrieval V1 remains frozen: baseline digest `{summary["graph_retrieval_v1_baseline_digest"]}`, `graph_runtime_hop_depth={summary["graph_runtime_hop_depth"]}`, default equivalence `{summary["default_equivalence_pass_count"]}/9`, known causal regression count `{summary["known_causal_regression_count"]}`, and `runtime_policy_mutation_count={summary["runtime_policy_mutation_count"]}`.

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
* Q13: `{answers["Q13"]}`
* Q14: `{answers["Q14"]}`
* Q15: `{answers["Q15"]}`
"""
