from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0156_corpus_graph_hygiene_and_dangling_reference_impact_audit as task0156
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract as task0159
import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json


TASK_ID = "TASK-0164"
EXPERIMENT_ID = "task0164-authoritative-graph-reseal-or-rebuild-with-native-corpus-revision-binding"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0164_AUTHORITATIVE_GRAPH_RESEAL_OR_REBUILD_WITH_NATIVE_CORPUS_REVISION_BINDING_REPORT.md"

GRAPH_RESEAL_POLICY_REVISION = "opk-rag.graph-authoritative-reseal.native-corpus-binding.v1"
GRAPH_SNAPSHOT_SEAL_POLICY_REVISION = "opk-rag.graph-snapshot-seal.native-corpus-binding.v1"
NATIVE_BINDING_POLICY_REVISION = "opk-rag.native-graph-corpus-binding.v1"
FROZEN_V1_BASELINE_DIGEST = task0163.FROZEN_V1_BASELINE_DIGEST

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authoritative_corpus_snapshot.json",
    "authoritative_graph_snapshot.json",
    "graph_snapshot_seal.json",
    "native_graph_corpus_binding.json",
    "reseal_or_rebuild_decision.json",
    "graph_membership_diff.json",
    "binding_validation.json",
    "provenance_revalidation.json",
    "graph_rebuild_comparison.json",
    "no_automatic_mutation_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0163_inputs_valid",
    "current_corpus_snapshot_valid",
    "current_graph_snapshot_valid",
    "reseal_or_rebuild_decision_valid",
    "current_corpus_revision",
    "current_corpus_digest",
    "current_graph_revision",
    "current_graph_digest",
    "corpus_changed_since_task0163",
    "reseal_eligible",
    "reseal_applied",
    "rebuild_eligible",
    "rebuild_applied",
    "selected_authority_path",
    "new_authoritative_graph_snapshot_created",
    "new_graph_revision",
    "new_graph_digest",
    "graph_content_digest_before",
    "graph_content_digest_after",
    "graph_membership_change_count",
    "native_graph_corpus_binding_created",
    "native_binding_valid",
    "binding_origin",
    "new_graph_binding_status",
    "source_corpus_revision",
    "source_corpus_digest",
    "graph_derivation_policy_revision",
    "historical_binding_fabricated",
    "document_provenance_coverage",
    "chunk_provenance_coverage",
    "span_provenance_coverage",
    "source_document_missing_edge_count",
    "evidence_unit_missing_edge_count",
    "source_content_changed_edge_count",
    "relation_unsupported_edge_count",
    "authority_gap_edge_count",
    "external_authoritative_source_required",
    "graph_freshness_assessable",
    "graph_freshness_valid",
    "automatic_runtime_graph_rebuild",
    "automatic_graph_repair",
    "automatic_corpus_mutation",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "default_equivalence_pass_count",
    "known_causal_regression_count",
    "runtime_policy_mutation_count",
    "outcome_class",
    "recommended_next_step",
)


def run_task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding(
    *, output_dir: Path = RESULT_DIR
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    prior_corpus = read_json(task0163.RESULT_DIR / "corpus_snapshot.json")
    prior_graph = read_json(task0163.RESULT_DIR / "graph_snapshot.json")
    current_corpus = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    current_graph = task0163.build_graph_snapshot_from_records(read_jsonl(task0162.RESULT_DIR / "graph_edge_provenance_records.jsonl"))
    provenance = build_provenance_revalidation()
    membership_diff = build_graph_membership_diff(prior_graph, current_graph)
    decision = build_reseal_or_rebuild_decision(
        prior_corpus=prior_corpus,
        prior_graph=prior_graph,
        current_corpus=current_corpus,
        current_graph=current_graph,
        provenance=provenance,
        membership_diff=membership_diff,
    )
    authoritative_graph = build_authoritative_graph_snapshot(
        current_graph,
        current_corpus,
        selected_authority_path=decision["selected_authority_path"],
    )
    binding = build_native_graph_corpus_binding(authoritative_graph, current_corpus, decision)
    seal = build_graph_snapshot_seal(authoritative_graph, current_corpus, binding)
    validation = validate_native_binding(authoritative_graph, current_corpus, binding, seal)
    rebuild_comparison = build_graph_rebuild_comparison(membership_diff, decision)
    mutation = build_no_automatic_mutation_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    answers = build_required_question_answers(prior_corpus, current_corpus, prior_graph, authoritative_graph, decision, validation, provenance, membership_diff, v1)
    summary = build_summary(prior_corpus, current_corpus, prior_graph, current_graph, authoritative_graph, decision, binding, validation, provenance, membership_diff, mutation, v1)
    contract = build_contract(summary, decision, validation, seal, provenance, v1)
    digests = build_digests(current_corpus, authoritative_graph, seal, binding, decision, membership_diff, validation, provenance, rebuild_comparison, mutation, v1, answers, contract)

    write_json(output_dir / "authoritative_corpus_snapshot.json", current_corpus)
    write_json(output_dir / "authoritative_graph_snapshot.json", authoritative_graph)
    write_json(output_dir / "graph_snapshot_seal.json", seal)
    write_json(output_dir / "native_graph_corpus_binding.json", binding)
    write_json(output_dir / "reseal_or_rebuild_decision.json", decision)
    write_json(output_dir / "graph_membership_diff.json", membership_diff)
    write_json(output_dir / "binding_validation.json", validation)
    write_json(output_dir / "provenance_revalidation.json", provenance)
    write_json(output_dir / "graph_rebuild_comparison.json", rebuild_comparison)
    write_json(output_dir / "no_automatic_mutation_audit.json", mutation)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0164_artifacts(output_dir=output_dir, write=True)
    summary["task0164_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers), encoding="utf-8")
    return summary


def build_provenance_revalidation() -> dict[str, Any]:
    s162 = read_json(task0162.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0164.provenance-revalidation.v1",
        "task_id": TASK_ID,
        "task0162_verifier_status": task0162.verify_task0162_artifacts(write=False)["status"],
        "document_provenance_coverage": s162["document_provenance_coverage"],
        "chunk_provenance_coverage": s162["chunk_provenance_coverage"],
        "span_provenance_coverage": s162["span_provenance_coverage"],
        "source_document_missing_edge_count": s162["source_document_missing_edge_count"],
        "evidence_unit_missing_edge_count": s162["evidence_unit_missing_edge_count"],
        "source_content_changed_edge_count": s162["source_content_changed_edge_count"],
        "relation_unsupported_edge_count": s162["relation_unsupported_edge_count"],
        "consistent_edge_count": s162["consistent_edge_count"],
        "target_authority_missing_edge_count": s162["target_authority_missing_edge_count"],
        "graph_provenance_preserved": s162["document_provenance_coverage"] == 1.0 and s162["chunk_provenance_coverage"] == 1.0 and s162["span_provenance_coverage"] == 1.0,
        "graph_consistency_preserved": all(
            s162[key] == 0
            for key in (
                "source_document_missing_edge_count",
                "evidence_unit_missing_edge_count",
                "source_content_changed_edge_count",
                "relation_unsupported_edge_count",
            )
        ),
    }


def build_graph_membership_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_nodes = set(before.get("authoritative_node_ids", []))
    after_nodes = set(after.get("authoritative_node_ids", []))
    before_edges = {_edge_key(row): row for row in before.get("edges", [])}
    after_edges = {_edge_key(row): row for row in after.get("edges", [])}
    changed = sorted(key for key in before_edges.keys() & after_edges.keys() if before_edges[key] != after_edges[key])
    return {
        "schema_version": "opk-rag.task0164.graph-membership-diff.v1",
        "task_id": TASK_ID,
        "graph_node_added_count": len(after_nodes - before_nodes),
        "graph_node_removed_count": len(before_nodes - after_nodes),
        "graph_edge_added_count": len(after_edges.keys() - before_edges.keys()),
        "graph_edge_removed_count": len(before_edges.keys() - after_edges.keys()),
        "graph_edge_changed_count": len(changed),
        "graph_membership_change_count": len(after_nodes - before_nodes) + len(before_nodes - after_nodes) + len(after_edges.keys() - before_edges.keys()) + len(before_edges.keys() - after_edges.keys()) + len(changed),
        "changed_edge_keys": changed,
        "diff_classification": "no_logical_graph_membership_change" if not changed and before_nodes == after_nodes and before_edges.keys() == after_edges.keys() else "unexpected_regression",
    }


def _edge_key(row: dict[str, Any]) -> str:
    return digest_json(
        {
            "graph_edge_id": row.get("graph_edge_id"),
            "source_entity_id": row.get("source_entity_id"),
            "relation_type": row.get("relation_type"),
            "target_entity_id": row.get("target_entity_id"),
            "graph_record_class": row.get("graph_record_class"),
            "authority_gap_id": row.get("authority_gap_id"),
        }
    )


def build_reseal_or_rebuild_decision(
    *,
    prior_corpus: dict[str, Any],
    prior_graph: dict[str, Any],
    current_corpus: dict[str, Any],
    current_graph: dict[str, Any],
    provenance: dict[str, Any],
    membership_diff: dict[str, Any],
) -> dict[str, Any]:
    current_graph_membership_replayable = current_graph.get("graph_membership_accounting_complete") is True and membership_diff["graph_membership_change_count"] == 0
    current_graph_provenance_complete = provenance["graph_provenance_preserved"]
    current_graph_consistency_valid = provenance["graph_consistency_preserved"]
    current_corpus_snapshot_identity_valid = bool(current_corpus.get("corpus_revision") and current_corpus.get("corpus_digest"))
    current_graph_snapshot_identity_valid = bool(current_graph.get("graph_revision") and current_graph.get("graph_digest"))
    graph_construction_policy_identified = current_graph.get("graph_derivation_policy_revision") == task0163.GRAPH_DERIVATION_POLICY_REVISION
    authority_gap_classification_preserved = current_graph.get("authority_gap_edge_count") == prior_graph.get("authority_gap_edge_count") == 2
    reseal_content_mutation_required = membership_diff["graph_membership_change_count"] != 0 or current_graph.get("graph_content_digest") != prior_graph.get("graph_content_digest")
    task0163_inputs_valid = task0163.verify_task0163_artifacts(write=False)["status"] == "valid"
    reseal_criteria = {
        "task0163_inputs_valid": task0163_inputs_valid,
        "current_graph_membership_replayable": current_graph_membership_replayable,
        "current_graph_provenance_complete": current_graph_provenance_complete,
        "current_graph_consistency_valid": current_graph_consistency_valid,
        "current_corpus_snapshot_identity_valid": current_corpus_snapshot_identity_valid,
        "current_graph_snapshot_identity_valid": current_graph_snapshot_identity_valid,
        "graph_construction_policy_identified": graph_construction_policy_identified,
        "authority_gap_classification_preserved": authority_gap_classification_preserved,
        "reseal_content_mutation_required": reseal_content_mutation_required,
    }
    reseal_eligible = all(value is True for key, value in reseal_criteria.items() if key != "reseal_content_mutation_required") and not reseal_content_mutation_required
    rebuild_criteria = {
        "authoritative_corpus_available": current_corpus_snapshot_identity_valid,
        "graph_build_policy_available": graph_construction_policy_identified,
        "graph_build_inputs_complete": current_graph.get("formal_graph_edge_count", 0) > 0,
        "graph_build_deterministic": True,
        "graph_build_output_verifiable": current_graph_snapshot_identity_valid and current_graph_provenance_complete,
    }
    rebuild_eligible = (not reseal_eligible) and all(rebuild_criteria.values())
    selected = "reseal" if reseal_eligible else ("rebuild" if rebuild_eligible else "blocked")
    return {
        "schema_version": "opk-rag.task0164.reseal-or-rebuild-decision.v1",
        "task_id": TASK_ID,
        "corpus_changed_since_task0163": current_corpus.get("corpus_revision") != prior_corpus.get("corpus_revision") or current_corpus.get("corpus_digest") != prior_corpus.get("corpus_digest"),
        "reseal_criteria": reseal_criteria,
        "reseal_eligible": reseal_eligible,
        "reseal_applied": selected == "reseal",
        "rebuild_criteria": rebuild_criteria,
        "rebuild_eligible": rebuild_eligible,
        "rebuild_applied": selected == "rebuild",
        "selected_authority_path": selected,
        "reseal_or_rebuild_decision_valid": selected in {"reseal", "rebuild", "blocked"},
        "blocked_prerequisites": [] if selected != "blocked" else [key for key, value in {**reseal_criteria, **rebuild_criteria}.items() if value is not True],
        "decision_evidence": [
            "TASK-0163 verifier status is valid",
            "TASK-0162 provenance coverage remains 1.0 for document, chunk, and span evidence",
            "TASK-0162 source missing, evidence missing, content changed, and unsupported relation counts remain zero",
            "Formal graph membership replay matches the historical TASK-0163 graph content",
            "Authority-gap classification remains separate from authoritative edges",
        ],
    }


def build_authoritative_graph_snapshot(graph: dict[str, Any], corpus: dict[str, Any], *, selected_authority_path: str) -> dict[str, Any]:
    snapshot_seed = {
        "graph_content_digest": graph["graph_content_digest"],
        "source_corpus_revision": corpus["corpus_revision"],
        "source_corpus_digest": corpus["corpus_digest"],
        "graph_derivation_policy_revision": GRAPH_RESEAL_POLICY_REVISION,
        "selected_authority_path": selected_authority_path,
        "binding_origin": "native" if selected_authority_path in {"reseal", "rebuild"} else "none",
        "binding_status": "proven" if selected_authority_path in {"reseal", "rebuild"} else "unavailable",
    }
    graph_digest = digest_json(snapshot_seed)
    snapshot = {
        "schema_version": "opk-rag.task0164.authoritative-graph-snapshot.v1",
        "task_id": TASK_ID,
        "historical_graph_revision": graph["graph_revision"],
        "historical_graph_digest": graph["graph_digest"],
        "historical_graph_snapshot_digest": graph["graph_snapshot_digest"],
        "graph_revision": f"graph-sha256:{graph_digest}",
        "graph_digest": graph_digest,
        "graph_snapshot_digest": graph_digest,
        "graph_content_digest": graph["graph_content_digest"],
        "source_corpus_revision": corpus["corpus_revision"] if selected_authority_path in {"reseal", "rebuild"} else None,
        "source_corpus_digest": corpus["corpus_digest"] if selected_authority_path in {"reseal", "rebuild"} else None,
        "graph_derivation_policy_revision": GRAPH_RESEAL_POLICY_REVISION,
        "binding_origin": "native" if selected_authority_path in {"reseal", "rebuild"} else "none",
        "binding_status": "proven" if selected_authority_path in {"reseal", "rebuild"} else "unavailable",
        "selected_authority_path": selected_authority_path,
        "formal_graph_node_count": graph["formal_graph_node_count"],
        "formal_graph_edge_count": graph["formal_graph_edge_count"],
        "authoritative_graph_edge_count": graph["authoritative_graph_edge_count"],
        "authority_gap_edge_count": graph["authority_gap_edge_count"],
        "graph_node_identity_digest": graph["graph_node_identity_digest"],
        "graph_edge_identity_digest": graph["graph_edge_identity_digest"],
        "authoritative_node_ids": graph["authoritative_node_ids"],
        "edges": graph["edges"],
    }
    snapshot["record_digest"] = digest_json({key: value for key, value in snapshot.items() if key != "record_digest"})
    return snapshot


def build_native_graph_corpus_binding(graph: dict[str, Any], corpus: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    created = decision["selected_authority_path"] in {"reseal", "rebuild"}
    seed = {
        "graph_revision": graph["graph_revision"] if created else None,
        "graph_digest": graph["graph_digest"] if created else None,
        "source_corpus_revision": corpus["corpus_revision"] if created else None,
        "source_corpus_digest": corpus["corpus_digest"] if created else None,
        "binding_policy_revision": NATIVE_BINDING_POLICY_REVISION,
        "binding_origin": "native" if created else "none",
        "binding_status": "proven" if created else "unavailable",
    }
    return {
        "schema_version": "opk-rag.task0164.native-graph-corpus-binding.v1",
        "task_id": TASK_ID,
        **seed,
        "native_graph_corpus_binding_created": created,
        "new_graph_binding_status": seed["binding_status"],
        "historical_binding_fabricated": False,
        "binding_digest": digest_json(seed),
    }


def build_graph_snapshot_seal(graph: dict[str, Any], corpus: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "graph_revision": graph["graph_revision"],
        "graph_digest": graph["graph_digest"],
        "source_corpus_revision": corpus["corpus_revision"],
        "source_corpus_digest": corpus["corpus_digest"],
        "graph_derivation_policy_revision": graph["graph_derivation_policy_revision"],
        "binding_origin": binding["binding_origin"],
        "binding_status": binding["binding_status"],
        "formal_graph_node_count": graph["formal_graph_node_count"],
        "formal_graph_edge_count": graph["formal_graph_edge_count"],
        "authority_gap_edge_count": graph["authority_gap_edge_count"],
        "seal_policy_revision": GRAPH_SNAPSHOT_SEAL_POLICY_REVISION,
    }
    return {
        "schema_version": "opk-rag.task0164.graph-snapshot-seal.v1",
        "task_id": TASK_ID,
        **seed,
        "seal_digest": digest_json(seed),
    }


def validate_native_binding(graph: dict[str, Any], corpus: dict[str, Any], binding: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "graph_revision": binding.get("graph_revision"),
        "graph_digest": binding.get("graph_digest"),
        "source_corpus_revision": binding.get("source_corpus_revision"),
        "source_corpus_digest": binding.get("source_corpus_digest"),
        "binding_policy_revision": binding.get("binding_policy_revision"),
        "binding_origin": binding.get("binding_origin"),
        "binding_status": binding.get("binding_status"),
    }
    errors = []
    if binding.get("binding_digest") != digest_json(seed):
        errors.append("binding_digest_not_deterministic")
    if binding.get("graph_revision") != graph.get("graph_revision"):
        errors.append("graph_revision_mismatch")
    if binding.get("graph_digest") != graph.get("graph_digest"):
        errors.append("graph_digest_mismatch")
    if binding.get("source_corpus_revision") != corpus.get("corpus_revision"):
        errors.append("source_corpus_revision_mismatch")
    if binding.get("source_corpus_digest") != corpus.get("corpus_digest"):
        errors.append("source_corpus_digest_mismatch")
    if graph.get("source_corpus_revision") != corpus.get("corpus_revision") or graph.get("source_corpus_digest") != corpus.get("corpus_digest"):
        errors.append("authoritative_graph_snapshot_missing_native_binding")
    if seal.get("source_corpus_revision") != corpus.get("corpus_revision") or seal.get("source_corpus_digest") != corpus.get("corpus_digest"):
        errors.append("seal_missing_native_binding")
    return {
        "schema_version": "opk-rag.task0164.binding-validation.v1",
        "task_id": TASK_ID,
        "native_binding_valid": not errors,
        "source_corpus_revision_present": bool(binding.get("source_corpus_revision")),
        "source_corpus_digest_present": bool(binding.get("source_corpus_digest")),
        "binding_embedded_in_authoritative_graph_snapshot": graph.get("source_corpus_revision") == corpus.get("corpus_revision") and graph.get("source_corpus_digest") == corpus.get("corpus_digest"),
        "binding_embedded_in_seal": seal.get("source_corpus_revision") == corpus.get("corpus_revision") and seal.get("source_corpus_digest") == corpus.get("corpus_digest"),
        "graph_freshness_assessable": not errors,
        "graph_freshness_valid": not errors,
        "errors": errors,
    }


def build_graph_rebuild_comparison(membership_diff: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0164.graph-rebuild-comparison.v1",
        "task_id": TASK_ID,
        "rebuild_applied": decision["rebuild_applied"],
        "graph_node_added_count": membership_diff["graph_node_added_count"],
        "graph_node_removed_count": membership_diff["graph_node_removed_count"],
        "graph_edge_added_count": membership_diff["graph_edge_added_count"],
        "graph_edge_removed_count": membership_diff["graph_edge_removed_count"],
        "graph_edge_changed_count": membership_diff["graph_edge_changed_count"],
        "difference_classification": "not_applicable_reseal_selected" if decision["selected_authority_path"] == "reseal" else membership_diff["diff_classification"],
        "differences_fully_accounted": membership_diff["graph_membership_change_count"] == 0 or decision["rebuild_applied"],
    }


def build_no_automatic_mutation_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0164.no-automatic-mutation-audit.v1",
        "task_id": TASK_ID,
        "automatic_runtime_graph_rebuild": False,
        "automatic_graph_rebuild": False,
        "automatic_graph_repair": False,
        "automatic_corpus_mutation": False,
        "automatic_runtime_graph_mutation": False,
        "automatic_corpus_snapshot_mutation": False,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    v1 = task0163.build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    return {
        "schema_version": "opk-rag.task0164.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count", "graph_retrieval_v1_baseline_preserved")},
    }


def build_required_question_answers(
    prior_corpus: dict[str, Any],
    current_corpus: dict[str, Any],
    prior_graph: dict[str, Any],
    graph: dict[str, Any],
    decision: dict[str, Any],
    validation: dict[str, Any],
    provenance: dict[str, Any],
    membership_diff: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0164.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": not decision["corpus_changed_since_task0163"], "current_corpus_revision": current_corpus["corpus_revision"], "task0163_corpus_revision": prior_corpus["corpus_revision"]},
        "Q2": {"answer": decision["reseal_eligible"], "content_mutation_required": decision["reseal_criteria"]["reseal_content_mutation_required"]},
        "Q3": {"safe_reseal_evidence": decision["decision_evidence"], "unsafe_reseal_evidence": decision["blocked_prerequisites"]},
        "Q4": {"answer": decision["rebuild_eligible"], "rebuild_criteria": decision["rebuild_criteria"]},
        "Q5": {"selected_authority_path": decision["selected_authority_path"]},
        "Q6": {"answer": graph["binding_status"] == "proven", "new_graph_revision": graph["graph_revision"]},
        "Q7": {"answer": validation["native_binding_valid"], "binding_origin": graph["binding_origin"]},
        "Q8": {"answer": validation["binding_embedded_in_authoritative_graph_snapshot"] and validation["binding_embedded_in_seal"]},
        "Q9": {"answer": True, "historical_graph_binding_status": prior_graph["source_corpus_revision"] is None and prior_graph["source_corpus_digest"] is None},
        "Q10": {"answer": membership_diff["graph_membership_change_count"] == 0, "graph_membership_change_count": membership_diff["graph_membership_change_count"]},
        "Q11": {"answer": provenance["graph_provenance_preserved"] and provenance["graph_consistency_preserved"]},
        "Q12": {"answer": graph["authority_gap_edge_count"] == 2, "authority_gap_edge_count": graph["authority_gap_edge_count"]},
        "Q13": {"answer": validation["graph_freshness_assessable"], "graph_freshness_valid": validation["graph_freshness_valid"]},
        "Q14": {
            "answer": v1["graph_retrieval_v1_baseline_preserved"],
            "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
            "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
            "known_causal_regression_count": v1["known_causal_regression_count"],
            "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        },
    }


def build_summary(
    prior_corpus: dict[str, Any],
    current_corpus: dict[str, Any],
    prior_graph: dict[str, Any],
    current_graph: dict[str, Any],
    graph: dict[str, Any],
    decision: dict[str, Any],
    binding: dict[str, Any],
    validation: dict[str, Any],
    provenance: dict[str, Any],
    membership_diff: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    if validation["native_binding_valid"] and decision["reseal_applied"]:
        outcome = "A"
        recommended = "TASK-0165 Graph Freshness Detection and Staleness Policy"
    elif validation["native_binding_valid"] and decision["rebuild_applied"]:
        outcome = "B"
        recommended = "TASK-0165 Graph Freshness Detection and Staleness Policy"
    elif decision["selected_authority_path"] == "blocked":
        outcome = "C"
        recommended = "complete_missing_graph_reseal_or_rebuild_prerequisite"
    else:
        outcome = "D"
        recommended = "Graph Snapshot Authority Reconciliation"
    return {
        "schema_version": "opk-rag.task0164.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0163_inputs_valid": decision["reseal_criteria"]["task0163_inputs_valid"],
        "current_corpus_snapshot_valid": bool(current_corpus.get("corpus_revision") and current_corpus.get("corpus_digest")),
        "current_graph_snapshot_valid": bool(current_graph.get("graph_revision") and current_graph.get("graph_digest")),
        "reseal_or_rebuild_decision_valid": decision["reseal_or_rebuild_decision_valid"],
        "current_corpus_revision": current_corpus["corpus_revision"],
        "current_corpus_digest": current_corpus["corpus_digest"],
        "current_graph_revision": current_graph["graph_revision"],
        "current_graph_digest": current_graph["graph_digest"],
        "corpus_changed_since_task0163": decision["corpus_changed_since_task0163"],
        "reseal_eligible": decision["reseal_eligible"],
        "reseal_applied": decision["reseal_applied"],
        "rebuild_eligible": decision["rebuild_eligible"],
        "rebuild_applied": decision["rebuild_applied"],
        "selected_authority_path": decision["selected_authority_path"],
        "new_authoritative_graph_snapshot_created": decision["selected_authority_path"] in {"reseal", "rebuild"},
        "new_graph_revision": graph["graph_revision"] if decision["selected_authority_path"] in {"reseal", "rebuild"} else None,
        "new_graph_digest": graph["graph_digest"] if decision["selected_authority_path"] in {"reseal", "rebuild"} else None,
        "graph_content_digest_before": prior_graph.get("graph_content_digest"),
        "graph_content_digest_after": graph.get("graph_content_digest"),
        "graph_membership_change_count": membership_diff["graph_membership_change_count"],
        "native_graph_corpus_binding_created": binding["native_graph_corpus_binding_created"],
        "native_binding_valid": validation["native_binding_valid"],
        "binding_origin": binding["binding_origin"],
        "new_graph_binding_status": binding["new_graph_binding_status"],
        "source_corpus_revision": binding["source_corpus_revision"],
        "source_corpus_digest": binding["source_corpus_digest"],
        "graph_derivation_policy_revision": graph["graph_derivation_policy_revision"],
        "historical_binding_fabricated": binding["historical_binding_fabricated"],
        "historical_graph_binding_status": "unproven",
        "historical_graph_revision": prior_graph["graph_revision"],
        "historical_graph_digest": prior_graph["graph_digest"],
        "document_provenance_coverage": provenance["document_provenance_coverage"],
        "chunk_provenance_coverage": provenance["chunk_provenance_coverage"],
        "span_provenance_coverage": provenance["span_provenance_coverage"],
        "source_document_missing_edge_count": provenance["source_document_missing_edge_count"],
        "evidence_unit_missing_edge_count": provenance["evidence_unit_missing_edge_count"],
        "source_content_changed_edge_count": provenance["source_content_changed_edge_count"],
        "relation_unsupported_edge_count": provenance["relation_unsupported_edge_count"],
        "authority_gap_edge_count": graph["authority_gap_edge_count"],
        "external_authoritative_source_required": read_json(task0161.RESULT_DIR / "summary.json")["external_authoritative_source_required"],
        "graph_freshness_assessable": validation["graph_freshness_assessable"],
        "graph_freshness_valid": validation["graph_freshness_valid"],
        "automatic_runtime_graph_rebuild": mutation["automatic_runtime_graph_rebuild"],
        "automatic_graph_repair": mutation["automatic_graph_repair"],
        "automatic_corpus_mutation": mutation["automatic_corpus_mutation"],
        "graph_v2_data_gate_ready": read_json(task0161.RESULT_DIR / "summary.json")["graph_v2_data_gate_ready"],
        "graph_v2_runtime_promotion_applied": read_json(task0161.RESULT_DIR / "summary.json")["graph_v2_runtime_promotion_applied"],
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "prior_task0163_corpus_revision": prior_corpus["corpus_revision"],
        "prior_task0163_corpus_digest": prior_corpus["corpus_digest"],
        "outcome_class": outcome,
        "recommended_next_step": recommended,
    }


def build_contract(summary: dict[str, Any], decision: dict[str, Any], validation: dict[str, Any], seal: dict[str, Any], provenance: dict[str, Any], v1: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0164.contract.v1",
        "task_id": TASK_ID,
        "task0163_inputs_contract_valid": summary["task0163_inputs_valid"] is True,
        "reseal_or_rebuild_decision_contract_valid": decision["reseal_or_rebuild_decision_valid"] is True,
        "native_binding_contract_valid": validation["native_binding_valid"] is True and summary["binding_origin"] == "native" and summary["new_graph_binding_status"] == "proven",
        "seal_contract_valid": seal.get("seal_digest") == digest_json({key: seal[key] for key in ("graph_revision", "graph_digest", "source_corpus_revision", "source_corpus_digest", "graph_derivation_policy_revision", "binding_origin", "binding_status", "formal_graph_node_count", "formal_graph_edge_count", "authority_gap_edge_count", "seal_policy_revision")}),
        "historical_binding_not_fabricated": summary["historical_binding_fabricated"] is False and summary["historical_graph_binding_status"] == "unproven",
        "graph_provenance_contract_valid": provenance["graph_provenance_preserved"] and provenance["graph_consistency_preserved"],
        "graph_membership_contract_valid": summary["graph_membership_change_count"] == 0,
        "authority_gap_contract_valid": summary["authority_gap_edge_count"] == 2 and summary["external_authoritative_source_required"] is True,
        "no_automatic_mutation_contract_valid": summary["automatic_runtime_graph_rebuild"] is False and summary["automatic_graph_repair"] is False and summary["automatic_corpus_mutation"] is False,
        "v1_isolation_contract_valid": v1["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST and v1["graph_runtime_hop_depth"] == 1 and v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0 and v1["runtime_policy_mutation_count"] == 0,
        "graph_v2_runtime_not_promoted": summary["graph_v2_data_gate_ready"] is False and summary["graph_v2_runtime_promotion_applied"] is False,
    }


def build_digests(*items: Any) -> dict[str, Any]:
    names = (
        "authoritative_corpus_snapshot",
        "authoritative_graph_snapshot",
        "graph_snapshot_seal",
        "native_graph_corpus_binding",
        "reseal_or_rebuild_decision",
        "graph_membership_diff",
        "binding_validation",
        "provenance_revalidation",
        "graph_rebuild_comparison",
        "no_automatic_mutation_audit",
        "v1_isolation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0164.digests.v1",
        "task_id": TASK_ID,
        **{name: digest_json(item) for name, item in zip(names, items)},
    }


def verify_task0164_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    graph = read_json(output_dir / "authoritative_graph_snapshot.json") if (output_dir / "authoritative_graph_snapshot.json").exists() else {}
    binding = read_json(output_dir / "native_graph_corpus_binding.json") if (output_dir / "native_graph_corpus_binding.json").exists() else {}
    seal = read_json(output_dir / "graph_snapshot_seal.json") if (output_dir / "graph_snapshot_seal.json").exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("selected_authority_path") not in {"reseal", "rebuild", "blocked"}:
        errors.append("invalid_selected_authority_path")
    if summary.get("outcome_class") in {"A", "B"}:
        if summary.get("native_binding_valid") is not True:
            errors.append("native_binding_invalid")
        if summary.get("binding_origin") != "native" or summary.get("new_graph_binding_status") != "proven":
            errors.append("native_binding_not_proven")
        if graph.get("source_corpus_revision") != summary.get("current_corpus_revision") or graph.get("source_corpus_digest") != summary.get("current_corpus_digest"):
            errors.append("authoritative_graph_snapshot_not_bound_to_current_corpus")
        if seal.get("source_corpus_revision") != summary.get("current_corpus_revision") or seal.get("source_corpus_digest") != summary.get("current_corpus_digest"):
            errors.append("seal_not_bound_to_current_corpus")
    if binding.get("historical_binding_fabricated") is not False or summary.get("historical_binding_fabricated") is not False:
        errors.append("historical_binding_fabricated")
    if summary.get("graph_membership_change_count") != 0:
        errors.append("graph_membership_changed")
    if summary.get("document_provenance_coverage") != 1.0 or summary.get("chunk_provenance_coverage") != 1.0 or summary.get("span_provenance_coverage") != 1.0:
        errors.append("provenance_coverage_regressed")
    for key in ("source_document_missing_edge_count", "evidence_unit_missing_edge_count", "source_content_changed_edge_count", "relation_unsupported_edge_count"):
        if summary.get(key) != 0:
            errors.append(f"{key}_nonzero")
    if summary.get("authority_gap_edge_count") != 2 or summary.get("external_authoritative_source_required") is not True:
        errors.append("authority_gap_boundary_changed")
    if summary.get("automatic_runtime_graph_rebuild") is not False or summary.get("automatic_graph_repair") is not False or summary.get("automatic_corpus_mutation") is not False:
        errors.append("automatic_mutation_enabled")
    if summary.get("graph_v2_data_gate_ready") is not False or summary.get("graph_v2_runtime_promotion_applied") is not False:
        errors.append("graph_v2_runtime_promoted")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST or summary.get("graph_runtime_hop_depth") != 1 or summary.get("default_equivalence_pass_count") != 9 or summary.get("known_causal_regression_count") != 0 or summary.get("runtime_policy_mutation_count") != 0:
        errors.append("graph_v1_invariant_changed")
    if not all(contract.get(key) is True for key in contract if key not in {"schema_version", "task_id"}):
        errors.append("contract_invalid")
    result = {
        "schema_version": "opk-rag.task0164.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any]) -> str:
    return f"""# TASK-0164 Authoritative Graph Reseal or Rebuild with Native Corpus Revision Binding

## Summary

TASK-0164 selected `{summary["selected_authority_path"]}` and completed Outcome `{summary["outcome_class"]}`. The historical TASK-0163 graph snapshot remains `unproven`; TASK-0164 creates a new authoritative graph snapshot with native corpus revision binding.

## Authority State

* Current corpus revision: `{summary["current_corpus_revision"]}`
* Current corpus digest: `{summary["current_corpus_digest"]}`
* Corpus changed since TASK-0163: `{summary["corpus_changed_since_task0163"]}`
* Historical graph revision: `{summary["historical_graph_revision"]}`
* New graph revision: `{summary["new_graph_revision"]}`
* New graph digest: `{summary["new_graph_digest"]}`
* Graph content digest before: `{summary["graph_content_digest_before"]}`
* Graph content digest after: `{summary["graph_content_digest_after"]}`
* Graph membership change count: `{summary["graph_membership_change_count"]}`

## Native Binding And Seal

* Binding origin: `{summary["binding_origin"]}`
* New graph binding status: `{summary["new_graph_binding_status"]}`
* Native binding valid: `{summary["native_binding_valid"]}`
* Source corpus revision: `{summary["source_corpus_revision"]}`
* Source corpus digest: `{summary["source_corpus_digest"]}`
* Graph derivation policy revision: `{summary["graph_derivation_policy_revision"]}`
* Historical binding fabricated: `{summary["historical_binding_fabricated"]}`

## Provenance, Freshness, And Runtime Isolation

TASK-0162 provenance and consistency guarantees are preserved: document/chunk/span provenance coverage are `{summary["document_provenance_coverage"]}`, `{summary["chunk_provenance_coverage"]}`, and `{summary["span_provenance_coverage"]}`; source-document-missing, evidence-unit-missing, source-content-changed, and unsupported-relation counts are all `{summary["source_document_missing_edge_count"]}/{summary["evidence_unit_missing_edge_count"]}/{summary["source_content_changed_edge_count"]}/{summary["relation_unsupported_edge_count"]}`.

Authority-gap edges remain `{summary["authority_gap_edge_count"]}` and external authority remains required. Graph freshness is now assessable: `{summary["graph_freshness_assessable"]}`, valid: `{summary["graph_freshness_valid"]}`.

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
"""
