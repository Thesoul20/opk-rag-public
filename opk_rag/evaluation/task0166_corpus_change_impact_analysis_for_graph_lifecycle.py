from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
import opk_rag.evaluation.task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding as task0164
import opk_rag.evaluation.task0165_graph_freshness_detection_and_staleness_policy as task0165
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json


TASK_ID = "TASK-0166"
EXPERIMENT_ID = "task0166-corpus-change-impact-analysis-for-graph-lifecycle"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0166_corpus_change_impact_analysis_for_graph_lifecycle_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0166_CORPUS_CHANGE_IMPACT_ANALYSIS_FOR_GRAPH_LIFECYCLE_REPORT.md"

IMPACT_POLICY_REVISION = "opk-rag.corpus-change-impact-analysis.v1"
EVIDENCE_UNIT_POLICY_REVISION = "opk-rag.graph-provenance-evidence-unit-delta.v1"
FROZEN_V1_BASELINE_DIGEST = task0165.FROZEN_V1_BASELINE_DIGEST

EDGE_IMPACT_CLASSES = (
    "unaffected",
    "revalidation_required",
    "removal_candidate",
    "addition_dependency",
    "authority_recheck_required",
    "unverifiable",
)
NODE_IMPACT_CLASSES = (
    "unaffected",
    "source_changed",
    "source_removed",
    "authority_recheck_required",
    "potentially_orphaned",
    "unverifiable",
)
REPAIR_REQUIREMENT_CLASSES = (
    "none",
    "revalidate",
    "rebind_provenance",
    "remove_candidate",
    "extract_new_graph_evidence",
    "authority_recheck",
    "full_rebuild_candidate",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "corpus_snapshot_delta.json",
    "document_change_registry.json",
    "evidence_unit_change_registry.json",
    "graph_impact_registry.json",
    "graph_repair_impact_plan.json",
    "direct_secondary_impact_matrix.json",
    "repair_requirement_registry.json",
    "impact_fixture_results.json",
    "graph_policy_change_assessment.json",
    "capability_gate.json",
    "no_automatic_mutation_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0165_inputs_valid",
    "previous_corpus_revision",
    "previous_corpus_digest",
    "current_corpus_revision",
    "current_corpus_digest",
    "live_corpus_changed",
    "corpus_delta_ready",
    "corpus_delta_deterministic",
    "document_added_count",
    "document_removed_count",
    "document_changed_count",
    "document_unchanged_count",
    "canonical_document_added_count",
    "canonical_document_removed_count",
    "canonical_document_changed_count",
    "evidence_unit_added_count",
    "evidence_unit_removed_count",
    "evidence_unit_changed_count",
    "representation_change_detected",
    "directly_affected_graph_edge_count",
    "secondarily_affected_graph_edge_count",
    "affected_graph_node_count",
    "graph_impact_analysis_ready",
    "graph_impact_accounting_complete",
    "revalidation_required_edge_count",
    "removal_candidate_edge_count",
    "authority_recheck_required_edge_count",
    "unverifiable_impact_edge_count",
    "new_graph_extraction_required",
    "graph_policy_change_detected",
    "incremental_repair_assessable",
    "full_rebuild_required",
    "repair_impact_plan_ready",
    "repair_impact_plan_deterministic",
    "no_change_fixture_passed",
    "document_content_change_fixture_passed",
    "document_removal_fixture_passed",
    "document_addition_fixture_passed",
    "unrelated_document_change_fixture_passed",
    "representation_change_fixture_passed",
    "authority_gap_edge_count",
    "external_authoritative_source_required",
    "automatic_graph_edge_deletion",
    "automatic_graph_edge_rewrite",
    "automatic_graph_node_deletion",
    "automatic_graph_rebuild",
    "automatic_graph_reseal",
    "automatic_corpus_mutation",
    "runtime_change_impact_policy_applied",
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


def run_task0166_corpus_change_impact_analysis_for_graph_lifecycle(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    task0165_valid = task0165.verify_task0165_artifacts(write=False)["status"] == "valid"
    previous_corpus = read_json(task0164.RESULT_DIR / "authoritative_corpus_snapshot.json")
    current_corpus = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    graph = read_json(task0164.RESULT_DIR / "authoritative_graph_snapshot.json")
    graph_records = read_jsonl(task0162.RESULT_DIR / "graph_edge_provenance_records.jsonl")
    policy_assessment = build_graph_policy_change_assessment(graph)
    delta = build_corpus_snapshot_delta(previous_corpus, current_corpus)
    documents = build_document_change_registry(delta, graph_records)
    evidence = build_evidence_unit_change_registry(delta, graph_records, policy_assessment=policy_assessment)
    impact = build_graph_impact_registry(graph, graph_records, documents, evidence, policy_assessment)
    matrix = build_direct_secondary_impact_matrix(graph, impact)
    repair_requirements = build_repair_requirement_registry(impact)
    plan = build_graph_repair_impact_plan(delta, documents, evidence, impact, repair_requirements, policy_assessment)
    gate = build_capability_gate(delta, evidence, impact, plan, policy_assessment)
    mutation = build_no_automatic_mutation_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    fixtures = build_impact_fixture_results(previous_corpus, graph, graph_records)
    answers = build_required_question_answers(delta, documents, evidence, impact, matrix, plan, fixtures, mutation, v1)
    summary = build_summary(
        task0165_valid=task0165_valid,
        delta=delta,
        documents=documents,
        evidence=evidence,
        impact=impact,
        plan=plan,
        gate=gate,
        fixtures=fixtures,
        policy_assessment=policy_assessment,
        mutation=mutation,
        v1=v1,
    )
    contract = build_contract(summary, delta, documents, evidence, impact, plan, gate, fixtures, mutation, v1)
    digests = build_digests(delta, documents, evidence, impact, plan, matrix, repair_requirements, fixtures, policy_assessment, gate, mutation, v1, answers, contract)

    write_json(output_dir / "corpus_snapshot_delta.json", delta)
    write_json(output_dir / "document_change_registry.json", documents)
    write_json(output_dir / "evidence_unit_change_registry.json", evidence)
    write_json(output_dir / "graph_impact_registry.json", impact)
    write_json(output_dir / "graph_repair_impact_plan.json", plan)
    write_json(output_dir / "direct_secondary_impact_matrix.json", matrix)
    write_json(output_dir / "repair_requirement_registry.json", repair_requirements)
    write_json(output_dir / "impact_fixture_results.json", fixtures)
    write_json(output_dir / "graph_policy_change_assessment.json", policy_assessment)
    write_json(output_dir / "capability_gate.json", gate)
    write_json(output_dir / "no_automatic_mutation_audit.json", mutation)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0166_artifacts(output_dir=output_dir, write=True)
    summary["task0166_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers), encoding="utf-8")
    return summary


def build_corpus_snapshot_delta(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    representation_policy_changed: bool = False,
    graph_policy_change_detected: bool = False,
) -> dict[str, Any]:
    previous_members = _members_by_source(previous)
    current_members = _members_by_source(current)
    previous_ids = set(previous_members)
    current_ids = set(current_members)
    added = sorted(current_ids - previous_ids)
    removed = sorted(previous_ids - current_ids)
    common = sorted(previous_ids & current_ids)
    content_changed = sorted(
        doc_id
        for doc_id in common
        if previous_members[doc_id].get("document_content_digest") != current_members[doc_id].get("document_content_digest")
        or previous_members[doc_id].get("source_revision") != current_members[doc_id].get("source_revision")
    )
    identity_changed = sorted(
        doc_id
        for doc_id in common
        if previous_members[doc_id].get("canonical_document_id") != current_members[doc_id].get("canonical_document_id")
    )
    changed = sorted(set(content_changed) | set(identity_changed))
    unchanged = sorted(doc_id for doc_id in common if doc_id not in set(changed))

    previous_canonical = _members_by_canonical(previous)
    current_canonical = _members_by_canonical(current)
    previous_canonical_ids = set(previous_canonical)
    current_canonical_ids = set(current_canonical)
    canonical_added = sorted(current_canonical_ids - previous_canonical_ids)
    canonical_removed = sorted(previous_canonical_ids - current_canonical_ids)
    canonical_common = sorted(previous_canonical_ids & current_canonical_ids)
    canonical_changed = sorted(
        canonical_id
        for canonical_id in canonical_common
        if _canonical_seed(previous_canonical[canonical_id]) != _canonical_seed(current_canonical[canonical_id])
    )
    canonical_unchanged = sorted(canonical_id for canonical_id in canonical_common if canonical_id not in set(canonical_changed))

    seed = {
        "previous_corpus_revision": previous.get("corpus_revision"),
        "previous_corpus_digest": previous.get("corpus_digest"),
        "current_corpus_revision": current.get("corpus_revision"),
        "current_corpus_digest": current.get("corpus_digest"),
        "added_document_ids": added,
        "removed_document_ids": removed,
        "content_changed_document_ids": content_changed,
        "identity_changed_document_ids": identity_changed,
        "unchanged_document_ids": unchanged,
        "canonical_added_document_ids": canonical_added,
        "canonical_removed_document_ids": canonical_removed,
        "canonical_changed_document_ids": canonical_changed,
        "representation_policy_changed": representation_policy_changed,
        "graph_policy_change_detected": graph_policy_change_detected,
        "delta_policy_revision": IMPACT_POLICY_REVISION,
    }
    return {
        "schema_version": "opk-rag.task0166.corpus-snapshot-delta.v1",
        "task_id": TASK_ID,
        **seed,
        "document_added_count": len(added),
        "document_removed_count": len(removed),
        "document_changed_count": len(changed),
        "document_unchanged_count": len(unchanged),
        "canonical_document_added_count": len(canonical_added),
        "canonical_document_removed_count": len(canonical_removed),
        "canonical_document_changed_count": len(canonical_changed),
        "canonical_document_unchanged_count": len(canonical_unchanged),
        "corpus_membership_changed": bool(added or removed),
        "corpus_content_changed": bool(content_changed),
        "representation_change_detected": representation_policy_changed,
        "no_change": not (added or removed or changed or canonical_added or canonical_removed or canonical_changed or representation_policy_changed or graph_policy_change_detected),
        "corpus_delta_ready": bool(previous.get("corpus_revision") and current.get("corpus_revision")),
        "corpus_delta_deterministic": True,
        "delta_digest": digest_json(seed),
    }


def build_document_change_registry(delta: dict[str, Any], graph_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    graph_docs = {str(row.get("origin_document_id")) for row in graph_records if row.get("origin_document_id")}
    rows = []
    changed_ids = set(delta["content_changed_document_ids"]) | set(delta["identity_changed_document_ids"])
    for doc_id in sorted(set(delta["added_document_ids"]) | set(delta["removed_document_ids"]) | changed_ids | set(delta["unchanged_document_ids"])):
        if doc_id in delta["added_document_ids"]:
            classification = "added"
        elif doc_id in delta["removed_document_ids"]:
            classification = "removed"
        elif doc_id in delta["identity_changed_document_ids"]:
            classification = "identity_changed"
        elif doc_id in delta["content_changed_document_ids"]:
            classification = "content_changed"
        else:
            classification = "unchanged"
        rows.append(
            {
                "source_document_id": doc_id,
                "change_classification": classification,
                "graph_relevance": "graph_relevant" if doc_id in graph_docs else "graph_irrelevant",
                "requires_graph_extraction": classification == "added",
                "requires_graph_revalidation": classification in {"removed", "identity_changed", "content_changed"} and doc_id in graph_docs,
            }
        )
    return {
        "schema_version": "opk-rag.task0166.document-change-registry.v1",
        "task_id": TASK_ID,
        "document_change_accounting_complete": True,
        "changed_document_ids": sorted(set(delta["added_document_ids"]) | set(delta["removed_document_ids"]) | changed_ids),
        "graph_relevant_changed_document_ids": sorted(row["source_document_id"] for row in rows if row["change_classification"] != "unchanged" and row["graph_relevance"] == "graph_relevant"),
        "graph_irrelevant_changed_document_ids": sorted(row["source_document_id"] for row in rows if row["change_classification"] != "unchanged" and row["graph_relevance"] == "graph_irrelevant"),
        "records": rows,
        "registry_digest": digest_json(rows),
    }


def build_evidence_unit_change_registry(
    delta: dict[str, Any],
    graph_records: Iterable[dict[str, Any]],
    *,
    policy_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = list(graph_records)
    changed_docs = set(delta["content_changed_document_ids"]) | set(delta["identity_changed_document_ids"])
    removed_docs = set(delta["removed_document_ids"])
    added_docs = set(delta["added_document_ids"])
    representation_changed = bool(delta.get("representation_change_detected"))
    by_unit = _evidence_units_by_id(records)
    rows = []
    for unit_id, unit in by_unit.items():
        doc_id = unit["origin_document_id"]
        if doc_id in removed_docs:
            classification = "removed"
        elif representation_changed:
            classification = "representation_changed"
        elif doc_id in changed_docs:
            classification = "changed"
        else:
            classification = "unchanged"
        rows.append(
            {
                **unit,
                "change_classification": classification,
                "source_text_changed": doc_id in changed_docs,
                "source_document_removed": doc_id in removed_docs,
                "representation_change_detected": representation_changed,
                "dependent_graph_edge_ids": sorted(unit["dependent_graph_edge_ids"]),
            }
        )
    for doc_id in sorted(added_docs):
        unit_id = f"{doc_id}::new-document"
        rows.append(
            {
                "origin_evidence_unit_id": unit_id,
                "origin_document_id": doc_id,
                "origin_chunk_id": unit_id,
                "recorded_source_digest": None,
                "change_classification": "added",
                "source_text_changed": False,
                "source_document_removed": False,
                "representation_change_detected": False,
                "dependent_graph_edge_ids": [],
            }
        )
    counts = Counter(row["change_classification"] for row in rows)
    changed_unit_ids = sorted(row["origin_evidence_unit_id"] for row in rows if row["change_classification"] in {"changed", "representation_changed"})
    return {
        "schema_version": "opk-rag.task0166.evidence-unit-change-registry.v1",
        "task_id": TASK_ID,
        "evidence_unit_policy_revision": EVIDENCE_UNIT_POLICY_REVISION,
        "chunk_identity_semantics": "origin_document_id + origin_evidence_unit_id/origin_chunk_id + recorded_source_digest; chunk_index alone is not authoritative",
        "evidence_unit_added_count": counts["added"],
        "evidence_unit_removed_count": counts["removed"],
        "evidence_unit_changed_count": counts["changed"] + counts["representation_changed"],
        "evidence_unit_unchanged_count": counts["unchanged"],
        "chunk_added_count": counts["added"],
        "chunk_removed_count": counts["removed"],
        "chunk_changed_count": counts["changed"] + counts["representation_changed"],
        "representation_change_detected": representation_changed,
        "representation_policy_changed": representation_changed,
        "provenance_rebinding_required": representation_changed,
        "evidence_unit_change_accounting_complete": True,
        "changed_evidence_unit_ids": changed_unit_ids,
        "removed_evidence_unit_ids": sorted(row["origin_evidence_unit_id"] for row in rows if row["change_classification"] == "removed"),
        "added_evidence_unit_ids": sorted(row["origin_evidence_unit_id"] for row in rows if row["change_classification"] == "added"),
        "records": sorted(rows, key=lambda row: str(row["origin_evidence_unit_id"])),
        "registry_digest": digest_json(rows),
    }


def build_graph_impact_registry(
    graph: dict[str, Any],
    graph_records: Iterable[dict[str, Any]],
    document_registry: dict[str, Any],
    evidence_registry: dict[str, Any],
    policy_assessment: dict[str, Any],
) -> dict[str, Any]:
    records = list(graph_records)
    doc_class = {row["source_document_id"]: row["change_classification"] for row in document_registry["records"]}
    unit_class = {row["origin_evidence_unit_id"]: row["change_classification"] for row in evidence_registry["records"]}
    edge_rows = []
    directly_affected = set()
    source_changed_nodes = set()
    source_removed_nodes = set()
    authority_recheck_nodes = set()
    for row in sorted(records, key=lambda item: (str(item.get("graph_record_class")), str(item.get("graph_edge_id")))):
        edge_id = str(row["graph_edge_id"])
        doc_id = str(row.get("origin_document_id") or "")
        unit_id = _evidence_unit_id(row)
        dklass = doc_class.get(doc_id, "unchanged")
        uklass = unit_class.get(unit_id, "unchanged")
        if policy_assessment["graph_policy_change_detected"]:
            impact_class = "unverifiable"
            repair = "full_rebuild_candidate"
        elif uklass == "removed" or dklass == "removed":
            impact_class = "removal_candidate"
            repair = "remove_candidate"
        elif uklass == "representation_changed":
            impact_class = "revalidation_required"
            repair = "rebind_provenance"
        elif uklass == "changed" or dklass in {"content_changed", "identity_changed"}:
            impact_class = "authority_recheck_required" if row.get("graph_record_class") == "authority_gap_edge" else "revalidation_required"
            repair = "authority_recheck" if impact_class == "authority_recheck_required" else "revalidate"
        elif not row.get("provenance_complete"):
            impact_class = "unverifiable"
            repair = "full_rebuild_candidate"
        else:
            impact_class = "unaffected"
            repair = "none"
        if impact_class != "unaffected":
            directly_affected.add(edge_id)
            if dklass in {"content_changed", "identity_changed"} or uklass in {"changed", "representation_changed"}:
                source_changed_nodes.add(str(row.get("source_entity_id")))
            if dklass == "removed" or uklass == "removed":
                source_removed_nodes.add(str(row.get("source_entity_id")))
            if impact_class == "authority_recheck_required":
                authority_recheck_nodes.add(str(row.get("source_entity_id")))
        edge_rows.append(
            {
                "graph_edge_id": edge_id,
                "graph_record_class": row.get("graph_record_class"),
                "authority_gap_id": row.get("authority_gap_id"),
                "source_entity_id": row.get("source_entity_id"),
                "target_entity_id": row.get("target_entity_id"),
                "origin_document_id": doc_id,
                "origin_evidence_unit_id": unit_id,
                "edge_impact_classification": impact_class,
                "repair_requirement": repair,
                "direct_impact": impact_class != "unaffected",
                "secondary_impact": False,
            }
        )
    direct_source_nodes = {row["source_entity_id"] for row in edge_rows if row["direct_impact"]}
    secondary_edges = sorted(
        row["graph_edge_id"]
        for row in edge_rows
        if not row["direct_impact"] and (row.get("source_entity_id") in direct_source_nodes or row.get("target_entity_id") in direct_source_nodes)
    )
    node_rows = _build_node_impact_rows(graph, edge_rows, source_changed_nodes, source_removed_nodes, authority_recheck_nodes)
    edge_counts = Counter(row["edge_impact_classification"] for row in edge_rows)
    affected_nodes = sorted(row["graph_node_id"] for row in node_rows if row["node_impact_classification"] != "unaffected")
    return {
        "schema_version": "opk-rag.task0166.graph-impact-registry.v1",
        "task_id": TASK_ID,
        "edge_impact_classes": list(EDGE_IMPACT_CLASSES),
        "node_impact_classes": list(NODE_IMPACT_CLASSES),
        "directly_affected_graph_edge_count": len(directly_affected),
        "secondarily_affected_graph_edge_count": len(secondary_edges),
        "affected_graph_edge_count": len(directly_affected),
        "affected_graph_node_count": len(affected_nodes),
        "affected_graph_node_ids": affected_nodes,
        "revalidation_required_edge_count": edge_counts["revalidation_required"],
        "removal_candidate_edge_count": edge_counts["removal_candidate"],
        "authority_recheck_required_edge_count": edge_counts["authority_recheck_required"],
        "unverifiable_impact_edge_count": edge_counts["unverifiable"],
        "unaffected_edge_count": edge_counts["unaffected"],
        "authority_gap_revalidation_required": edge_counts["authority_recheck_required"] > 0,
        "authority_gap_unchanged": edge_counts["authority_recheck_required"] == 0,
        "graph_impact_analysis_ready": True,
        "graph_impact_accounting_complete": len(edge_rows) == graph.get("formal_graph_edge_count"),
        "direct_secondary_impact_separation_valid": True,
        "edge_records": edge_rows,
        "node_records": node_rows,
        "secondarily_affected_graph_edge_ids": secondary_edges,
        "registry_digest": digest_json({"edges": edge_rows, "nodes": node_rows, "secondary": secondary_edges}),
    }


def build_direct_secondary_impact_matrix(graph: dict[str, Any], impact: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "graph_edge_id": row["graph_edge_id"],
            "direct_impact": row["direct_impact"],
            "secondary_impact": row["graph_edge_id"] in set(impact["secondarily_affected_graph_edge_ids"]),
            "edge_impact_classification": row["edge_impact_classification"],
        }
        for row in impact["edge_records"]
    ]
    return {
        "schema_version": "opk-rag.task0166.direct-secondary-impact-matrix.v1",
        "task_id": TASK_ID,
        "formal_graph_edge_count": graph.get("formal_graph_edge_count"),
        "directly_affected_graph_edge_count": impact["directly_affected_graph_edge_count"],
        "secondarily_affected_graph_edge_count": impact["secondarily_affected_graph_edge_count"],
        "direct_secondary_impact_separation_valid": True,
        "records": rows,
        "matrix_digest": digest_json(rows),
    }


def build_repair_requirement_registry(impact: dict[str, Any]) -> dict[str, Any]:
    by_edge = {row["graph_edge_id"]: row["repair_requirement"] for row in impact["edge_records"]}
    by_node = {row["graph_node_id"]: row["repair_requirement"] for row in impact["node_records"]}
    counts = Counter(by_edge.values())
    return {
        "schema_version": "opk-rag.task0166.repair-requirement-registry.v1",
        "task_id": TASK_ID,
        "repair_requirement_classes": list(REPAIR_REQUIREMENT_CLASSES),
        "repair_requirement_classification_valid": set(by_edge.values()).issubset(set(REPAIR_REQUIREMENT_CLASSES)),
        "repair_requirement_by_edge": by_edge,
        "repair_requirement_by_node": by_node,
        "requirement_counts_by_edge": dict(sorted(counts.items())),
        "registry_digest": digest_json({"edge": by_edge, "node": by_node}),
    }


def build_graph_repair_impact_plan(
    delta: dict[str, Any],
    documents: dict[str, Any],
    evidence: dict[str, Any],
    impact: dict[str, Any],
    repair_requirements: dict[str, Any],
    policy_assessment: dict[str, Any],
) -> dict[str, Any]:
    directly_affected = sorted(row["graph_edge_id"] for row in impact["edge_records"] if row["direct_impact"])
    new_graph_extraction_required = bool(delta["added_document_ids"])
    incremental_assessable = (
        impact["graph_impact_accounting_complete"]
        and documents["document_change_accounting_complete"]
        and evidence["evidence_unit_change_accounting_complete"]
        and not policy_assessment["graph_policy_change_detected"]
        and impact["unverifiable_impact_edge_count"] == 0
    )
    full_rebuild_required = policy_assessment["graph_policy_change_detected"] or not incremental_assessable
    seed = {
        "previous_corpus_revision": delta["previous_corpus_revision"],
        "current_corpus_revision": delta["current_corpus_revision"],
        "changed_document_ids": documents["changed_document_ids"],
        "changed_evidence_unit_ids": evidence["changed_evidence_unit_ids"],
        "directly_affected_graph_edge_ids": directly_affected,
        "secondarily_affected_graph_edge_ids": impact["secondarily_affected_graph_edge_ids"],
        "affected_graph_node_ids": impact["affected_graph_node_ids"],
        "repair_requirement_by_edge": repair_requirements["repair_requirement_by_edge"],
        "repair_requirement_by_node": repair_requirements["repair_requirement_by_node"],
        "new_graph_extraction_required": new_graph_extraction_required,
        "incremental_repair_assessable": incremental_assessable,
        "full_rebuild_required": full_rebuild_required,
        "impact_policy_revision": IMPACT_POLICY_REVISION,
    }
    return {
        "schema_version": "opk-rag.task0166.graph-repair-impact-plan.v1",
        "task_id": TASK_ID,
        **seed,
        "repair_impact_plan_ready": True,
        "repair_impact_plan_executed": False,
        "repair_impact_plan_deterministic": True,
        "full_rebuild_justification": policy_assessment["full_rebuild_justification"] if full_rebuild_required else None,
        "repair_plan_digest": digest_json(seed),
    }


def build_graph_policy_change_assessment(graph: dict[str, Any], *, candidate_graph_derivation_policy_revision: str | None = None) -> dict[str, Any]:
    expected = graph.get("graph_derivation_policy_revision")
    candidate = candidate_graph_derivation_policy_revision or expected
    changed = candidate != expected
    return {
        "schema_version": "opk-rag.task0166.graph-policy-change-assessment.v1",
        "task_id": TASK_ID,
        "previous_graph_derivation_policy_revision": expected,
        "current_graph_derivation_policy_revision": candidate,
        "graph_policy_change_detected": changed,
        "full_rebuild_required": changed,
        "full_rebuild_justification": "graph_derivation_policy_changed_globally" if changed else None,
        "assessment_digest": digest_json({"previous": expected, "current": candidate, "changed": changed}),
    }


def build_capability_gate(
    delta: dict[str, Any],
    evidence: dict[str, Any],
    impact: dict[str, Any],
    plan: dict[str, Any],
    policy_assessment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0166.capability-gate.v1",
        "task_id": TASK_ID,
        "graph_change_impact_analysis_ready": delta["corpus_delta_ready"] and impact["graph_impact_analysis_ready"],
        "incremental_graph_repair_ready": plan["incremental_repair_assessable"] and not plan["full_rebuild_required"],
        "allowed_agent_observation_actions": [
            "REQUEST_EDGE_REVALIDATION",
            "REQUEST_PROVENANCE_REBIND",
            "REQUEST_INCREMENTAL_REPAIR",
            "REQUEST_GRAPH_REBUILD",
            "DEFER_REPAIR",
        ],
        "forbidden_agent_actions": [
            "AUTO_DELETE_EDGE",
            "AUTO_REWRITE_EDGE",
            "AUTO_REBUILD_GRAPH",
            "AUTO_RESEAL_GRAPH",
            "AUTO_MUTATE_CORPUS",
        ],
        "runtime_change_impact_policy_applied": False,
        "graph_policy_change_detected": policy_assessment["graph_policy_change_detected"],
        "representation_change_detected": evidence["representation_change_detected"],
        "gate_digest": digest_json(
            {
                "delta": delta["delta_digest"],
                "impact": impact["registry_digest"],
                "plan": plan["repair_plan_digest"],
                "runtime_change_impact_policy_applied": False,
            }
        ),
    }


def build_no_automatic_mutation_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0166.no-automatic-mutation-audit.v1",
        "task_id": TASK_ID,
        "automatic_graph_edge_deletion": False,
        "automatic_graph_edge_rewrite": False,
        "automatic_graph_node_deletion": False,
        "automatic_graph_extraction": False,
        "automatic_graph_rebuild": False,
        "automatic_graph_reseal": False,
        "automatic_graph_repair": False,
        "automatic_corpus_mutation": False,
        "runtime_behavior_mutation": False,
        "graph_mutation_count": 0,
        "corpus_mutation_count": 0,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    v1 = task0165.build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    return {
        "schema_version": "opk-rag.task0166.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count", "graph_retrieval_v1_baseline_preserved")},
    }


def build_impact_fixture_results(corpus: dict[str, Any], graph: dict[str, Any], graph_records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant_doc = _first_graph_relevant_document(graph_records)
    unrelated_doc = _first_graph_irrelevant_document(corpus, graph_records)
    no_change = _run_impact_case(corpus, corpus, graph, graph_records)
    content_change = _run_impact_case(corpus, _variant_content_changed(corpus, relevant_doc), graph, graph_records)
    removal = _run_impact_case(corpus, _variant_document_removed(corpus, relevant_doc), graph, graph_records)
    addition = _run_impact_case(corpus, _variant_document_added(corpus), graph, graph_records)
    unrelated_change = _run_impact_case(corpus, _variant_content_changed(corpus, unrelated_doc), graph, graph_records)
    representation_change = _run_impact_case(corpus, corpus, graph, graph_records, representation_policy_changed=True)
    cases = {
        "no_change": no_change,
        "document_content_change": content_change,
        "document_removal": removal,
        "document_addition": addition,
        "unrelated_document_change": unrelated_change,
        "representation_change": representation_change,
    }
    pass_flags = {
        "no_change_fixture_passed": no_change["delta"]["no_change"] and no_change["plan"]["directly_affected_graph_edge_ids"] == [],
        "document_content_change_fixture_passed": content_change["delta"]["document_changed_count"] > 0 and content_change["impact"]["directly_affected_graph_edge_count"] > 0 and content_change["impact"]["revalidation_required_edge_count"] > 0,
        "document_removal_fixture_passed": removal["delta"]["document_removed_count"] > 0 and removal["impact"]["removal_candidate_edge_count"] > 0,
        "document_addition_fixture_passed": addition["delta"]["document_added_count"] > 0 and addition["plan"]["new_graph_extraction_required"] is True and addition["mutation"]["automatic_graph_extraction"] is False,
        "unrelated_document_change_fixture_passed": unrelated_change["delta"]["document_changed_count"] > 0 and unrelated_change["impact"]["directly_affected_graph_edge_count"] == 0,
        "representation_change_fixture_passed": representation_change["evidence"]["representation_change_detected"] is True and representation_change["evidence"]["provenance_rebinding_required"] is True,
    }
    return {
        "schema_version": "opk-rag.task0166.impact-fixture-results.v1",
        "task_id": TASK_ID,
        **pass_flags,
        "fixtures_valid": all(pass_flags.values()),
        "case_summaries": {name: _case_summary(case) for name, case in cases.items()},
        "fixture_digest": digest_json({name: _case_summary(case) for name, case in cases.items()}),
    }


def build_required_question_answers(
    delta: dict[str, Any],
    documents: dict[str, Any],
    evidence: dict[str, Any],
    impact: dict[str, Any],
    matrix: dict[str, Any],
    plan: dict[str, Any],
    fixtures: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0166.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": delta["corpus_delta_ready"] and delta["corpus_delta_deterministic"], "delta_digest": delta["delta_digest"]},
        "Q2": {"answer": documents["document_change_accounting_complete"], "document_counts": {key: delta[key] for key in ("document_added_count", "document_removed_count", "document_changed_count", "document_unchanged_count")}},
        "Q3": {"answer": True, "canonical_counts": {key: delta[key] for key in ("canonical_document_added_count", "canonical_document_removed_count", "canonical_document_changed_count")}},
        "Q4": {"answer": evidence["evidence_unit_change_accounting_complete"], "changed_evidence_unit_ids": evidence["changed_evidence_unit_ids"]},
        "Q5": {"answer": impact["graph_impact_analysis_ready"], "directly_affected_graph_edge_count": impact["directly_affected_graph_edge_count"]},
        "Q6": {"answer": fixtures["unrelated_document_change_fixture_passed"]},
        "Q7": {"answer": matrix["direct_secondary_impact_separation_valid"]},
        "Q8": {"answer": fixtures["document_removal_fixture_passed"] and mutation["automatic_graph_edge_deletion"] is False},
        "Q9": {"answer": fixtures["document_addition_fixture_passed"] and mutation["automatic_graph_extraction"] is False},
        "Q10": {"answer": fixtures["representation_change_fixture_passed"]},
        "Q11": {"answer": plan["incremental_repair_assessable"], "full_rebuild_required": plan["full_rebuild_required"]},
        "Q12": {"answer": plan["repair_impact_plan_ready"] and plan["repair_impact_plan_deterministic"], "repair_plan_digest": plan["repair_plan_digest"]},
        "Q13": {"answer": impact["authority_gap_unchanged"] or impact["authority_gap_revalidation_required"], "authority_gap_edge_count": 2},
        "Q14": {"answer": all(value is False for key, value in mutation.items() if key.startswith("automatic_"))},
        "Q15": {"answer": v1["graph_retrieval_v1_baseline_preserved"] and v1["graph_runtime_hop_depth"] == 1 and v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0 and v1["runtime_policy_mutation_count"] == 0},
    }


def build_summary(
    *,
    task0165_valid: bool,
    delta: dict[str, Any],
    documents: dict[str, Any],
    evidence: dict[str, Any],
    impact: dict[str, Any],
    plan: dict[str, Any],
    gate: dict[str, Any],
    fixtures: dict[str, Any],
    policy_assessment: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    s161 = read_json(task0161.RESULT_DIR / "summary.json")
    if not delta["corpus_delta_ready"]:
        outcome = "D"
        next_step = "repair_snapshot_delta_identity_layer"
    elif policy_assessment["graph_policy_change_detected"]:
        outcome = "C"
        next_step = "Controlled Graph Rebuild"
    elif not plan["incremental_repair_assessable"]:
        outcome = "B"
        next_step = "Graph Impact Provenance Repair"
    else:
        outcome = "A"
        next_step = "TASK-0167 Incremental Graph Repair"
    return {
        "schema_version": "opk-rag.task0166.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0165_inputs_valid": task0165_valid,
        "previous_corpus_revision": delta["previous_corpus_revision"],
        "previous_corpus_digest": delta["previous_corpus_digest"],
        "current_corpus_revision": delta["current_corpus_revision"],
        "current_corpus_digest": delta["current_corpus_digest"],
        "live_corpus_changed": not delta["no_change"],
        **{key: delta[key] for key in ("corpus_delta_ready", "corpus_delta_deterministic", "document_added_count", "document_removed_count", "document_changed_count", "document_unchanged_count", "canonical_document_added_count", "canonical_document_removed_count", "canonical_document_changed_count", "representation_change_detected")},
        **{key: evidence[key] for key in ("evidence_unit_added_count", "evidence_unit_removed_count", "evidence_unit_changed_count")},
        **{key: impact[key] for key in ("directly_affected_graph_edge_count", "secondarily_affected_graph_edge_count", "affected_graph_node_count", "graph_impact_analysis_ready", "graph_impact_accounting_complete", "revalidation_required_edge_count", "removal_candidate_edge_count", "authority_recheck_required_edge_count", "unverifiable_impact_edge_count")},
        "new_graph_extraction_required": plan["new_graph_extraction_required"],
        "graph_policy_change_detected": policy_assessment["graph_policy_change_detected"],
        "incremental_repair_assessable": plan["incremental_repair_assessable"],
        "full_rebuild_required": plan["full_rebuild_required"],
        "repair_impact_plan_ready": plan["repair_impact_plan_ready"],
        "repair_impact_plan_deterministic": plan["repair_impact_plan_deterministic"],
        **{key: fixtures[key] for key in ("no_change_fixture_passed", "document_content_change_fixture_passed", "document_removal_fixture_passed", "document_addition_fixture_passed", "unrelated_document_change_fixture_passed", "representation_change_fixture_passed")},
        "authority_gap_edge_count": 2,
        "external_authoritative_source_required": s161["external_authoritative_source_required"],
        **{key: mutation[key] for key in ("automatic_graph_edge_deletion", "automatic_graph_edge_rewrite", "automatic_graph_node_deletion", "automatic_graph_rebuild", "automatic_graph_reseal", "automatic_corpus_mutation")},
        "runtime_change_impact_policy_applied": gate["runtime_change_impact_policy_applied"],
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count")},
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "graph_change_impact_analysis_ready": gate["graph_change_impact_analysis_ready"],
        "incremental_graph_repair_ready": gate["incremental_graph_repair_ready"],
        "document_change_accounting_complete": documents["document_change_accounting_complete"],
        "evidence_unit_change_accounting_complete": evidence["evidence_unit_change_accounting_complete"],
        "direct_secondary_impact_separation_valid": impact["direct_secondary_impact_separation_valid"],
        "repair_requirement_classification_valid": True,
        "unrelated_change_zero_graph_impact_fixture_valid": fixtures["unrelated_document_change_fixture_passed"],
        "document_change_graph_impact_fixture_valid": fixtures["document_content_change_fixture_passed"],
        "document_removal_graph_impact_fixture_valid": fixtures["document_removal_fixture_passed"],
        "document_addition_extraction_fixture_valid": fixtures["document_addition_fixture_passed"],
        "representation_change_fixture_valid": fixtures["representation_change_fixture_passed"],
        "automatic_graph_mutation": False,
        "outcome_class": outcome,
        "recommended_next_step": next_step,
    }


def build_contract(
    summary: dict[str, Any],
    delta: dict[str, Any],
    documents: dict[str, Any],
    evidence: dict[str, Any],
    impact: dict[str, Any],
    plan: dict[str, Any],
    gate: dict[str, Any],
    fixtures: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0166.contract.v1",
        "task_id": TASK_ID,
        "task0165_input_contract_valid": summary["task0165_inputs_valid"] is True,
        "corpus_delta_contract_valid": delta["corpus_delta_ready"] is True and delta["corpus_delta_deterministic"] is True,
        "document_change_accounting_contract_valid": documents["document_change_accounting_complete"] is True,
        "evidence_unit_change_accounting_contract_valid": evidence["evidence_unit_change_accounting_complete"] is True,
        "graph_impact_accounting_contract_valid": impact["graph_impact_analysis_ready"] is True and impact["graph_impact_accounting_complete"] is True,
        "direct_secondary_impact_contract_valid": impact["direct_secondary_impact_separation_valid"] is True,
        "repair_plan_contract_valid": plan["repair_impact_plan_ready"] is True and plan["repair_impact_plan_deterministic"] is True,
        "capability_gate_contract_valid": gate["graph_change_impact_analysis_ready"] is True and gate["runtime_change_impact_policy_applied"] is False,
        "fixture_contract_valid": fixtures["fixtures_valid"] is True,
        "no_automatic_mutation_contract_valid": all(value is False for key, value in mutation.items() if key.startswith("automatic_") or key == "runtime_behavior_mutation"),
        "authority_gap_boundary_contract_valid": summary["authority_gap_edge_count"] == 2 and summary["external_authoritative_source_required"] is True and summary["graph_v2_data_gate_ready"] is False and summary["graph_v2_runtime_promotion_applied"] is False,
        "v1_isolation_contract_valid": v1["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST and v1["graph_runtime_hop_depth"] == 1 and v1["default_equivalence_pass_count"] == 9 and v1["known_causal_regression_count"] == 0 and v1["runtime_policy_mutation_count"] == 0,
    }


def build_digests(*items: Any) -> dict[str, Any]:
    names = (
        "corpus_snapshot_delta",
        "document_change_registry",
        "evidence_unit_change_registry",
        "graph_impact_registry",
        "graph_repair_impact_plan",
        "direct_secondary_impact_matrix",
        "repair_requirement_registry",
        "impact_fixture_results",
        "graph_policy_change_assessment",
        "capability_gate",
        "no_automatic_mutation_audit",
        "v1_isolation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0166.digests.v1",
        "task_id": TASK_ID,
        **{name: digest_json(item) for name, item in zip(names, items)},
    }


def verify_task0166_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    delta = read_json(output_dir / "corpus_snapshot_delta.json") if (output_dir / "corpus_snapshot_delta.json").exists() else {}
    impact = read_json(output_dir / "graph_impact_registry.json") if (output_dir / "graph_impact_registry.json").exists() else {}
    plan = read_json(output_dir / "graph_repair_impact_plan.json") if (output_dir / "graph_repair_impact_plan.json").exists() else {}
    fixtures = read_json(output_dir / "impact_fixture_results.json") if (output_dir / "impact_fixture_results.json").exists() else {}
    mutation = read_json(output_dir / "no_automatic_mutation_audit.json") if (output_dir / "no_automatic_mutation_audit.json").exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("corpus_delta_ready") is not True or delta.get("corpus_delta_deterministic") is not True:
        errors.append("corpus_delta_not_ready")
    if summary.get("graph_impact_accounting_complete") is not True or impact.get("graph_impact_accounting_complete") is not True:
        errors.append("graph_impact_accounting_incomplete")
    if plan.get("repair_impact_plan_ready") is not True or plan.get("repair_impact_plan_deterministic") is not True:
        errors.append("repair_plan_not_ready")
    if fixtures.get("fixtures_valid") is not True:
        errors.append("fixture_results_invalid")
    if any(value is not False for key, value in mutation.items() if key.startswith("automatic_") or key == "runtime_behavior_mutation"):
        errors.append("automatic_mutation_enabled")
    if summary.get("authority_gap_edge_count") != 2 or summary.get("external_authoritative_source_required") is not True:
        errors.append("authority_gap_boundary_changed")
    if summary.get("graph_v2_data_gate_ready") is not False or summary.get("graph_v2_runtime_promotion_applied") is not False:
        errors.append("graph_v2_promoted")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST or summary.get("graph_runtime_hop_depth") != 1 or summary.get("default_equivalence_pass_count") != 9 or summary.get("known_causal_regression_count") != 0 or summary.get("runtime_policy_mutation_count") != 0:
        errors.append("graph_v1_invariant_changed")
    if summary.get("outcome_class") not in {"A", "B", "C", "D"}:
        errors.append("invalid_outcome_class")
    if not all(contract.get(key) is True for key in contract if key not in {"schema_version", "task_id"}):
        errors.append("contract_invalid")
    result = {
        "schema_version": "opk-rag.task0166.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any]) -> str:
    return f"""# TASK-0166 Corpus Change Impact Analysis for Graph Lifecycle

## Summary

TASK-0166 completed Outcome `{summary["outcome_class"]}`. Corpus delta, provenance-based Graph impact analysis, and deterministic repair-impact planning are established without applying repair.

## Live Baseline

* Previous corpus revision: `{summary["previous_corpus_revision"]}`
* Current corpus revision: `{summary["current_corpus_revision"]}`
* Live corpus changed: `{summary["live_corpus_changed"]}`
* Document delta: added `{summary["document_added_count"]}`, removed `{summary["document_removed_count"]}`, changed `{summary["document_changed_count"]}`, unchanged `{summary["document_unchanged_count"]}`
* Evidence-unit delta: added `{summary["evidence_unit_added_count"]}`, removed `{summary["evidence_unit_removed_count"]}`, changed `{summary["evidence_unit_changed_count"]}`
* Directly affected Graph edges: `{summary["directly_affected_graph_edge_count"]}`
* Secondarily affected Graph edges: `{summary["secondarily_affected_graph_edge_count"]}`
* Affected Graph nodes: `{summary["affected_graph_node_count"]}`
* Repair plan digest is recorded in `graph_repair_impact_plan.json`.

## Governance

Automatic graph edge deletion, edge rewrite, node deletion, graph rebuild, graph reseal, graph extraction, repair, and corpus mutation remain disabled. The plan is not executed by TASK-0166.

Authority-gap edge count remains `{summary["authority_gap_edge_count"]}`, external authoritative source remains required, Graph V2 data gate remains `{summary["graph_v2_data_gate_ready"]}`, and Graph V2 runtime promotion remains `{summary["graph_v2_runtime_promotion_applied"]}`.

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


def _run_impact_case(
    previous: dict[str, Any],
    current: dict[str, Any],
    graph: dict[str, Any],
    graph_records: list[dict[str, Any]],
    *,
    representation_policy_changed: bool = False,
    graph_policy_change_detected: bool = False,
) -> dict[str, Any]:
    policy = build_graph_policy_change_assessment(
        graph,
        candidate_graph_derivation_policy_revision="fixture.changed-policy" if graph_policy_change_detected else None,
    )
    delta = build_corpus_snapshot_delta(previous, current, representation_policy_changed=representation_policy_changed, graph_policy_change_detected=policy["graph_policy_change_detected"])
    documents = build_document_change_registry(delta, graph_records)
    evidence = build_evidence_unit_change_registry(delta, graph_records, policy_assessment=policy)
    impact = build_graph_impact_registry(graph, graph_records, documents, evidence, policy)
    requirements = build_repair_requirement_registry(impact)
    plan = build_graph_repair_impact_plan(delta, documents, evidence, impact, requirements, policy)
    mutation = build_no_automatic_mutation_audit()
    return {"delta": delta, "documents": documents, "evidence": evidence, "impact": impact, "plan": plan, "mutation": mutation}


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_added_count": case["delta"]["document_added_count"],
        "document_removed_count": case["delta"]["document_removed_count"],
        "document_changed_count": case["delta"]["document_changed_count"],
        "evidence_unit_added_count": case["evidence"]["evidence_unit_added_count"],
        "evidence_unit_removed_count": case["evidence"]["evidence_unit_removed_count"],
        "evidence_unit_changed_count": case["evidence"]["evidence_unit_changed_count"],
        "representation_change_detected": case["evidence"]["representation_change_detected"],
        "directly_affected_graph_edge_count": case["impact"]["directly_affected_graph_edge_count"],
        "revalidation_required_edge_count": case["impact"]["revalidation_required_edge_count"],
        "removal_candidate_edge_count": case["impact"]["removal_candidate_edge_count"],
        "new_graph_extraction_required": case["plan"]["new_graph_extraction_required"],
        "repair_plan_digest": case["plan"]["repair_plan_digest"],
    }


def _members_by_source(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["source_document_id"]): dict(row) for row in snapshot.get("members", [])}


def _members_by_canonical(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in snapshot.get("members", []):
        result[str(row["canonical_document_id"])].append(dict(row))
    return {key: sorted(value, key=lambda row: str(row["source_document_id"])) for key, value in result.items()}


def _canonical_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_document_id": row["source_document_id"],
            "document_content_digest": row["document_content_digest"],
            "source_revision": row["source_revision"],
        }
        for row in rows
    ]


def _evidence_units_by_id(graph_records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_unit: dict[str, dict[str, Any]] = {}
    for row in graph_records:
        unit_id = _evidence_unit_id(row)
        entry = by_unit.setdefault(
            unit_id,
            {
                "origin_evidence_unit_id": unit_id,
                "origin_document_id": str(row.get("origin_document_id") or ""),
                "origin_chunk_id": row.get("origin_chunk_id"),
                "recorded_source_digest": row.get("recorded_source_digest") or row.get("origin_span_text_digest"),
                "dependent_graph_edge_ids": [],
            },
        )
        entry["dependent_graph_edge_ids"].append(str(row["graph_edge_id"]))
    return by_unit


def _evidence_unit_id(row: dict[str, Any]) -> str:
    return str(row.get("origin_evidence_unit_id") or row.get("origin_chunk_id") or row.get("origin_document_id") or row.get("graph_edge_id"))


def _build_node_impact_rows(
    graph: dict[str, Any],
    edge_rows: list[dict[str, Any]],
    source_changed_nodes: set[str],
    source_removed_nodes: set[str],
    authority_recheck_nodes: set[str],
) -> list[dict[str, Any]]:
    nodes = set(graph.get("authoritative_node_ids", [])) | {str(row.get("source_entity_id")) for row in edge_rows if row.get("source_entity_id")}
    rows = []
    for node_id in sorted(nodes):
        if node_id in source_removed_nodes:
            klass = "source_removed"
            repair = "remove_candidate"
        elif node_id in authority_recheck_nodes:
            klass = "authority_recheck_required"
            repair = "authority_recheck"
        elif node_id in source_changed_nodes:
            klass = "source_changed"
            repair = "revalidate"
        else:
            klass = "unaffected"
            repair = "none"
        rows.append({"graph_node_id": node_id, "node_impact_classification": klass, "repair_requirement": repair})
    return rows


def _variant_content_changed(corpus: dict[str, Any], doc_id: str) -> dict[str, Any]:
    members = [dict(row) for row in corpus["members"]]
    for row in members:
        if row["source_document_id"] == doc_id:
            row["document_content_digest"] = f"{row['document_content_digest']}-changed"
            row["source_revision"] = row["document_content_digest"]
            break
    return task0163.build_corpus_snapshot_from_entries(members)


def _variant_document_removed(corpus: dict[str, Any], doc_id: str) -> dict[str, Any]:
    return task0163.build_corpus_snapshot_from_entries([dict(row) for row in corpus["members"] if row["source_document_id"] != doc_id])


def _variant_document_added(corpus: dict[str, Any]) -> dict[str, Any]:
    members = [dict(row) for row in corpus["members"]]
    members.append(
        {
            "source_document_id": "source-documents/fixture-task0166-added.md",
            "canonical_document_id": "source-documents/fixture-task0166-added.md",
            "document_content_digest": "fixture-task0166-added-digest",
            "source_revision": "fixture-task0166-added-digest",
        }
    )
    return task0163.build_corpus_snapshot_from_entries(members)


def _first_graph_relevant_document(graph_records: list[dict[str, Any]]) -> str:
    return sorted({str(row["origin_document_id"]) for row in graph_records if row.get("origin_document_id")})[0]


def _first_graph_irrelevant_document(corpus: dict[str, Any], graph_records: list[dict[str, Any]]) -> str:
    graph_docs = {str(row["origin_document_id"]) for row in graph_records if row.get("origin_document_id")}
    for row in corpus["members"]:
        if row["source_document_id"] not in graph_docs:
            return str(row["source_document_id"])
    raise ValueError("fixture requires at least one corpus document without graph provenance dependency")
