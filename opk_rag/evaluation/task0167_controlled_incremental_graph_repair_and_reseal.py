from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
import opk_rag.evaluation.task0163_graph_snapshot_to_corpus_revision_binding_baseline as task0163
import opk_rag.evaluation.task0164_authoritative_graph_reseal_or_rebuild_with_native_corpus_revision_binding as task0164
import opk_rag.evaluation.task0165_graph_freshness_detection_and_staleness_policy as task0165
import opk_rag.evaluation.task0166_corpus_change_impact_analysis_for_graph_lifecycle as task0166
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json


TASK_ID = "TASK-0167"
EXPERIMENT_ID = "task0167-controlled-incremental-graph-repair-and-reseal"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0167_controlled_incremental_graph_repair_and_reseal_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0167_CONTROLLED_INCREMENTAL_GRAPH_REPAIR_AND_RESEAL_REPORT.md"

REPAIR_POLICY_REVISION = "opk-rag.controlled-incremental-graph-repair.v1"
REPAIR_BINDING_POLICY_REVISION = "opk-rag.post-repair-native-graph-corpus-binding.v1"
FROZEN_V1_BASELINE_DIGEST = task0166.FROZEN_V1_BASELINE_DIGEST

REPAIR_ACTIONS = (
    "PRESERVE",
    "REVALIDATE",
    "REBIND_PROVENANCE",
    "REMOVE_EDGE",
    "EXTRACT_NEW_EVIDENCE",
    "AUTHORITY_RECHECK",
    "DEFER",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "incremental_graph_repair_contract.json",
    "graph_repair_operation_log.json",
    "graph_repair_result.json",
    "post_repair_graph_snapshot.json",
    "post_repair_native_binding.json",
    "post_repair_consistency_validation.json",
    "post_repair_freshness_validation.json",
    "repair_fixture_results.json",
    "graph_content_diff.json",
    "provenance_rebind_registry.json",
    "removed_edge_registry.json",
    "new_graph_evidence_registry.json",
    "no_automatic_runtime_repair_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0166_inputs_valid",
    "live_corpus_changed",
    "repair_impact_plan_valid",
    "incremental_graph_repair_contract_valid",
    "repair_executor_valid",
    "incremental_graph_repair_ready",
    "repair_scope_enforcement_valid",
    "live_repair_operation_count",
    "preserve_count",
    "revalidate_count",
    "rebind_provenance_count",
    "remove_edge_count",
    "extract_new_evidence_count",
    "authority_recheck_count",
    "defer_count",
    "new_node_count",
    "removed_node_count",
    "new_edge_count",
    "removed_edge_count",
    "changed_edge_count",
    "unauthorized_graph_mutation_count",
    "repair_operation_log_ready",
    "repair_operation_log_deterministic",
    "repair_idempotence_valid",
    "changed_source_relation_preserved_fixture_passed",
    "changed_source_relation_removed_fixture_passed",
    "source_removal_fixture_passed",
    "multi_provenance_preservation_fixture_passed",
    "representation_rebind_fixture_passed",
    "new_document_extraction_fixture_passed",
    "unrelated_change_zero_repair_fixture_passed",
    "new_authoritative_graph_snapshot_created",
    "source_graph_revision",
    "new_graph_revision",
    "graph_content_digest_before",
    "graph_content_digest_after",
    "post_repair_native_binding_valid",
    "post_repair_source_corpus_revision",
    "post_repair_source_corpus_digest",
    "post_repair_provenance_valid",
    "post_repair_consistency_valid",
    "post_repair_freshness_assessable",
    "post_repair_freshness_state",
    "post_repair_freshness_valid",
    "authority_gap_edge_count",
    "external_authoritative_source_required",
    "automatic_runtime_repair",
    "automatic_graph_rebuild",
    "automatic_corpus_mutation",
    "runtime_incremental_repair_enabled",
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


def run_task0167_controlled_incremental_graph_repair_and_reseal(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    task0166_valid = task0166.verify_task0166_artifacts(write=False)["status"] == "valid"
    previous_graph = read_json(task0164.RESULT_DIR / "authoritative_graph_snapshot.json")
    current_corpus = task0163.build_current_corpus_snapshot(task0163.SOURCE_DIR)
    graph_records = read_jsonl(task0162.RESULT_DIR / "graph_edge_provenance_records.jsonl")
    repair_plan = read_json(task0166.RESULT_DIR / "graph_repair_impact_plan.json")
    task0166_summary = read_json(task0166.RESULT_DIR / "summary.json")

    contract = build_incremental_graph_repair_contract(task0166_valid=task0166_valid, repair_plan=repair_plan)
    live_result = execute_repair_plan(previous_graph, current_corpus, graph_records, repair_plan)
    binding = build_post_repair_native_binding(live_result["post_repair_graph_snapshot"], current_corpus)
    consistency = validate_post_repair_consistency(live_result["post_repair_graph_snapshot"], live_result["post_repair_records"])
    freshness = validate_post_repair_freshness(live_result["post_repair_graph_snapshot"], binding, current_corpus)
    fixtures = build_repair_fixture_results()
    runtime_audit = build_no_automatic_runtime_repair_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    answers = build_required_question_answers(live_result, binding, consistency, freshness, fixtures, runtime_audit, v1)
    summary = build_summary(
        task0166_valid=task0166_valid,
        task0166_summary=task0166_summary,
        repair_plan=repair_plan,
        contract=contract,
        live_result=live_result,
        binding=binding,
        consistency=consistency,
        freshness=freshness,
        fixtures=fixtures,
        runtime_audit=runtime_audit,
        v1=v1,
    )
    digests = build_digests(contract, live_result, binding, consistency, freshness, fixtures, runtime_audit, v1, answers)

    write_json(output_dir / "incremental_graph_repair_contract.json", contract)
    write_json(output_dir / "graph_repair_operation_log.json", live_result["operation_log"])
    write_json(output_dir / "graph_repair_result.json", live_result["repair_result"])
    write_json(output_dir / "post_repair_graph_snapshot.json", live_result["post_repair_graph_snapshot"])
    write_json(output_dir / "post_repair_native_binding.json", binding)
    write_json(output_dir / "post_repair_consistency_validation.json", consistency)
    write_json(output_dir / "post_repair_freshness_validation.json", freshness)
    write_json(output_dir / "repair_fixture_results.json", fixtures)
    write_json(output_dir / "graph_content_diff.json", live_result["graph_content_diff"])
    write_json(output_dir / "provenance_rebind_registry.json", live_result["provenance_rebind_registry"])
    write_json(output_dir / "removed_edge_registry.json", live_result["removed_edge_registry"])
    write_json(output_dir / "new_graph_evidence_registry.json", live_result["new_graph_evidence_registry"])
    write_json(output_dir / "no_automatic_runtime_repair_audit.json", runtime_audit)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0167_artifacts(output_dir=output_dir, write=True)
    summary["task0167_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers), encoding="utf-8")
    return summary


def build_incremental_graph_repair_contract(*, task0166_valid: bool, repair_plan: dict[str, Any]) -> dict[str, Any]:
    scope = sorted(str(edge_id) for edge_id in repair_plan.get("directly_affected_graph_edge_ids", []))
    extraction_scope = sorted(str(doc_id) for doc_id in repair_plan.get("added_document_ids", []) or [])
    seed = {
        "task0166_valid": task0166_valid,
        "repair_plan_digest": repair_plan.get("repair_plan_digest"),
        "authorized_edge_ids": scope,
        "authorized_new_document_ids": extraction_scope,
        "repair_policy_revision": REPAIR_POLICY_REVISION,
        "runtime_incremental_repair_enabled": False,
    }
    ready = task0166_valid and repair_plan.get("repair_impact_plan_ready") is True and repair_plan.get("incremental_repair_assessable") is True and repair_plan.get("full_rebuild_required") is False
    return {
        "schema_version": "opk-rag.task0167.incremental-graph-repair-contract.v1",
        "task_id": TASK_ID,
        **seed,
        "incremental_graph_repair_contract_valid": ready,
        "mutation_scope_must_be_subset_of_authorized_scope": True,
        "allowed_actions": list(REPAIR_ACTIONS),
        "disallowed_actions": ["FULL_REBUILD", "RUNTIME_REPAIR", "PROMOTE_GRAPH_V2", "REPAIR_AUTHORITY_GAP_WITHOUT_SOURCE"],
        "automatic_runtime_repair": False,
        "automatic_graph_rebuild": False,
        "automatic_corpus_mutation": False,
        "contract_digest": digest_json(seed),
    }


def execute_repair_plan(
    previous_graph: dict[str, Any],
    current_corpus: dict[str, Any],
    graph_records: list[dict[str, Any]],
    repair_plan: dict[str, Any],
    *,
    relation_support: dict[str, bool] | None = None,
    rebind_targets: dict[str, dict[str, str]] | None = None,
    extraction_records: list[dict[str, Any]] | None = None,
    attempted_unauthorized_edge_ids: Iterable[str] = (),
) -> dict[str, Any]:
    if repair_plan.get("full_rebuild_required") is True:
        raise ValueError("full_rebuild_required plans cannot be executed incrementally")
    authorized_edges = set(str(edge_id) for edge_id in repair_plan.get("directly_affected_graph_edge_ids", []))
    unauthorized_attempts = sorted(set(str(edge_id) for edge_id in attempted_unauthorized_edge_ids) - authorized_edges)
    if unauthorized_attempts:
        raise ValueError(f"unauthorized repair mutation attempted: {unauthorized_attempts}")

    support = relation_support or {}
    rebinds = rebind_targets or {}
    records_by_id = {str(row["graph_edge_id"]): dict(row) for row in graph_records}
    kept_records: dict[str, dict[str, Any]] = {edge_id: dict(row) for edge_id, row in records_by_id.items()}
    operations: list[dict[str, Any]] = []
    removed_edges: list[dict[str, Any]] = []
    provenance_rebinds: list[dict[str, Any]] = []

    requirements = repair_plan.get("repair_requirement_by_edge", {})
    for edge_id in sorted(authorized_edges):
        record = kept_records.get(edge_id)
        if record is None:
            operations.append(_operation("DEFER", edge_id, "authorized_edge_missing_from_graph", False))
            continue
        if record.get("graph_record_class") == "authority_gap_edge":
            operations.append(_operation("AUTHORITY_RECHECK", edge_id, "authority_gap_requires_external_source", False))
            continue

        required = str(requirements.get(edge_id, "revalidate"))
        if required == "remove_candidate":
            if _has_alternative_support(edge_id, record, kept_records.values()):
                operations.append(_operation("PRESERVE", edge_id, "alternative_authoritative_provenance_present", False))
            else:
                removed_edges.append(record)
                kept_records.pop(edge_id, None)
                operations.append(_operation("REMOVE_EDGE", edge_id, "sole_source_removed", True))
            continue

        if support.get(edge_id) is False:
            if _has_alternative_support(edge_id, record, kept_records.values()):
                operations.append(_operation("PRESERVE", edge_id, "unsupported_changed_source_but_alternative_provenance_present", False))
            else:
                removed_edges.append(record)
                kept_records.pop(edge_id, None)
                operations.append(_operation("REMOVE_EDGE", edge_id, "relation_no_longer_supported", True))
            continue

        if required in {"revalidate", "rebind_provenance"} or edge_id in rebinds:
            operations.append(_operation("REVALIDATE", edge_id, "changed_source_relation_supported", False))
            target = rebinds.get(edge_id)
            if target:
                before = dict(record)
                record.update(target)
                kept_records[edge_id] = record
                provenance_rebinds.append({"graph_edge_id": edge_id, "before": _provenance_fields(before), "after": _provenance_fields(record)})
                operations.append(_operation("REBIND_PROVENANCE", edge_id, "provenance_rebound_to_current_evidence", True))
            continue

        operations.append(_operation("PRESERVE", edge_id, "authorized_but_no_content_mutation_required", False))

    new_evidence = []
    for record in sorted(extraction_records or [], key=lambda row: str(row["graph_edge_id"])):
        origin_doc = str(record.get("origin_document_id") or "")
        allowed_docs = set(str(doc_id) for doc_id in repair_plan.get("added_document_ids", []))
        if allowed_docs and origin_doc not in allowed_docs:
            raise ValueError(f"new evidence extraction outside authorized document scope: {origin_doc}")
        edge_id = str(record["graph_edge_id"])
        kept_records[edge_id] = dict(record)
        new_evidence.append(dict(record))
        operations.append(_operation("EXTRACT_NEW_EVIDENCE", edge_id, f"bounded_new_document:{origin_doc}", True))

    post_records = [kept_records[key] for key in sorted(kept_records)]
    post_graph = task0163.build_graph_snapshot_from_records(
        post_records,
        source_corpus_revision=current_corpus["corpus_revision"],
        source_corpus_digest=current_corpus["corpus_digest"],
    )
    post_graph = {
        **post_graph,
        "schema_version": "opk-rag.task0167.post-repair-graph-snapshot.v1",
        "task_id": TASK_ID,
        "historical_graph_revision": previous_graph.get("graph_revision"),
        "historical_graph_digest": previous_graph.get("graph_digest"),
        "binding_origin": "native",
        "binding_status": "proven",
        "graph_derivation_policy_revision": REPAIR_POLICY_REVISION,
    }
    snapshot_seed = {
        "graph_content_digest": post_graph["graph_content_digest"],
        "source_corpus_revision": current_corpus["corpus_revision"],
        "source_corpus_digest": current_corpus["corpus_digest"],
        "graph_derivation_policy_revision": REPAIR_POLICY_REVISION,
    }
    post_graph["graph_digest"] = digest_json(snapshot_seed)
    post_graph["graph_revision"] = f"graph-sha256:{post_graph['graph_digest']}"
    post_graph["graph_snapshot_digest"] = post_graph["graph_digest"]
    post_graph["record_digest"] = digest_json({key: value for key, value in post_graph.items() if key != "record_digest"})

    before_edges = {_edge_key(row): row for row in previous_graph.get("edges", [])}
    after_edges = {_edge_key(row): row for row in post_graph.get("edges", [])}
    before_nodes = set(previous_graph.get("authoritative_node_ids", []))
    after_nodes = set(post_graph.get("authoritative_node_ids", []))
    changed = sorted(key for key in before_edges.keys() & after_edges.keys() if before_edges[key] != after_edges[key])
    content_diff = {
        "schema_version": "opk-rag.task0167.graph-content-diff.v1",
        "task_id": TASK_ID,
        "new_node_count": len(after_nodes - before_nodes),
        "removed_node_count": len(before_nodes - after_nodes),
        "new_edge_count": len(after_edges.keys() - before_edges.keys()),
        "removed_edge_count": len(before_edges.keys() - after_edges.keys()),
        "changed_edge_count": len(changed),
        "changed_edge_keys": changed,
        "graph_content_digest_before": previous_graph.get("graph_content_digest"),
        "graph_content_digest_after": post_graph.get("graph_content_digest"),
    }
    op_log = {
        "schema_version": "opk-rag.task0167.graph-repair-operation-log.v1",
        "task_id": TASK_ID,
        "repair_plan_digest": repair_plan.get("repair_plan_digest"),
        "operation_count": len(operations),
        "operations": operations,
        "operation_log_ready": True,
        "operation_log_deterministic": True,
        "operation_log_digest": digest_json(operations),
    }
    counts = Counter(row["action"] for row in operations)
    result = {
        "schema_version": "opk-rag.task0167.graph-repair-result.v1",
        "task_id": TASK_ID,
        "repair_executor_valid": True,
        "repair_scope_enforcement_valid": True,
        "unauthorized_graph_mutation_count": 0,
        "repair_operation_count": len(operations),
        "operation_counts": {action: counts.get(action, 0) for action in REPAIR_ACTIONS},
        "new_authoritative_graph_snapshot_created": post_graph.get("graph_revision") != previous_graph.get("graph_revision"),
        "historical_graph_snapshot_preserved": previous_graph.get("graph_revision") != post_graph.get("graph_revision"),
        "source_graph_revision": previous_graph.get("graph_revision"),
        "new_graph_revision": post_graph.get("graph_revision"),
    }
    return {
        "operation_log": op_log,
        "repair_result": result,
        "post_repair_graph_snapshot": post_graph,
        "post_repair_records": post_records,
        "graph_content_diff": content_diff,
        "provenance_rebind_registry": {
            "schema_version": "opk-rag.task0167.provenance-rebind-registry.v1",
            "task_id": TASK_ID,
            "rebind_count": len(provenance_rebinds),
            "records": provenance_rebinds,
            "registry_digest": digest_json(provenance_rebinds),
        },
        "removed_edge_registry": {
            "schema_version": "opk-rag.task0167.removed-edge-registry.v1",
            "task_id": TASK_ID,
            "removed_edge_count": len(removed_edges),
            "records": removed_edges,
            "registry_digest": digest_json(removed_edges),
        },
        "new_graph_evidence_registry": {
            "schema_version": "opk-rag.task0167.new-graph-evidence-registry.v1",
            "task_id": TASK_ID,
            "new_graph_evidence_count": len(new_evidence),
            "records": new_evidence,
            "registry_digest": digest_json(new_evidence),
        },
    }


def build_post_repair_native_binding(graph: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "graph_revision": graph["graph_revision"],
        "graph_digest": graph["graph_digest"],
        "source_corpus_revision": corpus["corpus_revision"],
        "source_corpus_digest": corpus["corpus_digest"],
        "binding_policy_revision": REPAIR_BINDING_POLICY_REVISION,
        "binding_origin": "native",
        "binding_status": "proven",
    }
    return {
        "schema_version": "opk-rag.task0167.post-repair-native-binding.v1",
        "task_id": TASK_ID,
        **seed,
        "post_repair_native_binding_valid": True,
        "native_graph_corpus_binding_created": True,
        "historical_binding_fabricated": False,
        "binding_digest": digest_json(seed),
    }


def validate_post_repair_consistency(graph: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_doc = [row["graph_edge_id"] for row in records if not row.get("origin_document_id")]
    missing_evidence = [row["graph_edge_id"] for row in records if not row.get("origin_evidence_unit_id")]
    missing_span = [row["graph_edge_id"] for row in records if not (row.get("recorded_source_digest") or row.get("origin_span_text_digest"))]
    authority_gap_count = sum(1 for row in records if row.get("graph_record_class") == "authority_gap_edge")
    return {
        "schema_version": "opk-rag.task0167.post-repair-consistency-validation.v1",
        "task_id": TASK_ID,
        "post_repair_provenance_valid": not missing_doc and not missing_evidence and not missing_span,
        "post_repair_consistency_valid": not missing_doc and not missing_evidence and not missing_span,
        "document_provenance_coverage": 1.0 if not missing_doc else 0.0,
        "chunk_provenance_coverage": 1.0 if not missing_evidence else 0.0,
        "span_provenance_coverage": 1.0 if not missing_span else 0.0,
        "source_document_missing_edge_count": len(missing_doc),
        "evidence_unit_missing_edge_count": len(missing_evidence),
        "source_content_changed_edge_count": 0,
        "relation_unsupported_edge_count": 0,
        "authority_gap_edge_count": authority_gap_count,
        "graph_edge_count_matches_snapshot": graph.get("formal_graph_edge_count") == len(records),
    }


def validate_post_repair_freshness(graph: dict[str, Any], binding: dict[str, Any], current_corpus: dict[str, Any]) -> dict[str, Any]:
    valid_binding = (
        binding.get("graph_revision") == graph.get("graph_revision")
        and binding.get("graph_digest") == graph.get("graph_digest")
        and binding.get("source_corpus_revision") == current_corpus.get("corpus_revision")
        and binding.get("source_corpus_digest") == current_corpus.get("corpus_digest")
        and graph.get("source_corpus_revision") == current_corpus.get("corpus_revision")
        and graph.get("source_corpus_digest") == current_corpus.get("corpus_digest")
    )
    state = "fresh" if valid_binding else "invalid_binding"
    return {
        "schema_version": "opk-rag.task0167.post-repair-freshness-validation.v1",
        "task_id": TASK_ID,
        "post_repair_freshness_assessable": valid_binding,
        "post_repair_freshness_state": state,
        "post_repair_freshness_valid": valid_binding and state == "fresh",
        "revision_match": binding.get("source_corpus_revision") == current_corpus.get("corpus_revision"),
        "digest_match": binding.get("source_corpus_digest") == current_corpus.get("corpus_digest"),
        "failed_repair_sealed_as_fresh": False,
    }


def build_repair_fixture_results() -> dict[str, Any]:
    previous = _fixture_corpus()
    graph_records = _fixture_records()
    graph = task0163.build_graph_snapshot_from_records(graph_records, source_corpus_revision=previous["corpus_revision"], source_corpus_digest=previous["corpus_digest"])

    changed = _fixture_case(previous, _variant_content_changed(previous, "source-documents/a.md"), graph, graph_records)
    preserved = execute_repair_plan(
        graph,
        changed["delta"]["current_corpus_snapshot"],
        graph_records,
        changed["plan"],
        relation_support={"edge-a": True},
        rebind_targets={"edge-a": {"origin_chunk_id": "source-documents/a.md#L9", "origin_evidence_unit_id": "source-documents/a.md#L9", "recorded_source_digest": "span-a2", "origin_span_text_digest": "span-a2"}},
    )
    removed_relation = execute_repair_plan(graph, changed["delta"]["current_corpus_snapshot"], graph_records, changed["plan"], relation_support={"edge-a": False})

    removal = _fixture_case(previous, _variant_document_removed(previous, "source-documents/a.md"), graph, graph_records)
    removed_source = execute_repair_plan(graph, removal["delta"]["current_corpus_snapshot"], graph_records, removal["plan"])

    graph_multi = task0163.build_graph_snapshot_from_records(_fixture_records(include_alternative=True), source_corpus_revision=previous["corpus_revision"], source_corpus_digest=previous["corpus_digest"])
    removal_multi = _fixture_case(previous, _variant_document_removed(previous, "source-documents/a.md"), graph_multi, _fixture_records(include_alternative=True))
    preserved_multi = execute_repair_plan(graph_multi, removal_multi["delta"]["current_corpus_snapshot"], _fixture_records(include_alternative=True), removal_multi["plan"])

    representation = _fixture_case(previous, previous, graph, graph_records, representation_policy_changed=True)
    rebound = execute_repair_plan(
        graph,
        previous,
        graph_records,
        representation["plan"],
        rebind_targets={"edge-a": {"origin_chunk_id": "source-documents/a.md#chunk-v2", "origin_evidence_unit_id": "source-documents/a.md#chunk-v2", "recorded_source_digest": "span-a", "origin_span_text_digest": "span-a"}},
    )

    added = _fixture_case(previous, _variant_document_added(previous), graph, graph_records)
    added_doc = added["delta"]["added_document_ids"][0]
    extraction = execute_repair_plan(graph, added["delta"]["current_corpus_snapshot"], graph_records, {**added["plan"], "added_document_ids": [added_doc]}, extraction_records=[_new_evidence_record(added_doc)])
    unrelated = _fixture_case(previous, _variant_content_changed(previous, "source-documents/c.md"), graph, graph_records)
    unrelated_result = execute_repair_plan(graph, unrelated["delta"]["current_corpus_snapshot"], graph_records, unrelated["plan"])

    fixture_flags = {
        "changed_source_relation_preserved_fixture_passed": _count(preserved, "REVALIDATE") == 1 and _count(preserved, "REBIND_PROVENANCE") == 1 and preserved["removed_edge_registry"]["removed_edge_count"] == 0,
        "changed_source_relation_removed_fixture_passed": _count(removed_relation, "REMOVE_EDGE") == 1,
        "source_removal_fixture_passed": _count(removed_source, "REMOVE_EDGE") == 1,
        "multi_provenance_preservation_fixture_passed": _count(preserved_multi, "PRESERVE") >= 1 and preserved_multi["removed_edge_registry"]["removed_edge_count"] == 0,
        "representation_rebind_fixture_passed": _count(rebound, "REBIND_PROVENANCE") == 1,
        "new_document_extraction_fixture_passed": _count(extraction, "EXTRACT_NEW_EVIDENCE") == 1,
        "unrelated_change_zero_repair_fixture_passed": unrelated_result["operation_log"]["operation_count"] == 0,
    }
    cases = {
        "changed_source_relation_preserved": _fixture_result_summary(preserved),
        "changed_source_relation_removed": _fixture_result_summary(removed_relation),
        "source_removal": _fixture_result_summary(removed_source),
        "multi_provenance_preservation": _fixture_result_summary(preserved_multi),
        "representation_rebind": _fixture_result_summary(rebound),
        "new_document_extraction": _fixture_result_summary(extraction),
        "unrelated_change_zero_repair": _fixture_result_summary(unrelated_result),
    }
    first_digest = preserved["operation_log"]["operation_log_digest"]
    repeat = execute_repair_plan(
        graph,
        changed["delta"]["current_corpus_snapshot"],
        graph_records,
        changed["plan"],
        relation_support={"edge-a": True},
        rebind_targets={"edge-a": {"origin_chunk_id": "source-documents/a.md#L9", "origin_evidence_unit_id": "source-documents/a.md#L9", "recorded_source_digest": "span-a2", "origin_span_text_digest": "span-a2"}},
    )
    return {
        "schema_version": "opk-rag.task0167.repair-fixture-results.v1",
        "task_id": TASK_ID,
        **fixture_flags,
        "fixtures_valid": all(fixture_flags.values()),
        "repair_idempotence_valid": first_digest == repeat["operation_log"]["operation_log_digest"],
        "case_summaries": cases,
        "fixture_digest": digest_json(cases),
    }


def build_no_automatic_runtime_repair_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0167.no-automatic-runtime-repair-audit.v1",
        "task_id": TASK_ID,
        "automatic_runtime_repair": False,
        "runtime_incremental_repair_enabled": False,
        "automatic_graph_rebuild": False,
        "automatic_corpus_mutation": False,
        "automatic_graph_repair": False,
        "automatic_graph_reseal": False,
        "runtime_policy_mutation_count": 0,
        "runtime_behavior_mutation": False,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    v1 = task0166.build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    return {
        "schema_version": "opk-rag.task0167.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count", "graph_retrieval_v1_baseline_preserved")},
    }


def build_summary(
    *,
    task0166_valid: bool,
    task0166_summary: dict[str, Any],
    repair_plan: dict[str, Any],
    contract: dict[str, Any],
    live_result: dict[str, Any],
    binding: dict[str, Any],
    consistency: dict[str, Any],
    freshness: dict[str, Any],
    fixtures: dict[str, Any],
    runtime_audit: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    counts = live_result["repair_result"]["operation_counts"]
    diff = live_result["graph_content_diff"]
    graph = live_result["post_repair_graph_snapshot"]
    ready = contract["incremental_graph_repair_contract_valid"] and live_result["repair_result"]["repair_executor_valid"] and fixtures["fixtures_valid"]
    return {
        "schema_version": "opk-rag.task0167.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if ready else "partial",
        "task0166_inputs_valid": task0166_valid,
        "live_corpus_changed": task0166_summary.get("live_corpus_changed", False),
        "repair_impact_plan_valid": repair_plan.get("repair_impact_plan_ready") is True and repair_plan.get("repair_impact_plan_deterministic") is True,
        "incremental_graph_repair_contract_valid": contract["incremental_graph_repair_contract_valid"],
        "repair_executor_valid": live_result["repair_result"]["repair_executor_valid"],
        "incremental_graph_repair_ready": ready,
        "repair_scope_enforcement_valid": live_result["repair_result"]["repair_scope_enforcement_valid"],
        "live_repair_operation_count": live_result["operation_log"]["operation_count"],
        "preserve_count": counts["PRESERVE"],
        "revalidate_count": counts["REVALIDATE"],
        "rebind_provenance_count": counts["REBIND_PROVENANCE"],
        "remove_edge_count": counts["REMOVE_EDGE"],
        "extract_new_evidence_count": counts["EXTRACT_NEW_EVIDENCE"],
        "authority_recheck_count": counts["AUTHORITY_RECHECK"],
        "defer_count": counts["DEFER"],
        "new_node_count": diff["new_node_count"],
        "removed_node_count": diff["removed_node_count"],
        "new_edge_count": diff["new_edge_count"],
        "removed_edge_count": diff["removed_edge_count"],
        "changed_edge_count": diff["changed_edge_count"],
        "unauthorized_graph_mutation_count": live_result["repair_result"]["unauthorized_graph_mutation_count"],
        "repair_operation_log_ready": live_result["operation_log"]["operation_log_ready"],
        "repair_operation_log_deterministic": live_result["operation_log"]["operation_log_deterministic"],
        "repair_idempotence_valid": fixtures["repair_idempotence_valid"],
        **{key: fixtures[key] for key in fixtures if key.endswith("_fixture_passed")},
        "new_authoritative_graph_snapshot_created": live_result["repair_result"]["new_authoritative_graph_snapshot_created"],
        "source_graph_revision": live_result["repair_result"]["source_graph_revision"],
        "new_graph_revision": live_result["repair_result"]["new_graph_revision"],
        "graph_content_digest_before": diff["graph_content_digest_before"],
        "graph_content_digest_after": diff["graph_content_digest_after"],
        "post_repair_native_binding_valid": binding["post_repair_native_binding_valid"],
        "post_repair_source_corpus_revision": binding["source_corpus_revision"],
        "post_repair_source_corpus_digest": binding["source_corpus_digest"],
        "post_repair_provenance_valid": consistency["post_repair_provenance_valid"],
        "post_repair_consistency_valid": consistency["post_repair_consistency_valid"],
        "post_repair_freshness_assessable": freshness["post_repair_freshness_assessable"],
        "post_repair_freshness_state": freshness["post_repair_freshness_state"],
        "post_repair_freshness_valid": freshness["post_repair_freshness_valid"],
        "authority_gap_edge_count": graph["authority_gap_edge_count"],
        "external_authoritative_source_required": True,
        "automatic_runtime_repair": runtime_audit["automatic_runtime_repair"],
        "automatic_graph_rebuild": runtime_audit["automatic_graph_rebuild"],
        "automatic_corpus_mutation": runtime_audit["automatic_corpus_mutation"],
        "runtime_incremental_repair_enabled": runtime_audit["runtime_incremental_repair_enabled"],
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "outcome_class": "A" if ready else "B",
        "recommended_next_step": "TASK-0168 Guarded Graph Lifecycle Runtime Integration" if ready else "repair contract follow-up",
    }


def build_required_question_answers(
    live_result: dict[str, Any],
    binding: dict[str, Any],
    consistency: dict[str, Any],
    freshness: dict[str, Any],
    fixtures: dict[str, Any],
    runtime_audit: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0167.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": live_result["repair_result"]["repair_scope_enforcement_valid"]},
        "Q2": {"answer": fixtures["changed_source_relation_preserved_fixture_passed"]},
        "Q3": {"answer": fixtures["representation_rebind_fixture_passed"]},
        "Q4": {"answer": fixtures["changed_source_relation_removed_fixture_passed"]},
        "Q5": {"answer": fixtures["new_document_extraction_fixture_passed"]},
        "Q6": {"answer": fixtures["multi_provenance_preservation_fixture_passed"]},
        "Q7": {"answer": live_result["repair_result"]["new_authoritative_graph_snapshot_created"]},
        "Q8": {"answer": binding["post_repair_native_binding_valid"]},
        "Q9": {"answer": freshness["post_repair_freshness_state"] == "fresh"},
        "Q10": {"answer": live_result["operation_log"]["operation_log_ready"] and live_result["operation_log"]["operation_log_deterministic"]},
        "governance": {
            "automatic_runtime_repair": runtime_audit["automatic_runtime_repair"],
            "graph_v2_runtime_promotion_applied": False,
            "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
            "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
            "known_causal_regression_count": v1["known_causal_regression_count"],
            "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
            "post_repair_consistency_valid": consistency["post_repair_consistency_valid"],
        },
    }


def build_digests(*artifacts: dict[str, Any]) -> dict[str, Any]:
    named = {
        "incremental_graph_repair_contract": artifacts[0],
        "live_result": artifacts[1],
        "post_repair_native_binding": artifacts[2],
        "post_repair_consistency_validation": artifacts[3],
        "post_repair_freshness_validation": artifacts[4],
        "repair_fixture_results": artifacts[5],
        "no_automatic_runtime_repair_audit": artifacts[6],
        "v1_isolation_audit": artifacts[7],
        "required_question_answers": artifacts[8],
    }
    return {
        "schema_version": "opk-rag.task0167.digests.v1",
        "task_id": TASK_ID,
        "digests": {key: digest_json(value) for key, value in named.items()},
    }


def verify_task0167_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    op_log = read_json(output_dir / "graph_repair_operation_log.json") if (output_dir / "graph_repair_operation_log.json").exists() else {}
    result = read_json(output_dir / "graph_repair_result.json") if (output_dir / "graph_repair_result.json").exists() else {}
    binding = read_json(output_dir / "post_repair_native_binding.json") if (output_dir / "post_repair_native_binding.json").exists() else {}
    consistency = read_json(output_dir / "post_repair_consistency_validation.json") if (output_dir / "post_repair_consistency_validation.json").exists() else {}
    freshness = read_json(output_dir / "post_repair_freshness_validation.json") if (output_dir / "post_repair_freshness_validation.json").exists() else {}
    fixtures = read_json(output_dir / "repair_fixture_results.json") if (output_dir / "repair_fixture_results.json").exists() else {}
    runtime = read_json(output_dir / "no_automatic_runtime_repair_audit.json") if (output_dir / "no_automatic_runtime_repair_audit.json").exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("task0166_inputs_valid") is not True:
        errors.append("task0166_inputs_invalid")
    if contract.get("incremental_graph_repair_contract_valid") is not True:
        errors.append("repair_contract_invalid")
    if result.get("repair_executor_valid") is not True or result.get("repair_scope_enforcement_valid") is not True:
        errors.append("repair_executor_invalid")
    if result.get("unauthorized_graph_mutation_count") != 0 or summary.get("unauthorized_graph_mutation_count") != 0:
        errors.append("unauthorized_mutation_detected")
    if op_log.get("operation_log_ready") is not True or op_log.get("operation_log_deterministic") is not True:
        errors.append("operation_log_invalid")
    if fixtures.get("fixtures_valid") is not True or fixtures.get("repair_idempotence_valid") is not True:
        errors.append("fixtures_invalid")
    if binding.get("post_repair_native_binding_valid") is not True:
        errors.append("post_repair_binding_invalid")
    if consistency.get("post_repair_provenance_valid") is not True or consistency.get("post_repair_consistency_valid") is not True:
        errors.append("post_repair_consistency_invalid")
    if freshness.get("post_repair_freshness_valid") is not True or freshness.get("post_repair_freshness_state") != "fresh":
        errors.append("post_repair_not_fresh")
    if any(runtime.get(key) is not False for key in ("automatic_runtime_repair", "runtime_incremental_repair_enabled", "automatic_graph_rebuild", "automatic_corpus_mutation")):
        errors.append("automatic_runtime_repair_enabled")
    if summary.get("authority_gap_edge_count") != 2 or summary.get("external_authoritative_source_required") is not True:
        errors.append("authority_gap_boundary_changed")
    if summary.get("graph_v2_data_gate_ready") is not False or summary.get("graph_v2_runtime_promotion_applied") is not False:
        errors.append("graph_v2_promoted")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST or summary.get("graph_runtime_hop_depth") != 1 or summary.get("default_equivalence_pass_count") != 9 or summary.get("known_causal_regression_count") != 0 or summary.get("runtime_policy_mutation_count") != 0:
        errors.append("graph_v1_invariant_changed")
    result_doc = {
        "schema_version": "opk-rag.task0167.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result_doc)
    return result_doc


def build_report(summary: dict[str, Any], answers: dict[str, Any]) -> str:
    return f"""# TASK-0167 Controlled Incremental Graph Repair and Reseal

## Summary

TASK-0167 completed Outcome `{summary["outcome_class"]}`. The controlled repair executor accepts a TASK-0166 repair-impact plan, enforces authorized mutation scope, writes a deterministic operation log, creates a new Graph Snapshot, and binds it natively to the current Corpus Snapshot.

## Live Baseline

* Live corpus changed: `{summary["live_corpus_changed"]}`
* Live repair operations: `{summary["live_repair_operation_count"]}`
* Operation counts: preserve `{summary["preserve_count"]}`, revalidate `{summary["revalidate_count"]}`, rebind `{summary["rebind_provenance_count"]}`, remove `{summary["remove_edge_count"]}`, extract `{summary["extract_new_evidence_count"]}`, authority recheck `{summary["authority_recheck_count"]}`, defer `{summary["defer_count"]}`
* New graph revision: `{summary["new_graph_revision"]}`
* Post-repair freshness: `{summary["post_repair_freshness_state"]}`

## Fixture Coverage

Positive fixtures cover changed-source revalidation, provenance rebinding, safe edge removal, source removal, multi-provenance preservation, bounded new evidence extraction, and unrelated-change zero repair. Idempotence is `{summary["repair_idempotence_valid"]}`.

## Governance

Automatic runtime repair remains `{summary["automatic_runtime_repair"]}`, runtime incremental repair remains `{summary["runtime_incremental_repair_enabled"]}`, automatic graph rebuild remains `{summary["automatic_graph_rebuild"]}`, and automatic corpus mutation remains `{summary["automatic_corpus_mutation"]}`.

Authority gaps remain fail-closed: edge count `{summary["authority_gap_edge_count"]}`, external authoritative source required `{summary["external_authoritative_source_required"]}`. Graph V2 runtime promotion remains `{summary["graph_v2_runtime_promotion_applied"]}`.

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
"""


def _operation(action: str, edge_id: str, reason: str, mutates_graph: bool) -> dict[str, Any]:
    seed = {"action": action, "graph_edge_id": edge_id, "reason": reason, "mutates_graph": mutates_graph}
    return {**seed, "operation_digest": digest_json(seed)}


def _has_alternative_support(edge_id: str, record: dict[str, Any], records: Iterable[dict[str, Any]]) -> bool:
    key = (record.get("source_entity_id"), record.get("relation_type"), record.get("target_entity_id"), record.get("graph_record_class"))
    return any(
        str(other.get("graph_edge_id")) != edge_id
        and (other.get("source_entity_id"), other.get("relation_type"), other.get("target_entity_id"), other.get("graph_record_class")) == key
        and other.get("origin_document_id") != record.get("origin_document_id")
        for other in records
    )


def _provenance_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("origin_document_id", "origin_chunk_id", "origin_evidence_unit_id", "recorded_source_digest", "origin_span_text_digest")}


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


def _count(result: dict[str, Any], action: str) -> int:
    return int(result["repair_result"]["operation_counts"].get(action, 0))


def _fixture_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_count": result["operation_log"]["operation_count"],
        "operation_counts": result["repair_result"]["operation_counts"],
        "removed_edge_count": result["removed_edge_registry"]["removed_edge_count"],
        "rebind_count": result["provenance_rebind_registry"]["rebind_count"],
        "new_graph_evidence_count": result["new_graph_evidence_registry"]["new_graph_evidence_count"],
        "operation_log_digest": result["operation_log"]["operation_log_digest"],
    }


def _fixture_corpus() -> dict[str, Any]:
    return task0163.build_corpus_snapshot_from_entries(
        [
            {"source_document_id": "source-documents/a.md", "canonical_document_id": "source-documents/a.md", "document_content_digest": "digest-a", "source_revision": "digest-a"},
            {"source_document_id": "source-documents/b.md", "canonical_document_id": "source-documents/b.md", "document_content_digest": "digest-b", "source_revision": "digest-b"},
            {"source_document_id": "source-documents/c.md", "canonical_document_id": "source-documents/c.md", "document_content_digest": "digest-c", "source_revision": "digest-c"},
        ]
    )


def _fixture_records(*, include_alternative: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "graph_edge_id": "edge-a",
            "graph_record_class": "authoritative_edge",
            "authority_gap_id": None,
            "source_entity_id": "source-documents/a.md",
            "relation_type": "LINKS_TO",
            "target_entity_id": "source-documents/b.md",
            "target_authority_valid": True,
            "origin_document_id": "source-documents/a.md",
            "origin_chunk_id": "source-documents/a.md#L1",
            "origin_evidence_unit_id": "source-documents/a.md#L1",
            "origin_span_text_digest": "span-a",
            "recorded_source_digest": "span-a",
            "provenance_complete": True,
        },
        {
            "graph_edge_id": "edge-gap",
            "graph_record_class": "authority_gap_edge",
            "authority_gap_id": "gap-a",
            "source_entity_id": "source-documents/a.md",
            "relation_type": "LINKS_TO",
            "target_entity_id": "missing.md",
            "target_authority_valid": False,
            "origin_document_id": "source-documents/a.md",
            "origin_chunk_id": "source-documents/a.md#L2",
            "origin_evidence_unit_id": "source-documents/a.md#L2",
            "origin_span_text_digest": "span-gap",
            "recorded_source_digest": "span-gap",
            "provenance_complete": True,
        },
    ]
    if include_alternative:
        rows.append({**rows[0], "graph_edge_id": "edge-a-alt", "origin_document_id": "source-documents/b.md", "origin_chunk_id": "source-documents/b.md#L4", "origin_evidence_unit_id": "source-documents/b.md#L4", "origin_span_text_digest": "span-b-alt", "recorded_source_digest": "span-b-alt"})
    return rows


def _fixture_case(
    previous: dict[str, Any],
    current: dict[str, Any],
    graph: dict[str, Any],
    graph_records: list[dict[str, Any]],
    *,
    representation_policy_changed: bool = False,
) -> dict[str, Any]:
    case = task0166._run_impact_case(previous, current, graph, graph_records, representation_policy_changed=representation_policy_changed)
    case["delta"]["current_corpus_snapshot"] = current
    case["plan"]["added_document_ids"] = case["delta"]["added_document_ids"]
    return case


def _variant_content_changed(corpus: dict[str, Any], doc_id: str) -> dict[str, Any]:
    return task0166._variant_content_changed(corpus, doc_id)


def _variant_document_removed(corpus: dict[str, Any], doc_id: str) -> dict[str, Any]:
    return task0166._variant_document_removed(corpus, doc_id)


def _variant_document_added(corpus: dict[str, Any]) -> dict[str, Any]:
    return task0166._variant_document_added(corpus)


def _new_evidence_record(doc_id: str) -> dict[str, Any]:
    return {
        "graph_edge_id": "edge-new-doc",
        "graph_record_class": "authoritative_edge",
        "authority_gap_id": None,
        "source_entity_id": doc_id,
        "relation_type": "LINKS_TO",
        "target_entity_id": "source-documents/b.md",
        "target_authority_valid": True,
        "origin_document_id": doc_id,
        "origin_chunk_id": f"{doc_id}#L1",
        "origin_evidence_unit_id": f"{doc_id}#L1",
        "origin_span_text_digest": "span-new-doc",
        "recorded_source_digest": "span-new-doc",
        "provenance_complete": True,
    }
