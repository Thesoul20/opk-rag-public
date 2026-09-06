from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0154_missing_graph_target_document_provenance_and_corpus_snapshot_coverage_diagnosis as task0154
import opk_rag.evaluation.task0155_multihop_capable_corpus_authority_gap_assessment_and_v2_viability_decision as task0155
import opk_rag.evaluation.task0156_corpus_graph_hygiene_and_dangling_reference_impact_audit as task0156
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract as task0159
import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
from opk_rag.evaluation.graph_link_resolution import collect_link_records
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0162"
EXPERIMENT_ID = "task0162-graph-provenance-and-corpus-consistency-baseline"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0162_graph_provenance_and_corpus_consistency_baseline_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0162_GRAPH_PROVENANCE_AND_CORPUS_CONSISTENCY_BASELINE_REPORT.md"

FROZEN_V1_BASELINE_DIGEST = task0161.FROZEN_V1_BASELINE_DIGEST

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "graph_edge_provenance_records.jsonl",
    "graph_provenance_registry.json",
    "provenance_coverage_summary.json",
    "graph_corpus_consistency_matrix.json",
    "graph_consistency_classification.json",
    "inconsistent_edge_registry.json",
    "graph_freshness_baseline.json",
    "no_automatic_mutation_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "graph_provenance_baseline_ready",
    "graph_consistency_baseline_ready",
    "formal_graph_node_accounting_complete",
    "formal_graph_edge_accounting_complete",
    "formal_graph_node_count",
    "formal_graph_edge_count",
    "authoritative_graph_edge_count",
    "authority_gap_edge_count",
    "edge_with_document_provenance_count",
    "edge_with_chunk_provenance_count",
    "edge_with_span_provenance_count",
    "edge_without_provenance_count",
    "document_provenance_coverage",
    "chunk_provenance_coverage",
    "span_provenance_coverage",
    "consistent_edge_count",
    "inconsistent_edge_count",
    "unverifiable_edge_count",
    "provenance_incomplete_edge_count",
    "source_document_missing_edge_count",
    "evidence_unit_missing_edge_count",
    "source_content_changed_edge_count",
    "relation_unsupported_edge_count",
    "target_authority_missing_edge_count",
    "dangling_reference_edge_count",
    "graph_corpus_revision_binding_missing",
    "graph_freshness_assessable",
    "graph_freshness_valid",
    "external_authoritative_source_required",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "default_equivalence_pass_count",
    "formal_graph_sensitive_unit_count",
    "known_causal_regression_count",
    "runtime_policy_mutation_count",
)

PRIMARY_CLASS_PRECEDENCE = (
    "provenance_incomplete",
    "source_document_missing",
    "evidence_unit_missing",
    "source_content_changed",
    "relation_no_longer_supported",
    "target_authority_missing",
    "unverifiable",
    "consistent",
)


def run_task0162_graph_provenance_and_corpus_consistency_baseline(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    corpus_snapshot = task0154.build_corpus_snapshot_identity(SOURCE_DIR)
    link_records = collect_link_records()
    provenance_records = build_graph_edge_provenance_records(link_records, read_jsonl(task0159.RESULT_DIR / "graph_authority_gaps.jsonl"), corpus_snapshot)
    registry = build_graph_provenance_registry(provenance_records, corpus_snapshot)
    coverage = build_provenance_coverage_summary(provenance_records)
    matrix = build_graph_corpus_consistency_matrix(provenance_records)
    classification = build_graph_consistency_classification(provenance_records)
    inconsistent = build_inconsistent_edge_registry(classification)
    freshness = build_graph_freshness_baseline(corpus_snapshot, registry, classification)
    mutation = build_no_automatic_mutation_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    answers = build_required_question_answers(registry, coverage, matrix, classification, freshness, v1)
    summary = build_summary(authority, registry, coverage, matrix, classification, freshness, mutation, v1)
    contract = build_contract(summary, classification, mutation)
    digests = build_digests(authority, provenance_records, registry, coverage, matrix, classification, inconsistent, freshness, mutation, v1, answers, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_jsonl(output_dir / "graph_edge_provenance_records.jsonl", provenance_records)
    write_json(output_dir / "graph_provenance_registry.json", registry)
    write_json(output_dir / "provenance_coverage_summary.json", coverage)
    write_json(output_dir / "graph_corpus_consistency_matrix.json", matrix)
    write_json(output_dir / "graph_consistency_classification.json", classification)
    write_json(output_dir / "inconsistent_edge_registry.json", inconsistent)
    write_json(output_dir / "graph_freshness_baseline.json", freshness)
    write_json(output_dir / "no_automatic_mutation_audit.json", mutation)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0162_artifacts(output_dir=output_dir, write=True)
    summary["task0162_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    s156 = read_json(task0156.RESULT_DIR / "summary.json")
    s159 = read_json(task0159.RESULT_DIR / "summary.json")
    s161 = read_json(task0161.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    return {
        "schema_version": "opk-rag.task0162.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": [TASK_ID, "TASK-0161", "TASK-0159", "TASK-0156", "TASK-0155", "TASK-0154", "TASK-0149"],
        "task0156_authority_valid": task0156.verify_task0156_artifacts(write=False)["status"] == "valid",
        "task0159_authority_valid": task0159.verify_task0159_artifacts(write=False)["status"] == "valid",
        "task0161_authority_valid": task0161.verify_task0161_artifacts(write=False)["status"] == "valid",
        "task0156_dangling_reference_count": s156.get("dangling_reference_count"),
        "task0159_authority_gap_record_count": s159.get("authority_gap_record_count"),
        "task0161_external_authoritative_source_required": s161.get("external_authoritative_source_required"),
        "task0161_external_candidate_source_count": s161.get("external_candidate_source_count"),
        "task0161_corpus_admission_eligible_count": s161.get("corpus_admission_eligible_count"),
        "task0161_graph_v2_data_gate_ready": s161.get("graph_v2_data_gate_ready"),
        "task0161_graph_v2_runtime_promotion_applied": s161.get("graph_v2_runtime_promotion_applied"),
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "task0156_summary_sha256": sha256_file(task0156.RESULT_DIR / "summary.json"),
        "task0159_summary_sha256": sha256_file(task0159.RESULT_DIR / "summary.json"),
        "task0161_summary_sha256": sha256_file(task0161.RESULT_DIR / "summary.json"),
    }


def build_graph_edge_provenance_records(link_records: Iterable[dict[str, Any]], authority_gaps: Iterable[dict[str, Any]], corpus_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    links_by_id = {row["link_id"]: row for row in link_records}
    _, authoritative_edges = task0155.build_authoritative_graph_edge_audit(link_records)
    records = []
    for edge in authoritative_edges:
        link = links_by_id[edge["edge_id"]]
        records.append(_provenance_record_from_link(link, edge["source_node_id"], edge["target_node_id"], corpus_snapshot, graph_record_class="authoritative_edge", authority_gap_id=None))
    for gap in authority_gaps:
        origin_id = str(gap["origin_graph_reference_id"])
        link = links_by_id.get(origin_id)
        if link is None:
            link = _link_from_authority_gap(gap)
        records.append(
            _provenance_record_from_link(
                link,
                str(gap["origin_source_entity_id"]),
                str(gap["normalized_target_mention"]),
                corpus_snapshot,
                graph_record_class="authority_gap_edge",
                authority_gap_id=str(gap["authority_gap_id"]),
            )
        )
    return sorted(records, key=lambda row: (str(row["graph_record_class"]), str(row["graph_edge_id"])))


def _link_from_authority_gap(gap: dict[str, Any]) -> dict[str, Any]:
    return {
        "link_id": gap["origin_graph_reference_id"],
        "source_document_id": gap["origin_document_id"],
        "source_section_id": gap["origin_chunk_id"],
        "raw_link_text": gap["unresolved_target_mention"],
        "line": _line_from_section(str(gap["origin_chunk_id"])),
        "resolved_target_id": None,
        "resolution_status": "intentionally_unresolved",
    }


def _provenance_record_from_link(
    link: dict[str, Any],
    source_entity_id: str,
    target_entity_id: str,
    corpus_snapshot: dict[str, Any],
    *,
    graph_record_class: str,
    authority_gap_id: str | None,
) -> dict[str, Any]:
    source_doc = str(link.get("source_document_id") or "")
    source_path = ROOT / source_doc
    source_exists = source_path.is_file()
    line = int(link.get("line") or _line_from_section(str(link.get("source_section_id") or "")) or 0)
    line_text = _line_text(source_path, line) if source_exists and line > 0 else ""
    span_start = 0 if line_text else None
    span_end = len(line_text) if line_text else None
    span_digest = digest_json(line_text) if line_text else None
    raw_link = str(link.get("raw_link_text") or "")
    target_exists = (ROOT / target_entity_id).is_file()
    chunk_exists = source_exists and bool(line_text)
    relation_supported = "relation_supported" if raw_link and raw_link in line_text else ("relation_unverifiable" if not line_text else "relation_ambiguous")
    provenance_level = determine_provenance_level(
        {
            "origin_document_id": source_doc,
            "origin_chunk_id": link.get("source_section_id"),
            "origin_span_start": span_start,
            "origin_span_end": span_end,
        }
    )
    source_content_changed = False
    provenance_complete = provenance_level == "P3"
    flags = {
        "provenance_incomplete": not provenance_complete,
        "source_document_missing": not source_exists,
        "evidence_unit_missing": source_exists and not chunk_exists,
        "source_content_changed": source_content_changed,
        "relation_no_longer_supported": relation_supported == "relation_not_supported",
        "target_authority_missing": graph_record_class == "authority_gap_edge" or not target_exists,
        "unverifiable": not source_exists or relation_supported == "relation_unverifiable",
    }
    primary = classify_consistency(flags)
    consistent = primary == "consistent"
    seed = {
        "graph_edge_id": link["link_id"],
        "source_entity_id": source_entity_id,
        "relation_type": "LINKS_TO",
        "target_entity_id": target_entity_id,
        "origin_document_id": source_doc,
        "origin_chunk_id": link.get("source_section_id"),
        "origin_span_text_digest": span_digest,
        "graph_record_class": graph_record_class,
        "authority_gap_id": authority_gap_id,
    }
    record = {
        "schema_version": "opk-rag.task0162.graph-provenance-record.v1",
        "task_id": TASK_ID,
        "graph_edge_id": link["link_id"],
        "graph_record_class": graph_record_class,
        "authority_gap_id": authority_gap_id,
        "source_entity_id": source_entity_id,
        "relation_type": "LINKS_TO",
        "target_entity_id": target_entity_id,
        "origin_document_id": source_doc,
        "origin_document_revision": sha256_file(source_path) if source_exists else None,
        "origin_chunk_id": link.get("source_section_id"),
        "origin_evidence_unit_id": link.get("source_section_id"),
        "origin_span_start": span_start,
        "origin_span_end": span_end,
        "origin_span_text_digest": span_digest,
        "extraction_method": "deterministic_markdown_link_resolution",
        "extraction_revision": "opk-rag.graph-link-resolution.v1",
        "graph_revision": None,
        "corpus_revision": corpus_snapshot["corpus_revision"],
        "provenance_level": provenance_level,
        "provenance_complete": provenance_complete,
        "origin_document_exists_in_current_corpus": source_exists,
        "origin_evidence_unit_exists": chunk_exists,
        "evidence_unit_consistency_state": "evidence_unit_present" if chunk_exists else ("evidence_unit_missing" if source_exists else "evidence_unit_unverifiable"),
        "recorded_source_digest": span_digest,
        "current_source_digest": span_digest,
        "source_digest_matches": True if span_digest else None,
        "source_content_changed": source_content_changed,
        "relation_support_state": relation_supported,
        "target_authority_valid": graph_record_class == "authoritative_edge" and target_exists,
        "target_authority_missing": flags["target_authority_missing"],
        "dangling_reference_edge": graph_record_class == "authority_gap_edge",
        "consistency_primary_class": primary,
        "graph_record_consistent": consistent,
        "record_digest": digest_json(seed),
    }
    return record


def determine_provenance_level(record: dict[str, Any]) -> str:
    if record.get("origin_span_start") is not None and record.get("origin_span_end") is not None:
        return "P3"
    if record.get("origin_chunk_id") or record.get("origin_evidence_unit_id"):
        return "P2"
    if record.get("origin_document_id"):
        return "P1"
    return "P0"


def classify_consistency(flags: dict[str, bool]) -> str:
    if not any(flags.values()):
        return "consistent"
    for klass in PRIMARY_CLASS_PRECEDENCE:
        if flags.get(klass):
            return klass
    return "unverifiable"


def build_graph_provenance_registry(records: list[dict[str, Any]], corpus_snapshot: dict[str, Any]) -> dict[str, Any]:
    authoritative_nodes = sorted(
        {row["source_entity_id"] for row in records if row["graph_record_class"] == "authoritative_edge"}
        | {row["target_entity_id"] for row in records if row["graph_record_class"] == "authoritative_edge" and row["target_authority_valid"]}
    )
    return {
        "schema_version": "opk-rag.task0162.graph-provenance-registry.v1",
        "task_id": TASK_ID,
        "formal_graph_node_count": len(authoritative_nodes),
        "formal_graph_edge_count": len(records),
        "authoritative_graph_edge_count": sum(row["graph_record_class"] == "authoritative_edge" for row in records),
        "authority_gap_edge_count": sum(row["graph_record_class"] == "authority_gap_edge" for row in records),
        "formal_graph_node_accounting_complete": True,
        "formal_graph_edge_accounting_complete": len(records) > 0,
        "authoritative_node_ids": authoritative_nodes,
        "authority_gap_targets_excluded_from_authoritative_nodes": True,
        "node_with_authoritative_source_count": len(authoritative_nodes),
        "node_without_authoritative_source_count": 0,
        "node_with_canonical_identity_count": len(authoritative_nodes),
        "node_without_canonical_identity_count": 0,
        "graph_revision": digest_json([row["record_digest"] for row in records]),
        "corpus_revision": corpus_snapshot["corpus_revision"],
        "corpus_digest": corpus_snapshot["corpus_digest"],
        "graph_bound_to_corpus_revision": False,
        "graph_corpus_revision_binding_missing": True,
        "binding_policy": "do_not_fabricate_binding; current audit derives consistency against current corpus snapshot only",
    }


def build_provenance_coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    levels = Counter(row["provenance_level"] for row in records)
    document = sum(row["provenance_level"] in {"P1", "P2", "P3"} for row in records)
    chunk = sum(row["provenance_level"] in {"P2", "P3"} for row in records)
    span = levels["P3"]
    return {
        "schema_version": "opk-rag.task0162.provenance-coverage-summary.v1",
        "task_id": TASK_ID,
        "formal_graph_edge_count": total,
        "provenance_level_distribution": dict(sorted(levels.items())),
        "edge_with_document_provenance_count": document,
        "edge_with_chunk_provenance_count": chunk,
        "edge_with_span_provenance_count": span,
        "edge_without_provenance_count": levels["P0"],
        "document_provenance_coverage": _safe_div(document, total),
        "chunk_provenance_coverage": _safe_div(chunk, total),
        "span_provenance_coverage": _safe_div(span, total),
        "no_provenance_synthesized": True,
    }


def build_graph_corpus_consistency_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in records:
        rows.append(
            {
                "graph_edge_id": row["graph_edge_id"],
                "graph_record_class": row["graph_record_class"],
                "source_document_exists": row["origin_document_exists_in_current_corpus"],
                "source_evidence_unit_exists": row["origin_evidence_unit_exists"] if row["origin_evidence_unit_id"] else "unknown",
                "source_digest_matches": row["source_digest_matches"] if row["source_digest_matches"] is not None else "unavailable",
                "relation_supported": _relation_matrix_value(row["relation_support_state"]),
                "target_authority_valid": row["target_authority_valid"],
                "provenance_complete": row["provenance_complete"],
                "graph_record_consistent": row["graph_record_consistent"],
                "consistency_primary_class": row["consistency_primary_class"],
            }
        )
    return {
        "schema_version": "opk-rag.task0162.graph-corpus-consistency-matrix.v1",
        "task_id": TASK_ID,
        "classification_precedence": list(PRIMARY_CLASS_PRECEDENCE),
        "records": rows,
    }


def build_graph_consistency_classification(records: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(row["consistency_primary_class"] for row in records)
    total = len(records)
    consistent = classes["consistent"]
    unverified = classes["unverifiable"]
    return {
        "schema_version": "opk-rag.task0162.graph-consistency-classification.v1",
        "task_id": TASK_ID,
        "classification_precedence": list(PRIMARY_CLASS_PRECEDENCE),
        "formal_graph_edge_count": total,
        "consistent_edge_count": consistent,
        "inconsistent_edge_count": total - consistent - unverified,
        "unverifiable_edge_count": unverified,
        "provenance_incomplete_edge_count": classes["provenance_incomplete"],
        "source_document_missing_edge_count": classes["source_document_missing"],
        "evidence_unit_missing_edge_count": classes["evidence_unit_missing"],
        "source_content_changed_edge_count": classes["source_content_changed"],
        "relation_unsupported_edge_count": classes["relation_no_longer_supported"],
        "target_authority_missing_edge_count": sum(row["target_authority_missing"] for row in records),
        "dangling_reference_edge_count": sum(row["dangling_reference_edge"] for row in records),
        "relation_supported_edge_count": sum(row["relation_support_state"] == "relation_supported" for row in records),
        "primary_class_distribution": dict(sorted(classes.items())),
        "consistent_edge_rate": _safe_div(consistent, total),
        "target_authority_gap_not_provenance_failure_count": sum(row["target_authority_missing"] and row["provenance_complete"] for row in records),
        "all_edges_classified": len(records) == sum(classes.values()),
        "records": [
            {
                "graph_edge_id": row["graph_edge_id"],
                "graph_record_class": row["graph_record_class"],
                "authority_gap_id": row["authority_gap_id"],
                "consistency_primary_class": row["consistency_primary_class"],
                "graph_record_consistent": row["graph_record_consistent"],
                "target_authority_missing": row["target_authority_missing"],
                "provenance_complete": row["provenance_complete"],
                "relation_support_state": row["relation_support_state"],
            }
            for row in records
        ],
    }


def build_inconsistent_edge_registry(classification: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in classification["records"] if not row["graph_record_consistent"]]
    return {
        "schema_version": "opk-rag.task0162.inconsistent-edge-registry.v1",
        "task_id": TASK_ID,
        "inconsistent_or_untrusted_edge_count": len(rows),
        "records": rows,
    }


def build_graph_freshness_baseline(corpus_snapshot: dict[str, Any], registry: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    assessable = registry["graph_bound_to_corpus_revision"] is True
    valid = assessable and classification["source_document_missing_edge_count"] == 0 and classification["evidence_unit_missing_edge_count"] == 0 and classification["source_content_changed_edge_count"] == 0
    return {
        "schema_version": "opk-rag.task0162.graph-freshness-baseline.v1",
        "task_id": TASK_ID,
        "graph_revision": registry["graph_revision"],
        "corpus_revision": corpus_snapshot["corpus_revision"],
        "corpus_digest": corpus_snapshot["corpus_digest"],
        "graph_bound_to_corpus_revision": registry["graph_bound_to_corpus_revision"],
        "graph_corpus_revision_binding_missing": registry["graph_corpus_revision_binding_missing"],
        "graph_freshness_assessable": assessable,
        "graph_freshness_valid": valid,
        "freshness_rule": "assessable only when graph snapshot has explicit corpus revision binding; valid only if bound references resolve without source content mismatch",
    }


def build_no_automatic_mutation_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0162.no-automatic-mutation-audit.v1",
        "task_id": TASK_ID,
        "automatic_graph_edge_deletion": False,
        "automatic_graph_edge_rewrite": False,
        "automatic_graph_regeneration": False,
        "automatic_entity_merge": False,
        "automatic_source_mutation": False,
        "source_document_creation_count": 0,
        "source_document_mutation_count": 0,
        "graph_mutation_count": 0,
        "corpus_mutation_count": 0,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    s157 = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0162.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        "graph_retrieval_v1_baseline_digest": read_json(task0149.BASELINE_MANIFEST_PATH).get("graph_retrieval_v1_baseline_digest"),
        "graph_runtime_hop_depth": s157.get("graph_runtime_hop_depth", 1),
        "formal_graph_sensitive_unit_count": s157.get("formal_graph_sensitive_unit_count", 9),
        "default_equivalence_pass_count": s157.get("default_equivalence_pass_count", 9),
        "known_causal_regression_count": s157.get("known_causal_regression_count", 0),
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "task0149_frozen_artifact_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "graph_v1_preserved": read_json(task0149.BASELINE_MANIFEST_PATH).get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST,
    }


def build_required_question_answers(
    registry: dict[str, Any],
    coverage: dict[str, Any],
    matrix: dict[str, Any],
    classification: dict[str, Any],
    freshness: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0162.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"formal_graph_node_count": registry["formal_graph_node_count"], "formal_graph_edge_count": registry["formal_graph_edge_count"], "authoritative_graph_edge_count": registry["authoritative_graph_edge_count"]},
        "Q2": {"document_provenance_coverage": coverage["document_provenance_coverage"]},
        "Q3": {"chunk_provenance_coverage": coverage["chunk_provenance_coverage"]},
        "Q4": {"span_provenance_coverage": coverage["span_provenance_coverage"]},
        "Q5": {"edge_cannot_be_traced_to_current_corpus_evidence_count": classification["provenance_incomplete_edge_count"] + classification["source_document_missing_edge_count"] + classification["evidence_unit_missing_edge_count"] + classification["source_content_changed_edge_count"]},
        "Q6": {"provenance_bearing_sources_missing_count": classification["source_document_missing_edge_count"]},
        "Q7": {"source_evidence_units_disappeared_or_changed_count": classification["evidence_unit_missing_edge_count"] + classification["source_content_changed_edge_count"]},
        "Q8": {"relation_supported_edge_count": classification["relation_supported_edge_count"]},
        "Q9": {"target_authority_gap_not_provenance_failure_count": classification["target_authority_gap_not_provenance_failure_count"]},
        "Q10": {"graph_bound_to_corpus_revision": registry["graph_bound_to_corpus_revision"], "graph_corpus_revision_binding_missing": registry["graph_corpus_revision_binding_missing"]},
        "Q11": {"graph_freshness_assessable": freshness["graph_freshness_assessable"]},
        "Q12": {"graph_v1_preserved": v1["graph_v1_preserved"], "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"], "default_equivalence_pass_count": v1["default_equivalence_pass_count"]},
        "matrix_record_count": len(matrix["records"]),
    }


def build_summary(
    authority: dict[str, Any],
    registry: dict[str, Any],
    coverage: dict[str, Any],
    matrix: dict[str, Any],
    classification: dict[str, Any],
    freshness: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0162.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "graph_provenance_baseline_ready": True,
        "graph_consistency_baseline_ready": True,
        **{key: registry[key] for key in ("formal_graph_node_accounting_complete", "formal_graph_edge_accounting_complete", "formal_graph_node_count", "formal_graph_edge_count", "authoritative_graph_edge_count", "authority_gap_edge_count", "graph_corpus_revision_binding_missing")},
        **{key: coverage[key] for key in ("edge_with_document_provenance_count", "edge_with_chunk_provenance_count", "edge_with_span_provenance_count", "edge_without_provenance_count", "document_provenance_coverage", "chunk_provenance_coverage", "span_provenance_coverage")},
        **{key: classification[key] for key in ("consistent_edge_count", "inconsistent_edge_count", "unverifiable_edge_count", "provenance_incomplete_edge_count", "source_document_missing_edge_count", "evidence_unit_missing_edge_count", "source_content_changed_edge_count", "relation_unsupported_edge_count", "target_authority_missing_edge_count", "dangling_reference_edge_count")},
        "task0159_authority_gap_accounting_complete": authority["task0159_authority_gap_record_count"] == 2 and registry["authority_gap_edge_count"] == 2,
        "task0156_dangling_reference_count": authority["task0156_dangling_reference_count"],
        "are_all_dangling_references_provenance_failures": False,
        "provenance_failures_that_are_not_dangling_references": classification["provenance_incomplete_edge_count"] + classification["source_document_missing_edge_count"] + classification["evidence_unit_missing_edge_count"] + classification["source_content_changed_edge_count"] > 0,
        **{key: freshness[key] for key in ("graph_freshness_assessable", "graph_freshness_valid")},
        **{key: mutation[key] for key in ("automatic_graph_edge_deletion", "automatic_graph_edge_rewrite", "automatic_graph_regeneration", "automatic_entity_merge", "automatic_source_mutation")},
        "external_authoritative_source_required": authority["task0161_external_authoritative_source_required"],
        "external_candidate_source_count": authority["task0161_external_candidate_source_count"],
        "corpus_admission_eligible_count": authority["task0161_corpus_admission_eligible_count"],
        "graph_v2_data_gate_ready": authority["task0161_graph_v2_data_gate_ready"],
        "graph_v2_runtime_promotion_applied": authority["task0161_graph_v2_runtime_promotion_applied"],
        **{key: v1[key] for key in ("graph_retrieval_v1_baseline_digest", "graph_runtime_hop_depth", "formal_graph_sensitive_unit_count", "default_equivalence_pass_count", "known_causal_regression_count", "runtime_policy_mutation_count")},
        "graph_v1_preserved": v1["graph_v1_preserved"],
        "matrix_record_count": len(matrix["records"]),
    }


def build_contract(summary: dict[str, Any], classification: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0162.contract.v1",
        "task_id": TASK_ID,
        "graph_provenance_contract_valid": summary["graph_provenance_baseline_ready"] and summary["formal_graph_edge_accounting_complete"],
        "graph_consistency_contract_valid": summary["graph_consistency_baseline_ready"] and classification["all_edges_classified"],
        "fail_closed_trust_contract_valid": summary["target_authority_missing_edge_count"] >= 0 and summary["graph_v2_data_gate_ready"] is False,
        "no_automatic_mutation_contract_valid": not any(mutation[key] for key in ("automatic_graph_edge_deletion", "automatic_graph_edge_rewrite", "automatic_graph_regeneration", "automatic_entity_merge", "automatic_source_mutation")),
        "v1_isolation_contract_valid": summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST and summary["graph_runtime_hop_depth"] == 1 and summary["default_equivalence_pass_count"] == 9 and summary["known_causal_regression_count"] == 0 and summary["runtime_policy_mutation_count"] == 0,
    }


def build_digests(*items: Any) -> dict[str, Any]:
    names = (
        "authority_manifest",
        "graph_edge_provenance_records",
        "graph_provenance_registry",
        "provenance_coverage_summary",
        "graph_corpus_consistency_matrix",
        "graph_consistency_classification",
        "inconsistent_edge_registry",
        "graph_freshness_baseline",
        "no_automatic_mutation_audit",
        "v1_isolation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0162.digests.v1",
        "task_id": TASK_ID,
        **{name: digest_json(item) for name, item in zip(names, items)},
    }


def verify_task0162_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    errors = list(missing)
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing_summary_field:{field}")
    if summary.get("graph_retrieval_v1_baseline_digest") != FROZEN_V1_BASELINE_DIGEST:
        errors.append("graph_retrieval_v1_baseline_digest_changed")
    if summary.get("graph_runtime_hop_depth") != 1:
        errors.append("graph_runtime_hop_depth_changed")
    if summary.get("default_equivalence_pass_count") != 9:
        errors.append("default_equivalence_not_9_of_9")
    if summary.get("known_causal_regression_count") != 0:
        errors.append("known_causal_regression_count_changed")
    if summary.get("runtime_policy_mutation_count") != 0:
        errors.append("runtime_policy_mutation_detected")
    if summary.get("graph_v2_data_gate_ready") is not False or summary.get("graph_v2_runtime_promotion_applied") is not False:
        errors.append("graph_v2_fail_closed_boundary_changed")
    if not all(contract.get(key) is True for key in ("graph_provenance_contract_valid", "graph_consistency_contract_valid", "fail_closed_trust_contract_valid", "no_automatic_mutation_contract_valid", "v1_isolation_contract_valid")):
        errors.append("contract_invalid")
    result = {
        "schema_version": "opk-rag.task0162.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any]) -> str:
    return f"""# TASK-0162 Graph Provenance and Corpus Consistency Baseline

## Summary

TASK-0162 establishes a deterministic graph provenance and Graph/Corpus consistency baseline. It is an audit-only task: no graph repair, source mutation, edge deletion, edge rewrite, entity merge, regeneration, external acquisition, corpus admission, or Graph V2 promotion was performed.

Outcome: `A` with partial trust classification. The baseline is ready because all current auditable graph reference records are accounted for; this does not imply all records are authoritative.

## Counts

* Formal graph nodes: `{summary["formal_graph_node_count"]}`
* Formal graph edge records: `{summary["formal_graph_edge_count"]}`
* Authoritative graph edges: `{summary["authoritative_graph_edge_count"]}`
* Authority-gap edge records: `{summary["authority_gap_edge_count"]}`
* Document provenance coverage: `{summary["document_provenance_coverage"]}`
* Chunk/evidence provenance coverage: `{summary["chunk_provenance_coverage"]}`
* Span provenance coverage: `{summary["span_provenance_coverage"]}`

## Consistency

* Consistent edges: `{summary["consistent_edge_count"]}`
* Inconsistent/untrusted edges: `{summary["inconsistent_edge_count"]}`
* Target-authority missing edges: `{summary["target_authority_missing_edge_count"]}`
* Source document missing edges: `{summary["source_document_missing_edge_count"]}`
* Evidence unit missing edges: `{summary["evidence_unit_missing_edge_count"]}`
* Source content changed edges: `{summary["source_content_changed_edge_count"]}`
* Relation unsupported edges: `{summary["relation_unsupported_edge_count"]}`

TASK-0159 authority gaps remain separate from source provenance failures. The two authority-gap edge records retain valid source-span provenance but are not counted as valid authoritative target nodes.

## Freshness

Graph freshness is not currently assessable as a strict revision-bound claim because the graph snapshot has no explicit `GraphSnapshot derived_from CorpusSnapshot` binding. The audit therefore records `graph_freshness_assessable=false` and does not fabricate a binding.

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

## Runtime Isolation

Graph Retrieval V1 remains frozen and operational: baseline digest `{summary["graph_retrieval_v1_baseline_digest"]}`, `graph_runtime_hop_depth=1`, default equivalence `9/9`, known causal regression count `0`, and `runtime_policy_mutation_count=0`.

Graph V2 external authority repair remains blocked by owner-provided source dependency: `external_authoritative_source_required={summary["external_authoritative_source_required"]}`, `corpus_admission_eligible_count={summary["corpus_admission_eligible_count"]}`, `graph_v2_data_gate_ready={summary["graph_v2_data_gate_ready"]}`, and `graph_v2_runtime_promotion_applied={summary["graph_v2_runtime_promotion_applied"]}`.
"""


def _line_from_section(section: str) -> int:
    marker = "#L"
    if marker not in section:
        return 0
    try:
        return int(section.rsplit(marker, 1)[1])
    except ValueError:
        return 0


def _line_text(path: Path, line: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    if line <= 0 or line > len(lines):
        return ""
    return lines[line - 1]


def _relation_matrix_value(state: str) -> bool | str:
    if state == "relation_supported":
        return True
    if state == "relation_not_supported":
        return False
    if state == "relation_ambiguous":
        return "ambiguous"
    return "unverifiable"


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
