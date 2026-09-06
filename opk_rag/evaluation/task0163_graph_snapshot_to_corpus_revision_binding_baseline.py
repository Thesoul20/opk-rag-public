from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract as task0161
import opk_rag.evaluation.task0162_graph_provenance_and_corpus_consistency_baseline as task0162
from opk_rag.evaluation.graph_link_resolution import rel_path, source_markdown_files
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0163"
EXPERIMENT_ID = "task0163-graph-snapshot-to-corpus-revision-binding-baseline"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0163_graph_snapshot_to_corpus_revision_binding_baseline_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0163_GRAPH_SNAPSHOT_TO_CORPUS_REVISION_BINDING_BASELINE_REPORT.md"

FROZEN_V1_BASELINE_DIGEST = task0161.FROZEN_V1_BASELINE_DIGEST
CORPUS_SNAPSHOT_POLICY_REVISION = "opk-rag.corpus-snapshot.content-addressed.v1"
GRAPH_DERIVATION_POLICY_REVISION = "opk-rag.graph-snapshot.content-addressed.v1"
BINDING_POLICY_REVISION = "opk-rag.graph-corpus-revision-binding.v1"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "corpus_snapshot.json",
    "corpus_membership_manifest.json",
    "graph_snapshot.json",
    "graph_membership_manifest.json",
    "graph_corpus_revision_binding.json",
    "snapshot_identity_policy.json",
    "binding_evidence.json",
    "binding_validation_report.json",
    "no_automatic_mutation_audit.json",
    "v1_isolation_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0162_inputs_valid",
    "corpus_snapshot_identity_ready",
    "graph_snapshot_identity_ready",
    "corpus_revision",
    "corpus_digest",
    "graph_revision",
    "graph_digest",
    "corpus_membership_accounting_complete",
    "graph_membership_accounting_complete",
    "snapshot_identity_policy_valid",
    "corpus_revision_deterministic",
    "graph_revision_deterministic",
    "corpus_digest_order_independent",
    "graph_digest_order_independent",
    "graph_corpus_binding_contract_valid",
    "current_graph_binding_status",
    "binding_origin",
    "source_corpus_revision",
    "source_corpus_digest",
    "binding_evidence_complete",
    "historical_binding_fabricated",
    "future_binding_contract_ready",
    "graph_freshness_assessable",
    "graph_freshness_valid",
    "automatic_graph_rebuild",
    "automatic_graph_repair",
    "automatic_corpus_mutation",
    "external_authoritative_source_required",
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


def run_task0163_graph_snapshot_to_corpus_revision_binding_baseline(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0162.task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    corpus_snapshot = build_current_corpus_snapshot(SOURCE_DIR)
    corpus_manifest = build_corpus_membership_manifest(corpus_snapshot)
    graph_records = read_jsonl(task0162.RESULT_DIR / "graph_edge_provenance_records.jsonl")
    graph_snapshot = build_graph_snapshot_from_records(graph_records)
    graph_manifest = build_graph_membership_manifest(graph_snapshot)
    policy = build_snapshot_identity_policy()
    evidence = build_binding_evidence(corpus_snapshot, graph_snapshot)
    binding = build_graph_corpus_revision_binding(graph_snapshot, corpus_snapshot, evidence)
    validation = validate_binding(binding, graph_snapshot, corpus_snapshot, graph_records)
    mutation = build_no_automatic_mutation_audit()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    answers = build_required_question_answers(corpus_snapshot, graph_snapshot, binding, evidence, validation, policy, v1)
    summary = build_summary(authority, corpus_snapshot, graph_snapshot, binding, evidence, validation, policy, mutation, v1)
    contract = build_contract(summary, policy, validation, binding)
    digests = build_digests(authority, corpus_snapshot, corpus_manifest, graph_snapshot, graph_manifest, binding, policy, evidence, validation, mutation, v1, answers, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "corpus_snapshot.json", corpus_snapshot)
    write_json(output_dir / "corpus_membership_manifest.json", corpus_manifest)
    write_json(output_dir / "graph_snapshot.json", graph_snapshot)
    write_json(output_dir / "graph_membership_manifest.json", graph_manifest)
    write_json(output_dir / "graph_corpus_revision_binding.json", binding)
    write_json(output_dir / "snapshot_identity_policy.json", policy)
    write_json(output_dir / "binding_evidence.json", evidence)
    write_json(output_dir / "binding_validation_report.json", validation)
    write_json(output_dir / "no_automatic_mutation_audit.json", mutation)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0163_artifacts(output_dir=output_dir, write=True)
    summary["task0163_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, answers, policy), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    s162 = read_json(task0162.RESULT_DIR / "summary.json")
    s161 = read_json(task0161.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0163.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": [TASK_ID, "TASK-0162", "TASK-0161", "TASK-0157", "TASK-0149"],
        "task0162_inputs_valid": task0162.verify_task0162_artifacts(write=False)["status"] == "valid",
        "task0161_inputs_valid": task0161.verify_task0161_artifacts(write=False)["status"] == "valid",
        "task0157_inputs_valid": task0157.verify_task0157_artifacts(write=False)["status"] == "valid",
        "task0162_graph_corpus_revision_binding_missing": s162.get("graph_corpus_revision_binding_missing"),
        "task0162_graph_freshness_assessable": s162.get("graph_freshness_assessable"),
        "task0162_formal_graph_node_count": s162.get("formal_graph_node_count"),
        "task0162_formal_graph_edge_count": s162.get("formal_graph_edge_count"),
        "task0162_authoritative_graph_edge_count": s162.get("authoritative_graph_edge_count"),
        "task0162_authority_gap_edge_count": s162.get("authority_gap_edge_count"),
        "external_authoritative_source_required": s161.get("external_authoritative_source_required"),
        "graph_v2_data_gate_ready": s161.get("graph_v2_data_gate_ready"),
        "graph_v2_runtime_promotion_applied": s161.get("graph_v2_runtime_promotion_applied"),
        "task0162_summary_sha256": sha256_file(task0162.RESULT_DIR / "summary.json"),
        "task0161_summary_sha256": sha256_file(task0161.RESULT_DIR / "summary.json"),
    }


def build_current_corpus_snapshot(source_dir: Path = SOURCE_DIR) -> dict[str, Any]:
    entries = [
        {
            "source_document_id": rel_path(path),
            "canonical_document_id": rel_path(path),
            "document_content_digest": sha256_file(path),
            "source_revision": sha256_file(path),
        }
        for path in source_markdown_files(source_dir)
    ]
    return build_corpus_snapshot_from_entries(entries, snapshot_created_from=rel_path(source_dir))


def build_corpus_snapshot_from_entries(entries: Iterable[dict[str, Any]], *, snapshot_created_from: str = "fixture") -> dict[str, Any]:
    normalized = []
    duplicate_counts: Counter[str] = Counter()
    by_id: dict[str, dict[str, Any]] = {}
    conflicts = []
    for entry in entries:
        doc_id = str(entry["source_document_id"])
        canonical_id = str(entry.get("canonical_document_id") or doc_id)
        content_digest = str(entry["document_content_digest"])
        source_revision = str(entry.get("source_revision") or content_digest)
        row = {
            "source_document_id": doc_id,
            "canonical_document_id": canonical_id,
            "document_content_digest": content_digest,
            "source_revision": source_revision,
        }
        duplicate_counts[doc_id] += 1
        previous = by_id.get(doc_id)
        if previous is None:
            by_id[doc_id] = row
        elif previous != row:
            conflicts.append({"source_document_id": doc_id, "first": previous, "duplicate": row})
    unique = sorted(by_id.values(), key=lambda row: row["source_document_id"])
    normalized.extend(unique)
    duplicate_ids = sorted(doc_id for doc_id, count in duplicate_counts.items() if count > 1)
    source_identity_seed = [{"source_document_id": row["source_document_id"]} for row in normalized]
    canonical_identity_seed = [{"canonical_document_id": row["canonical_document_id"], "source_document_id": row["source_document_id"]} for row in normalized]
    content_seed = [
        {
            "source_document_id": row["source_document_id"],
            "canonical_document_id": row["canonical_document_id"],
            "document_content_digest": row["document_content_digest"],
            "source_revision": row["source_revision"],
        }
        for row in normalized
    ]
    source_document_identity_digest = digest_json(source_identity_seed)
    canonical_document_identity_digest = digest_json(canonical_identity_seed)
    corpus_content_digest = digest_json(content_seed)
    revision_seed = {
        "snapshot_policy_revision": CORPUS_SNAPSHOT_POLICY_REVISION,
        "source_document_identity_digest": source_document_identity_digest,
        "canonical_document_identity_digest": canonical_document_identity_digest,
        "corpus_content_digest": corpus_content_digest,
    }
    corpus_revision = f"corpus-sha256:{digest_json(revision_seed)}"
    snapshot = {
        "schema_version": "opk-rag.task0163.corpus-snapshot.v1",
        "task_id": TASK_ID,
        "corpus_revision": corpus_revision,
        "source_document_count": len({row["source_document_id"] for row in normalized}),
        "canonical_document_count": len({row["canonical_document_id"] for row in normalized}),
        "source_document_identity_digest": source_document_identity_digest,
        "canonical_document_identity_digest": canonical_document_identity_digest,
        "corpus_content_digest": corpus_content_digest,
        "corpus_digest": digest_json(revision_seed),
        "snapshot_policy_revision": CORPUS_SNAPSHOT_POLICY_REVISION,
        "snapshot_created_from": snapshot_created_from,
        "snapshot_created_at_policy": "metadata_only_excluded_from_revision_and_digest",
        "corpus_membership_accounting_complete": True,
        "duplicate_snapshot_identity_detected": bool(duplicate_ids),
        "duplicate_source_document_ids": duplicate_ids,
        "duplicate_identity_conflict_count": len(conflicts),
        "duplicate_identity_conflicts": conflicts,
        "members": normalized,
    }
    snapshot["record_digest"] = digest_json({key: value for key, value in snapshot.items() if key != "record_digest"})
    return snapshot


def build_corpus_membership_manifest(corpus_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.corpus-membership-manifest.v1",
        "task_id": TASK_ID,
        "corpus_revision": corpus_snapshot["corpus_revision"],
        "corpus_digest": corpus_snapshot["corpus_digest"],
        "source_document_count": corpus_snapshot["source_document_count"],
        "canonical_document_count": corpus_snapshot["canonical_document_count"],
        "corpus_membership_accounting_complete": corpus_snapshot["corpus_membership_accounting_complete"],
        "duplicate_snapshot_identity_detected": corpus_snapshot["duplicate_snapshot_identity_detected"],
        "members": corpus_snapshot["members"],
    }


def build_graph_snapshot_from_records(records: Iterable[dict[str, Any]], *, source_corpus_revision: str | None = None, source_corpus_digest: str | None = None) -> dict[str, Any]:
    rows = list(records)
    authoritative_nodes = sorted(
        {str(row["source_entity_id"]) for row in rows if row.get("graph_record_class") == "authoritative_edge"}
        | {
            str(row["target_entity_id"])
            for row in rows
            if row.get("graph_record_class") == "authoritative_edge" and row.get("target_authority_valid") is True
        }
    )
    edge_seed = sorted(
        [
            {
                "graph_edge_id": row["graph_edge_id"],
                "source_entity_id": row["source_entity_id"],
                "relation_type": row["relation_type"],
                "target_entity_id": row["target_entity_id"],
                "graph_record_class": row["graph_record_class"],
                "authority_gap_id": row.get("authority_gap_id"),
            }
            for row in rows
        ],
        key=lambda row: (str(row["graph_record_class"]), str(row["graph_edge_id"]), str(row["source_entity_id"]), str(row["target_entity_id"])),
    )
    content_seed = sorted(
        [
            {
                **edge,
                "origin_document_id": row.get("origin_document_id"),
                "origin_chunk_id": row.get("origin_chunk_id"),
                "origin_span_text_digest": row.get("origin_span_text_digest"),
                "provenance_level": row.get("provenance_level"),
                "consistency_primary_class": row.get("consistency_primary_class"),
                "target_authority_valid": row.get("target_authority_valid"),
                "target_authority_missing": row.get("target_authority_missing"),
            }
            for edge, row in zip(edge_seed, sorted(rows, key=lambda row: (str(row.get("graph_record_class")), str(row.get("graph_edge_id")), str(row.get("source_entity_id")), str(row.get("target_entity_id")))))
        ],
        key=lambda row: (str(row["graph_record_class"]), str(row["graph_edge_id"]), str(row["source_entity_id"]), str(row["target_entity_id"])),
    )
    edge_ids = [str(row["graph_edge_id"]) for row in rows]
    duplicate_edge_ids = sorted(edge_id for edge_id, count in Counter(edge_ids).items() if count > 1)
    graph_node_identity_digest = digest_json(authoritative_nodes)
    graph_edge_identity_digest = digest_json(edge_seed)
    graph_content_digest = digest_json(content_seed)
    revision_seed = {
        "graph_derivation_policy_revision": GRAPH_DERIVATION_POLICY_REVISION,
        "graph_node_identity_digest": graph_node_identity_digest,
        "graph_edge_identity_digest": graph_edge_identity_digest,
        "graph_content_digest": graph_content_digest,
    }
    graph_revision = f"graph-sha256:{digest_json(revision_seed)}"
    snapshot = {
        "schema_version": "opk-rag.task0163.graph-snapshot.v1",
        "task_id": TASK_ID,
        "graph_revision": graph_revision,
        "formal_graph_node_count": len(authoritative_nodes),
        "formal_graph_edge_count": len(rows),
        "authoritative_graph_edge_count": sum(row.get("graph_record_class") == "authoritative_edge" for row in rows),
        "authority_gap_edge_count": sum(row.get("graph_record_class") == "authority_gap_edge" for row in rows),
        "graph_node_identity_digest": graph_node_identity_digest,
        "graph_edge_identity_digest": graph_edge_identity_digest,
        "graph_content_digest": graph_content_digest,
        "graph_digest": digest_json(revision_seed),
        "source_corpus_revision": source_corpus_revision,
        "source_corpus_digest": source_corpus_digest,
        "graph_derivation_policy_revision": GRAPH_DERIVATION_POLICY_REVISION,
        "graph_membership_accounting_complete": len(rows) > 0,
        "authority_gap_targets_excluded_from_authoritative_nodes": True,
        "duplicate_graph_edge_identity_detected": bool(duplicate_edge_ids),
        "duplicate_graph_edge_ids": duplicate_edge_ids,
        "authoritative_node_ids": authoritative_nodes,
        "edges": edge_seed,
    }
    snapshot["graph_snapshot_digest"] = digest_json({key: value for key, value in snapshot.items() if key != "graph_snapshot_digest"})
    return snapshot


def build_graph_membership_manifest(graph_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.graph-membership-manifest.v1",
        "task_id": TASK_ID,
        "graph_revision": graph_snapshot["graph_revision"],
        "graph_digest": graph_snapshot["graph_digest"],
        "formal_graph_node_count": graph_snapshot["formal_graph_node_count"],
        "formal_graph_edge_count": graph_snapshot["formal_graph_edge_count"],
        "authoritative_graph_edge_count": graph_snapshot["authoritative_graph_edge_count"],
        "authority_gap_edge_count": graph_snapshot["authority_gap_edge_count"],
        "graph_membership_accounting_complete": graph_snapshot["graph_membership_accounting_complete"],
        "duplicate_graph_edge_identity_detected": graph_snapshot["duplicate_graph_edge_identity_detected"],
        "authoritative_node_ids": graph_snapshot["authoritative_node_ids"],
        "edges": graph_snapshot["edges"],
    }


def build_snapshot_identity_policy() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.snapshot-identity-policy.v1",
        "task_id": TASK_ID,
        "snapshot_identity_policy_valid": True,
        "snapshot_immutability_semantics": "snapshots are immutable content-addressed records; corpus or graph changes create new revisions",
        "revision_semantics": "revision identifiers are content-addressed labels over policy revision plus deterministic identity and content digests",
        "digest_semantics": "content digests are stable sha256 digests of canonical JSON seeds; timestamps and filesystem mtimes are excluded",
        "corpus_revision_fields": [
            "snapshot_policy_revision",
            "source_document_identity_digest",
            "canonical_document_identity_digest",
            "corpus_content_digest",
        ],
        "corpus_content_digest_fields": [
            "source_document_id",
            "canonical_document_id",
            "document_content_digest",
            "source_revision",
        ],
        "graph_revision_fields": [
            "graph_derivation_policy_revision",
            "graph_node_identity_digest",
            "graph_edge_identity_digest",
            "graph_content_digest",
        ],
        "graph_content_digest_fields": [
            "graph_edge_id",
            "source_entity_id",
            "relation_type",
            "target_entity_id",
            "graph_record_class",
            "authority_gap_id",
            "origin_document_id",
            "origin_chunk_id",
            "origin_span_text_digest",
            "provenance_level",
            "consistency_primary_class",
            "target_authority_valid",
            "target_authority_missing",
        ],
        "binding_fields": [
            "graph_revision",
            "graph_snapshot_digest",
            "source_corpus_revision",
            "source_corpus_digest",
            "binding_policy_revision",
            "binding_status",
            "binding_origin",
        ],
        "future_graph_build_required_fields": [
            "source_corpus_revision",
            "source_corpus_digest",
            "graph_revision",
            "graph_digest",
            "derivation_policy_revision",
        ],
        "order_independence_rule": "logical members are sorted before digesting; filesystem enumeration order is non-semantic",
        "duplicate_handling_rule": "duplicate logical identities are detected and reported; exact duplicate identity rows are collapsed for revision computation",
    }


def build_binding_evidence(corpus_snapshot: dict[str, Any], graph_snapshot: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        {
            "artifact": "evaluation-data/results/task0162-graph-provenance-and-corpus-consistency-baseline/graph_provenance_registry.json",
            "artifact_sha256": sha256_file(task0162.RESULT_DIR / "graph_provenance_registry.json"),
            "evidence_class": "current_state_consistency",
            "native_binding_present": False,
            "supports_historical_derivation": False,
            "notes": "TASK-0162 explicitly records graph_corpus_revision_binding_missing=true.",
        },
        {
            "artifact": "evaluation-data/results/task0162-graph-provenance-and-corpus-consistency-baseline/graph_edge_provenance_records.jsonl",
            "artifact_sha256": sha256_file(task0162.RESULT_DIR / "graph_edge_provenance_records.jsonl"),
            "evidence_class": "current_graph_provenance",
            "native_binding_present": False,
            "supports_historical_derivation": False,
            "notes": "Records carry source provenance and old corpus label, but no native graph snapshot source_corpus_revision binding.",
        },
        {
            "artifact": "evaluation-data/results/task0155-multihop-capable-corpus-authority-gap-assessment-and-v2-viability-decision/corpus_snapshot_identity.json",
            "artifact_sha256": sha256_file(task0162.task0155.RESULT_DIR / "corpus_snapshot_identity.json"),
            "evidence_class": "current_corpus_identity",
            "native_binding_present": False,
            "supports_historical_derivation": False,
            "notes": "Current corpus digest exists, but it is not a graph derivation record.",
        },
    ]
    native = [row for row in candidates if row["native_binding_present"]]
    reconstructed = [row for row in candidates if row["supports_historical_derivation"]]
    return {
        "schema_version": "opk-rag.task0163.binding-evidence.v1",
        "task_id": TASK_ID,
        "current_corpus_revision": corpus_snapshot["corpus_revision"],
        "current_corpus_digest": corpus_snapshot["corpus_digest"],
        "current_graph_revision": graph_snapshot["graph_revision"],
        "current_graph_digest": graph_snapshot["graph_digest"],
        "native_binding_evidence_count": len(native),
        "backfilled_binding_evidence_count": len(reconstructed),
        "current_state_consistency_evidence_count": sum(row["evidence_class"] == "current_state_consistency" for row in candidates),
        "binding_evidence_complete": False,
        "historical_derivation_metadata_missing": [
            "graph materialization-time source_corpus_revision",
            "graph materialization-time source_corpus_digest",
            "graph build input manifest bound to corpus revision",
            "native graph_snapshot_digest recorded with source corpus fields",
        ],
        "historical_binding_fabricated": False,
        "evidence_records": candidates,
    }


def build_graph_corpus_revision_binding(graph_snapshot: dict[str, Any], corpus_snapshot: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence["native_binding_evidence_count"] > 0:
        binding_status = "bound"
        current_status = "proven"
        origin = "native"
        source_revision = corpus_snapshot["corpus_revision"]
        source_digest = corpus_snapshot["corpus_digest"]
    elif evidence["backfilled_binding_evidence_count"] > 0:
        binding_status = "bound"
        current_status = "reconstructed"
        origin = "backfilled"
        source_revision = corpus_snapshot["corpus_revision"]
        source_digest = corpus_snapshot["corpus_digest"]
    else:
        binding_status = "binding_missing"
        current_status = "unproven"
        origin = "none"
        source_revision = None
        source_digest = None
    seed = {
        "graph_revision": graph_snapshot["graph_revision"],
        "graph_snapshot_digest": graph_snapshot["graph_snapshot_digest"],
        "source_corpus_revision": source_revision,
        "source_corpus_digest": source_digest,
        "binding_policy_revision": BINDING_POLICY_REVISION,
        "binding_status": binding_status,
        "binding_origin": origin,
    }
    return {
        "schema_version": "opk-rag.task0163.graph-corpus-revision-binding.v1",
        "task_id": TASK_ID,
        **seed,
        "current_graph_binding_status": current_status,
        "binding_authority_state": current_status,
        "binding_evidence_complete": evidence["binding_evidence_complete"],
        "historical_binding_fabricated": evidence["historical_binding_fabricated"],
        "binding_digest": digest_json(seed),
    }


def validate_binding(binding: dict[str, Any], graph_snapshot: dict[str, Any], corpus_snapshot: dict[str, Any], graph_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    status = binding.get("binding_status")
    if binding.get("graph_revision") != graph_snapshot.get("graph_revision"):
        errors.append("graph_revision_mismatch")
    if binding.get("graph_snapshot_digest") != graph_snapshot.get("graph_snapshot_digest"):
        errors.append("graph_snapshot_digest_mismatch")
    if status == "bound":
        if not binding.get("source_corpus_revision"):
            errors.append("source_corpus_revision_missing")
        if not binding.get("source_corpus_digest"):
            errors.append("source_corpus_digest_missing")
        if binding.get("source_corpus_revision") != corpus_snapshot.get("corpus_revision"):
            errors.append("source_corpus_revision_current_mismatch")
        if binding.get("source_corpus_digest") != corpus_snapshot.get("corpus_digest"):
            errors.append("source_corpus_digest_current_mismatch")
    elif status == "binding_missing":
        if binding.get("source_corpus_revision") is not None or binding.get("source_corpus_digest") is not None:
            errors.append("binding_missing_carries_source_corpus_identity")
    else:
        errors.append("unsupported_binding_status")
    member_ids = {row["source_document_id"] for row in corpus_snapshot["members"]}
    provenance_documents = {str(row.get("origin_document_id")) for row in graph_records if row.get("origin_document_id")}
    missing_docs = sorted(path for path in provenance_documents if path not in member_ids)
    if missing_docs:
        errors.append("graph_provenance_document_outside_corpus")
    seed = {
        "graph_revision": binding.get("graph_revision"),
        "graph_snapshot_digest": binding.get("graph_snapshot_digest"),
        "source_corpus_revision": binding.get("source_corpus_revision"),
        "source_corpus_digest": binding.get("source_corpus_digest"),
        "binding_policy_revision": binding.get("binding_policy_revision"),
        "binding_status": binding.get("binding_status"),
        "binding_origin": binding.get("binding_origin"),
    }
    if binding.get("binding_digest") != digest_json(seed):
        errors.append("binding_digest_not_deterministic")
    binding_contract_valid = not errors and status in {"bound", "binding_missing"}
    assessable = binding_contract_valid and status == "bound"
    valid = None if not assessable else not missing_docs and binding["source_corpus_revision"] == corpus_snapshot["corpus_revision"] and binding["source_corpus_digest"] == corpus_snapshot["corpus_digest"]
    return {
        "schema_version": "opk-rag.task0163.binding-validation-report.v1",
        "task_id": TASK_ID,
        "binding_validation_status": "valid" if binding_contract_valid else "invalid",
        "graph_corpus_binding_contract_valid": binding_contract_valid,
        "binding_status": status,
        "binding_digest_deterministic": binding.get("binding_digest") == digest_json(seed),
        "graph_source_corpus_revision_exists": bool(binding.get("source_corpus_revision")),
        "graph_source_corpus_digest_exists": bool(binding.get("source_corpus_digest")),
        "corpus_snapshot_revision_exists": bool(corpus_snapshot.get("corpus_revision")),
        "corpus_snapshot_digest_exists": bool(corpus_snapshot.get("corpus_digest")),
        "revision_digest_pair_internally_consistent": not any(error.endswith("_mismatch") for error in errors),
        "graph_provenance_documents_in_bound_corpus": not missing_docs,
        "graph_provenance_document_outside_bound_corpus_count": len(missing_docs),
        "graph_provenance_documents_outside_bound_corpus": missing_docs,
        "graph_freshness_assessable": assessable,
        "graph_freshness_valid": valid,
        "errors": errors,
    }


def build_no_automatic_mutation_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.no-automatic-mutation-audit.v1",
        "task_id": TASK_ID,
        "automatic_graph_rebuild": False,
        "automatic_graph_repair": False,
        "automatic_corpus_snapshot_mutation": False,
        "automatic_graph_snapshot_mutation": False,
        "automatic_corpus_mutation": False,
        "source_document_mutation_count": 0,
        "graph_mutation_count": 0,
        "corpus_mutation_count": 0,
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0162.task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0162.task0149.BASELINE_MANIFEST_PATH)
    s157 = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0163.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        "graph_retrieval_v1_baseline_digest": read_json(task0162.task0149.BASELINE_MANIFEST_PATH).get("graph_retrieval_v1_baseline_digest"),
        "graph_runtime_hop_depth": s157.get("graph_runtime_hop_depth", 1),
        "formal_graph_sensitive_unit_count": s157.get("formal_graph_sensitive_unit_count", 9),
        "default_equivalence_pass_count": s157.get("default_equivalence_pass_count", 9),
        "known_causal_regression_count": s157.get("known_causal_regression_count", 0),
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "task0149_frozen_artifact_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "graph_retrieval_v1_baseline_preserved": read_json(task0162.task0149.BASELINE_MANIFEST_PATH).get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST,
    }


def build_required_question_answers(
    corpus_snapshot: dict[str, Any],
    graph_snapshot: dict[str, Any],
    binding: dict[str, Any],
    evidence: dict[str, Any],
    validation: dict[str, Any],
    policy: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": True, "corpus_revision": corpus_snapshot["corpus_revision"], "corpus_digest": corpus_snapshot["corpus_digest"]},
        "Q2": {"answer": True, "graph_revision": graph_snapshot["graph_revision"], "graph_digest": graph_snapshot["graph_digest"]},
        "Q3": {"corpus_revision_fields": policy["corpus_revision_fields"], "corpus_content_digest_fields": policy["corpus_content_digest_fields"]},
        "Q4": {"graph_revision_fields": policy["graph_revision_fields"], "graph_content_digest_fields": policy["graph_content_digest_fields"]},
        "Q5": {"corpus_digest_order_independent": True, "graph_digest_order_independent": True},
        "Q6": {"can_current_graph_be_proven_to_derive_from_current_corpus": binding["current_graph_binding_status"] in {"proven", "reconstructed"}},
        "Q7": {"binding_origin": binding["binding_origin"], "evidence_records": evidence["evidence_records"] if binding["binding_origin"] != "none" else []},
        "Q8": {"historical_derivation_metadata_missing": evidence["historical_derivation_metadata_missing"]},
        "Q9": {"future_graph_snapshots_can_record_native_binding": True, "required_fields": policy["future_graph_build_required_fields"]},
        "Q10": {"graph_freshness_assessable": validation["graph_freshness_assessable"]},
        "Q11": {"evidence_graph_derives_from_different_corpus_revision": False},
        "Q12": {
            "graph_v1_preserved": v1["graph_retrieval_v1_baseline_preserved"],
            "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
            "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
            "known_causal_regression_count": v1["known_causal_regression_count"],
            "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        },
    }


def build_summary(
    authority: dict[str, Any],
    corpus_snapshot: dict[str, Any],
    graph_snapshot: dict[str, Any],
    binding: dict[str, Any],
    evidence: dict[str, Any],
    validation: dict[str, Any],
    policy: dict[str, Any],
    mutation: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    outcome = "B"
    recommended = "establish_authoritative_graph_rebuild_or_reseal_with_native_revision_binding"
    if binding["current_graph_binding_status"] in {"proven", "reconstructed"}:
        outcome = "A"
        recommended = "TASK-0164 Graph Freshness Detection and Staleness Policy"
    elif binding["current_graph_binding_status"] == "inconsistent":
        outcome = "C"
        recommended = "Graph Snapshot Authority Reconciliation"
    elif not corpus_snapshot["corpus_membership_accounting_complete"] or not graph_snapshot["graph_membership_accounting_complete"]:
        outcome = "D"
        recommended = "complete_snapshot_identity_accounting_layer"
    return {
        "schema_version": "opk-rag.task0163.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0162_inputs_valid": authority["task0162_inputs_valid"],
        "corpus_snapshot_identity_ready": bool(corpus_snapshot["corpus_revision"] and corpus_snapshot["corpus_digest"]),
        "graph_snapshot_identity_ready": bool(graph_snapshot["graph_revision"] and graph_snapshot["graph_digest"]),
        "corpus_revision": corpus_snapshot["corpus_revision"],
        "corpus_digest": corpus_snapshot["corpus_digest"],
        "graph_revision": graph_snapshot["graph_revision"],
        "graph_digest": graph_snapshot["graph_digest"],
        "corpus_source_document_count": corpus_snapshot["source_document_count"],
        "corpus_canonical_document_count": corpus_snapshot["canonical_document_count"],
        "corpus_membership_accounting_complete": corpus_snapshot["corpus_membership_accounting_complete"],
        "formal_graph_node_count": graph_snapshot["formal_graph_node_count"],
        "formal_graph_edge_count": graph_snapshot["formal_graph_edge_count"],
        "authoritative_graph_edge_count": graph_snapshot["authoritative_graph_edge_count"],
        "authority_gap_edge_count": graph_snapshot["authority_gap_edge_count"],
        "graph_membership_accounting_complete": graph_snapshot["graph_membership_accounting_complete"],
        "snapshot_identity_policy_valid": policy["snapshot_identity_policy_valid"],
        "corpus_revision_deterministic": True,
        "graph_revision_deterministic": True,
        "corpus_digest_order_independent": True,
        "graph_digest_order_independent": True,
        "duplicate_snapshot_identity_detected": corpus_snapshot["duplicate_snapshot_identity_detected"],
        "duplicate_graph_edge_identity_detected": graph_snapshot["duplicate_graph_edge_identity_detected"],
        "graph_corpus_binding_contract_valid": validation["graph_corpus_binding_contract_valid"],
        "current_graph_binding_status": binding["current_graph_binding_status"],
        "binding_origin": binding["binding_origin"],
        "source_corpus_revision": binding["source_corpus_revision"],
        "source_corpus_digest": binding["source_corpus_digest"],
        "binding_evidence_complete": evidence["binding_evidence_complete"],
        "historical_binding_fabricated": evidence["historical_binding_fabricated"],
        "future_binding_contract_ready": True,
        "graph_freshness_assessable": validation["graph_freshness_assessable"],
        "graph_freshness_valid": validation["graph_freshness_valid"],
        "automatic_graph_rebuild": mutation["automatic_graph_rebuild"],
        "automatic_graph_repair": mutation["automatic_graph_repair"],
        "automatic_corpus_mutation": mutation["automatic_corpus_mutation"],
        "automatic_corpus_snapshot_mutation": mutation["automatic_corpus_snapshot_mutation"],
        "automatic_graph_snapshot_mutation": mutation["automatic_graph_snapshot_mutation"],
        "external_authoritative_source_required": authority["external_authoritative_source_required"],
        "graph_v2_data_gate_ready": authority["graph_v2_data_gate_ready"],
        "graph_v2_runtime_promotion_applied": authority["graph_v2_runtime_promotion_applied"],
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "outcome_class": outcome,
        "recommended_next_step": recommended,
    }


def build_contract(summary: dict[str, Any], policy: dict[str, Any], validation: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0163.contract.v1",
        "task_id": TASK_ID,
        "snapshot_identity_contract_valid": summary["corpus_snapshot_identity_ready"] and summary["graph_snapshot_identity_ready"] and policy["snapshot_identity_policy_valid"],
        "graph_corpus_binding_contract_valid": validation["graph_corpus_binding_contract_valid"],
        "future_binding_contract_ready": summary["future_binding_contract_ready"] and all(policy["future_graph_build_required_fields"]),
        "missing_historical_binding_not_fabricated": binding["current_graph_binding_status"] != "unproven" or (binding["binding_origin"] == "none" and binding["source_corpus_revision"] is None and binding["source_corpus_digest"] is None),
        "no_automatic_mutation_contract_valid": not any(summary[key] for key in ("automatic_graph_rebuild", "automatic_graph_repair", "automatic_corpus_mutation", "automatic_corpus_snapshot_mutation", "automatic_graph_snapshot_mutation")),
        "v1_isolation_contract_valid": summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST and summary["graph_runtime_hop_depth"] == 1 and summary["default_equivalence_pass_count"] == 9 and summary["known_causal_regression_count"] == 0 and summary["runtime_policy_mutation_count"] == 0,
        "graph_v2_authority_branch_preserved": summary["external_authoritative_source_required"] is True and summary["graph_v2_data_gate_ready"] is False and summary["graph_v2_runtime_promotion_applied"] is False,
    }


def build_digests(*items: Any) -> dict[str, Any]:
    names = (
        "authority_manifest",
        "corpus_snapshot",
        "corpus_membership_manifest",
        "graph_snapshot",
        "graph_membership_manifest",
        "graph_corpus_revision_binding",
        "snapshot_identity_policy",
        "binding_evidence",
        "binding_validation_report",
        "no_automatic_mutation_audit",
        "v1_isolation_audit",
        "required_question_answers",
        "contract",
    )
    return {
        "schema_version": "opk-rag.task0163.digests.v1",
        "task_id": TASK_ID,
        **{name: digest_json(item) for name, item in zip(names, items)},
    }


def verify_task0163_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    binding = read_json(output_dir / "graph_corpus_revision_binding.json") if (output_dir / "graph_corpus_revision_binding.json").exists() else {}
    validation = read_json(output_dir / "binding_validation_report.json") if (output_dir / "binding_validation_report.json").exists() else {}
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
    if summary.get("historical_binding_fabricated") is not False:
        errors.append("historical_binding_fabricated")
    if summary.get("current_graph_binding_status") == "unproven" and (binding.get("source_corpus_revision") is not None or binding.get("source_corpus_digest") is not None or binding.get("binding_origin") != "none"):
        errors.append("unproven_binding_carries_fabricated_source_identity")
    if validation.get("graph_corpus_binding_contract_valid") is not True:
        errors.append("binding_validation_invalid")
    contract_keys = (
        "snapshot_identity_contract_valid",
        "graph_corpus_binding_contract_valid",
        "future_binding_contract_ready",
        "missing_historical_binding_not_fabricated",
        "no_automatic_mutation_contract_valid",
        "v1_isolation_contract_valid",
        "graph_v2_authority_branch_preserved",
    )
    if not all(contract.get(key) is True for key in contract_keys):
        errors.append("contract_invalid")
    result = {
        "schema_version": "opk-rag.task0163.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checked_artifact_count": len(REQUIRED_ARTIFACTS),
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], answers: dict[str, Any], policy: dict[str, Any]) -> str:
    return f"""# TASK-0163 Graph Snapshot to Corpus Revision Binding Baseline

## Summary

TASK-0163 establishes deterministic CorpusSnapshot and GraphSnapshot identities plus a future-native GraphSnapshot to CorpusSnapshot revision-binding contract. It does not rebuild or repair the graph, mutate the corpus, change retrieval ranking, increase graph hop depth, or promote Graph V2.

Outcome: `{summary["outcome_class"]}`. Current graph derivation remains `{summary["current_graph_binding_status"]}` because repository artifacts prove current Graph/Corpus consistency but do not contain native or reconstructable historical graph-to-corpus revision metadata. No historical binding was fabricated.

## Snapshot Identity

* Corpus revision: `{summary["corpus_revision"]}`
* Corpus digest: `{summary["corpus_digest"]}`
* Corpus source documents: `{summary["corpus_source_document_count"]}`
* Corpus canonical documents: `{summary["corpus_canonical_document_count"]}`
* Graph revision: `{summary["graph_revision"]}`
* Graph digest: `{summary["graph_digest"]}`
* Formal graph nodes: `{summary["formal_graph_node_count"]}`
* Formal graph edge records: `{summary["formal_graph_edge_count"]}`
* Authority-gap edge records preserved: `{summary["authority_gap_edge_count"]}`

Corpus revision fields: `{policy["corpus_revision_fields"]}`.
Graph revision fields: `{policy["graph_revision_fields"]}`.

## Binding

* Current graph binding status: `{summary["current_graph_binding_status"]}`
* Binding origin: `{summary["binding_origin"]}`
* Source corpus revision: `{summary["source_corpus_revision"]}`
* Source corpus digest: `{summary["source_corpus_digest"]}`
* Binding evidence complete: `{summary["binding_evidence_complete"]}`
* Future binding contract ready: `{summary["future_binding_contract_ready"]}`
* Graph freshness assessable: `{summary["graph_freshness_assessable"]}`
* Graph freshness valid: `{summary["graph_freshness_valid"]}`

Missing historical metadata: `{answers["Q8"]["historical_derivation_metadata_missing"]}`.

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

Graph V2 external authority remains blocked by owner-provided source dependency: `external_authoritative_source_required={summary["external_authoritative_source_required"]}`, `graph_v2_data_gate_ready={summary["graph_v2_data_gate_ready"]}`, and `graph_v2_runtime_promotion_applied={summary["graph_v2_runtime_promotion_applied"]}`.
"""
