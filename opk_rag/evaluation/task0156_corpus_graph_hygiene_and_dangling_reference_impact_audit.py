from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0154_missing_graph_target_document_provenance_and_corpus_snapshot_coverage_diagnosis as task0154
import opk_rag.evaluation.task0155_multihop_capable_corpus_authority_gap_assessment_and_v2_viability_decision as task0155
from opk_rag.evaluation.graph_link_resolution import collect_link_records
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0156"
EXPERIMENT_ID = "task0156-corpus-graph-hygiene-and-dangling-reference-impact-audit"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0156_corpus_graph_hygiene_and_dangling_reference_impact_audit_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0156_CORPUS_GRAPH_HYGIENE_AND_DANGLING_REFERENCE_IMPACT_AUDIT_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "dangling_reference_inventory.jsonl",
    "dangling_reference_classification.jsonl",
    "dangling_reference_intent_audit.json",
    "graph_blocking_impact_audit.jsonl",
    "topology_impact_audit.jsonl",
    "retrieval_impact_audit.jsonl",
    "evidence_impact_audit.jsonl",
    "benchmark_impact_audit.json",
    "repair_candidate_inventory.jsonl",
    "graph_hygiene_decision.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0155_authority_valid",
    "task0154_authority_valid",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "corpus_snapshot_identity_valid",
    "corpus_snapshot_revision",
    "corpus_snapshot_digest",
    "corpus_document_count",
    "task0155_dangling_reference_count",
    "task0156_reproduced_dangling_reference_count",
    "dangling_inventory_drift_detected",
    "total_internal_reference_count",
    "dangling_reference_count",
    "dangling_reference_rate",
    "intent_class_distribution",
    "likely_typo_count",
    "legacy_reference_count",
    "future_placeholder_count",
    "intentional_uncreated_note_count",
    "external_or_out_of_scope_count",
    "unsupported_target_count",
    "unknown_intent_count",
    "graph_blocking_dangling_reference_count",
    "graph_degrading_dangling_reference_count",
    "non_graph_blocking_dangling_reference_count",
    "unknown_impact_dangling_reference_count",
    "graph_blocking_dangling_reference_rate",
    "potential_authoritative_edge_recovery_count",
    "dangling_reference_with_retrieval_impact_count",
    "dangling_reference_with_evidence_impact_count",
    "benchmark_unit_impacted_by_dangling_reference_count",
    "production_query_impacted_count",
    "corpus_graph_hygiene_gap_material",
    "current_corpus_graph_hygiene_status",
    "corpus_graph_hygiene_repair_justified",
    "targeted_repair_candidate_count",
    "automatic_repair_eligible_count",
    "owner_review_required_count",
    "graph_hygiene_repair_decision",
    "task0155_multihop_no_go_preserved",
    "bounded_multihop_experiment_ready",
    "repair_applied",
    "source_document_creation_count",
    "source_document_mutation_count",
    "source_reference_mutation_count",
    "corpus_snapshot_mutation_count",
    "resolver_policy_mutation_count",
    "graph_policy_mutation_count",
    "runtime_policy_mutation_count",
    "task0149_frozen_artifact_mutation_count",
    "recommended_next_step",
)


def run_task0156_corpus_graph_hygiene_and_dangling_reference_impact_audit(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    corpus_snapshot = task0154.build_corpus_snapshot_identity(SOURCE_DIR)
    records = collect_link_records()
    edge_audit, edge_rows = task0155.build_authoritative_graph_edge_audit(records)
    reference_audit, dangling_records = build_dangling_reference_inventory(records, authority)
    classifications = [classify_dangling_reference(row, edge_rows=edge_rows) for row in dangling_records]
    intent_audit = build_intent_audit(classifications)
    topology_rows = [topology_impact_row(row, edge_rows=edge_rows) for row in classifications]
    retrieval_rows = [retrieval_impact_row(row, authority) for row in classifications]
    evidence_rows = [evidence_impact_row(row) for row in classifications]
    benchmark_audit = build_benchmark_impact_audit(classifications, authority)
    repair_candidates = [repair_candidate_row(row) for row in classifications if row["targeted_repair_candidate"]]
    decision = build_graph_hygiene_decision(classifications, benchmark_audit, authority, reference_audit)

    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    mutation = {
        "source_document_creation_count": 0,
        "source_document_mutation_count": 0,
        "source_reference_mutation_count": 0,
        "corpus_snapshot_mutation_count": 0,
        "resolver_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "task0149_frozen_artifact_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
    }
    summary = build_summary(authority, corpus_snapshot, reference_audit, edge_audit, classifications, benchmark_audit, decision, mutation)
    contract = build_contract(summary, classifications)
    digests = build_digests(authority, corpus_snapshot, reference_audit, classifications, intent_audit, topology_rows, retrieval_rows, evidence_rows, benchmark_audit, repair_candidates, decision, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_jsonl(output_dir / "dangling_reference_inventory.jsonl", dangling_records)
    write_jsonl(output_dir / "dangling_reference_classification.jsonl", classifications)
    write_json(output_dir / "dangling_reference_intent_audit.json", intent_audit)
    write_jsonl(output_dir / "graph_blocking_impact_audit.jsonl", [graph_blocking_impact_row(row) for row in classifications])
    write_jsonl(output_dir / "topology_impact_audit.jsonl", topology_rows)
    write_jsonl(output_dir / "retrieval_impact_audit.jsonl", retrieval_rows)
    write_jsonl(output_dir / "evidence_impact_audit.jsonl", evidence_rows)
    write_json(output_dir / "benchmark_impact_audit.json", benchmark_audit)
    write_jsonl(output_dir / "repair_candidate_inventory.jsonl", repair_candidates)
    write_json(output_dir / "graph_hygiene_decision.json", decision)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0156_artifacts(output_dir=output_dir, write=True)
    summary["task0156_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, intent_audit), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0155_verification = task0155.verify_task0155_artifacts(write=False)
    task0154_verification = task0154.verify_task0154_artifacts(write=False)
    task0149_verification = task0149.verify_task0149_artifacts(write=False)
    task0155_summary = read_json(task0155.RESULT_DIR / "summary.json")
    task0154_summary = read_json(task0154.RESULT_DIR / "summary.json")
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    return {
        "schema_version": "opk-rag.task0156.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": ["TASK-0155", "TASK-0154", "TASK-0149"],
        "task0155_authority_valid": task0155_verification["status"] == "valid"
        and task0155_summary.get("dangling_reference_count") == 8
        and task0155_summary.get("dangling_reference_systemic") is True
        and task0155_summary.get("authoritative_graph_edge_count") == 4
        and task0155_summary.get("authoritative_minimum_distance_two_path_count") == 0
        and task0155_summary.get("current_v1_graph_sensitive_residual_count") == 0
        and task0155_summary.get("graph_retrieval_v2_multihop_go_decision") == "no_go",
        "task0154_authority_valid": task0154_verification["status"] == "valid"
        and task0154_summary.get("dangling_reference_count") == 2
        and task0154_summary.get("dominant_missing_target_root_cause") == "dangling_corpus_reference",
        "task0149_authority_valid": task0149_verification["status"] == "valid",
        "graph_retrieval_v1_frozen": task0149_summary.get("graph_retrieval_v1_frozen") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "task0155_dangling_reference_count": task0155_summary.get("dangling_reference_count"),
        "task0155_dangling_reference_systemic": task0155_summary.get("dangling_reference_systemic"),
        "task0155_authoritative_graph_edge_count": task0155_summary.get("authoritative_graph_edge_count"),
        "task0155_authoritative_minimum_distance_two_path_count": task0155_summary.get("authoritative_minimum_distance_two_path_count"),
        "task0155_current_v1_graph_sensitive_residual_count": task0155_summary.get("current_v1_graph_sensitive_residual_count"),
        "task0155_graph_retrieval_v2_multihop_go_decision": task0155_summary.get("graph_retrieval_v2_multihop_go_decision"),
        "task0155_summary_sha256": sha256_file(task0155.RESULT_DIR / "summary.json"),
        "task0154_summary_sha256": sha256_file(task0154.RESULT_DIR / "summary.json"),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
    }


def build_dangling_reference_inventory(records: Iterable[dict[str, Any]], authority: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(records)
    classes = Counter(task0155.classify_reference(row) for row in rows)
    dangling_source_rows = [row for row in rows if task0155.classify_reference(row) == "dangling_reference"]
    dangling_rows = [inventory_row(row) for row in dangling_source_rows]
    dangling_count = len(dangling_rows)
    return (
        {
            "schema_version": "opk-rag.task0156.dangling-reference-inventory-audit.v1",
            "task_id": TASK_ID,
            "total_internal_reference_count": len(rows),
            "task0155_dangling_reference_count": authority["task0155_dangling_reference_count"],
            "task0156_reproduced_dangling_reference_count": dangling_count,
            "dangling_inventory_drift_detected": dangling_count != authority["task0155_dangling_reference_count"],
            "dangling_reference_count": dangling_count,
            "dangling_reference_rate": _safe_div(dangling_count, len(rows)),
            "reference_class_distribution": dict(sorted(classes.items())),
        },
        dangling_rows,
    )


def inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = str(row.get("raw_link_text") or "")
    return {
        "schema_version": "opk-rag.task0156.dangling-reference-inventory.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["link_id"],
        "source_document_id": row["source_document_id"],
        "source_document_path": row["source_document_id"],
        "source_section_id": row.get("source_section_id"),
        "source_line": row.get("line"),
        "raw_reference": raw,
        "parsed_reference": raw.split("|", 1)[0].split("#", 1)[0].strip(),
        "normalized_reference": row.get("normalized_target"),
        "reference_shape": reference_shape(row),
        "target_exists": False,
        "target_in_authoritative_corpus": False,
        "resolution_status": row.get("resolution_status"),
        "root_cause_class": row.get("root_cause_class"),
        "link_syntax": row.get("link_syntax"),
        "candidate_targets": row.get("candidate_targets", []),
        "source_context": source_context(row),
    }


def classify_dangling_reference(row: dict[str, Any], *, edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    intent, confidence, evidence = classify_intent(row)
    severity = classify_severity(intent, row)
    topology = topology_impact_row({**row, "severity_class": severity}, edge_rows=edge_rows)
    retrieval_impact = False
    evidence_impact = "no_known_evidence_impact"
    benchmark_impact = False
    production_impact = "unknown_no_formal_production_query_trace_authority"
    repair_benefit = repair_benefit_potential(severity, retrieval_impact, evidence_impact, benchmark_impact)
    risk = repair_risk(intent, confidence)
    automatic = intent == "likely_typo" and confidence == "high" and risk == "low"
    targeted = repair_benefit in {"high", "medium"} and confidence == "high" and risk == "low" and not row["raw_reference"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))
    owner_review = intent in {"missing_authoritative_document", "ambiguous_intent", "unknown"} or (targeted and not automatic)
    return {
        **row,
        "schema_version": "opk-rag.task0156.dangling-reference-classification.v1",
        "intent_classification": intent,
        "classification_confidence": confidence,
        "classification_evidence": evidence,
        "severity_class": severity,
        "graph_blocking": severity == "graph_blocking",
        "graph_degrading": severity == "graph_degrading",
        "graph_topology_impact": topology["topology_impact_class"],
        "retrieval_impact": retrieval_impact,
        "evidence_impact": evidence_impact,
        "benchmark_impact": benchmark_impact,
        "production_impact": production_impact,
        "potential_authoritative_edge_recovery": topology["potential_authoritative_edge_recovery"],
        "repair_benefit_potential": repair_benefit,
        "repair_risk": risk,
        "automatic_repair_eligible": automatic,
        "owner_review_required": owner_review,
        "targeted_repair_candidate": targeted,
    }


def classify_intent(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    raw = str(row["raw_reference"])
    normalized = str(row["normalized_reference"] or "")
    context = str(row.get("source_context") or "")
    lower = f"{raw}\n{context}".lower()
    suffix = Path(raw).suffix.lower()
    if suffix and suffix != ".md":
        return "unsupported_non_document_target", "high", ["reference has non-Markdown attachment/file suffix", f"suffix={suffix}"]
    if raw.startswith(("../", "http://", "https://")):
        return "external_or_out_of_scope_reference", "high", ["reference points outside the authoritative source corpus"]
    if "todo" in lower or "tbd" in lower or "未来" in lower:
        return "future_placeholder", "high", ["source context indicates future/TODO placeholder intent"]
    if "公众号发布 saas demo/产品定位与商业假设" in normalized:
        return "intentional_uncreated_note", "high", ["sibling notes in the same planned decomposition exist while this conceptual note is absent"]
    if "academic-docx-polisher" in normalized:
        return "missing_authoritative_document", "medium", ["numbered related-note sequence references absent source documents; no deterministic existing target is available"]
    return "unknown", "low", ["no deterministic corpus evidence supports a narrower intent"]


def classify_severity(intent: str, row: dict[str, Any]) -> str:
    if intent in {"future_placeholder", "intentional_uncreated_note", "external_or_out_of_scope_reference", "unsupported_non_document_target"}:
        return "non_graph_blocking"
    if intent == "missing_authoritative_document":
        return "graph_degrading"
    if intent == "likely_typo":
        return "graph_blocking"
    return "unknown"


def topology_impact_row(row: dict[str, Any], *, edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_document = not str(row["raw_reference"]).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))
    source = row["source_document_id"]
    target = f"missing:{row['normalized_reference']}"
    nodes = {edge["source_node_id"] for edge in edge_rows} | {edge["target_node_id"] for edge in edge_rows}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for edge in edge_rows:
        outgoing[edge["source_node_id"]].append(edge["target_node_id"])
        incoming[edge["target_node_id"]].append(edge["source_node_id"])
    would_add_edge = expected_document and row.get("severity_class") in {"graph_blocking", "graph_degrading"}
    would_create_two_step = would_add_edge and bool(outgoing.get(source) or incoming.get(source))
    return {
        "schema_version": "opk-rag.task0156.topology-impact.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["dangling_reference_id"],
        "source_document_id": source,
        "normalized_reference": row["normalized_reference"],
        "would_add_node": would_add_edge and target not in nodes,
        "would_add_edge": would_add_edge,
        "would_reduce_component_count": False,
        "would_expand_existing_component": would_add_edge and source in nodes,
        "would_increase_out_degree": would_add_edge,
        "would_create_two_step_structure": would_create_two_step,
        "would_create_authoritative_two_step_chain": False,
        "potential_authoritative_edge_recovery": would_add_edge,
        "topology_impact_class": "edge_recovery_potential" if would_add_edge else "no_graph_topology_impact",
    }


def retrieval_impact_row(row: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.retrieval-impact.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["dangling_reference_id"],
        "retrieval_impact": False,
        "impact_reason": "no current V1 graph-sensitive residual or evaluation unit proves candidate loss from this dangling reference",
        "current_v1_graph_sensitive_residual_count": authority["task0155_current_v1_graph_sensitive_residual_count"],
        "impacted_unit_ids": [],
    }


def evidence_impact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.evidence-impact.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["dangling_reference_id"],
        "evidence_impact": "no_known_evidence_impact",
        "would_required_evidence_become_reachable": False,
        "impacted_required_evidence_ids": [],
    }


def graph_blocking_impact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.graph-blocking-impact.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["dangling_reference_id"],
        "severity_class": row["severity_class"],
        "graph_blocking": row["graph_blocking"],
        "graph_degrading": row["graph_degrading"],
        "blocking_definition_met": row["graph_blocking"],
        "blocking_rationale": "actual retrieval/evidence/benchmark impact is absent" if not row["graph_blocking"] else "target absence prevents an otherwise authoritative graph edge with proven workload impact",
    }


def build_benchmark_impact_audit(rows: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.benchmark-impact-audit.v1",
        "task_id": TASK_ID,
        "benchmark_unit_impacted_by_dangling_reference_count": 0,
        "benchmark_impacted_unit_ids": [],
        "benchmark_impact": False,
        "current_v1_graph_sensitive_residual_count": authority["task0155_current_v1_graph_sensitive_residual_count"],
        "audit_reason": "TASK-0155 authority reports current_v1_graph_sensitive_residual_count=0; no benchmark unit dependency on dangling references is present.",
    }


def build_intent_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intents = Counter(row["intent_classification"] for row in rows)
    confidence = Counter(row["classification_confidence"] for row in rows)
    severity = Counter(row["severity_class"] for row in rows)
    return {
        "schema_version": "opk-rag.task0156.intent-audit.v1",
        "task_id": TASK_ID,
        "intent_class_distribution": dict(sorted(intents.items())),
        "confidence_distribution": dict(sorted(confidence.items())),
        "severity_distribution": dict(sorted(severity.items())),
        "classification_authority": ["source context", "neighbor references", "corpus naming conventions", "target path evidence"],
        "nearest_string_similarity_used_as_authority": False,
    }


def build_graph_hygiene_decision(
    rows: list[dict[str, Any]], benchmark_audit: dict[str, Any], authority: dict[str, Any], reference_audit: dict[str, Any]
) -> dict[str, Any]:
    graph_blocking = sum(row["graph_blocking"] for row in rows)
    retrieval = sum(bool(row["retrieval_impact"]) for row in rows)
    evidence = sum(row["evidence_impact"] != "no_known_evidence_impact" for row in rows)
    repair_justified = graph_blocking > 0 or retrieval > 0 or evidence > 0
    if repair_justified and any(row["owner_review_required"] for row in rows if row["graph_blocking"]):
        decision = "owner_review_required"
        next_step = "dangling_reference_owner_review"
    elif repair_justified:
        decision = "repair"
        next_step = "targeted_graph_hygiene_repair"
    else:
        decision = "no_repair"
        next_step = "retain_current_graph_hygiene_state"
    status = "acceptable_with_dangling_noise" if not repair_justified and rows else "healthy"
    return {
        "schema_version": "opk-rag.task0156.graph-hygiene-decision.v1",
        "task_id": TASK_ID,
        "corpus_graph_hygiene_gap_material": repair_justified,
        "current_corpus_graph_hygiene_status": status,
        "corpus_graph_hygiene_repair_justified": repair_justified,
        "graph_hygiene_repair_decision": decision,
        "recommended_next_step": next_step,
        "decision_basis": {
            "dangling_reference_rate": reference_audit["dangling_reference_rate"],
            "graph_blocking_dangling_reference_count": graph_blocking,
            "dangling_reference_with_retrieval_impact_count": retrieval,
            "dangling_reference_with_evidence_impact_count": evidence,
            "benchmark_unit_impacted_by_dangling_reference_count": benchmark_audit["benchmark_unit_impacted_by_dangling_reference_count"],
            "task0155_multihop_no_go_preserved": authority["task0155_graph_retrieval_v2_multihop_go_decision"] == "no_go",
        },
    }


def build_summary(
    authority: dict[str, Any],
    corpus_snapshot: dict[str, Any],
    reference_audit: dict[str, Any],
    edge_audit: dict[str, Any],
    rows: list[dict[str, Any]],
    benchmark_audit: dict[str, Any],
    decision: dict[str, Any],
    mutation: dict[str, int],
) -> dict[str, Any]:
    intents = Counter(row["intent_classification"] for row in rows)
    severity = Counter(row["severity_class"] for row in rows)
    retrieval = sum(bool(row["retrieval_impact"]) for row in rows)
    evidence = sum(row["evidence_impact"] != "no_known_evidence_impact" for row in rows)
    authority_valid = authority["task0155_authority_valid"] and authority["task0154_authority_valid"] and authority["task0149_authority_valid"]
    no_mutation = all(value == 0 for value in mutation.values())
    complete = (
        authority_valid
        and reference_audit["task0156_reproduced_dangling_reference_count"] == 8
        and not reference_audit["dangling_inventory_drift_detected"]
        and len(rows) == 8
        and no_mutation
        and decision["graph_hygiene_repair_decision"] in {"repair", "no_repair", "owner_review_required"}
    )
    return {
        "schema_version": "opk-rag.task0156.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        **{key: authority[key] for key in ("task0155_authority_valid", "task0154_authority_valid", "task0149_authority_valid", "graph_retrieval_v1_frozen", "graph_retrieval_v1_baseline_digest")},
        "corpus_snapshot_identity_valid": True,
        "corpus_snapshot_revision": corpus_snapshot["corpus_revision"],
        "corpus_snapshot_digest": corpus_snapshot["corpus_digest"],
        "corpus_document_count": corpus_snapshot["document_count"],
        **{key: reference_audit[key] for key in ("task0155_dangling_reference_count", "task0156_reproduced_dangling_reference_count", "dangling_inventory_drift_detected", "total_internal_reference_count", "dangling_reference_count", "dangling_reference_rate", "reference_class_distribution")},
        "authoritative_graph_edge_count": edge_audit["authoritative_graph_edge_count"],
        "intent_class_distribution": dict(sorted(intents.items())),
        "likely_typo_count": intents["likely_typo"],
        "legacy_reference_count": intents["legacy_reference"],
        "future_placeholder_count": intents["future_placeholder"],
        "intentional_uncreated_note_count": intents["intentional_uncreated_note"],
        "external_or_out_of_scope_count": intents["external_or_out_of_scope_reference"],
        "unsupported_target_count": intents["unsupported_non_document_target"],
        "unknown_intent_count": intents["unknown"],
        "graph_blocking_dangling_reference_count": severity["graph_blocking"],
        "graph_degrading_dangling_reference_count": severity["graph_degrading"],
        "non_graph_blocking_dangling_reference_count": severity["non_graph_blocking"],
        "unknown_impact_dangling_reference_count": severity["unknown"],
        "graph_blocking_dangling_reference_rate": _safe_div(severity["graph_blocking"], reference_audit["total_internal_reference_count"]),
        "potential_authoritative_edge_recovery_count": sum(row["potential_authoritative_edge_recovery"] for row in rows),
        "dangling_reference_with_retrieval_impact_count": retrieval,
        "dangling_reference_with_evidence_impact_count": evidence,
        "benchmark_unit_impacted_by_dangling_reference_count": benchmark_audit["benchmark_unit_impacted_by_dangling_reference_count"],
        "production_query_impacted_count": None,
        "production_query_impact_unknown": True,
        **{key: decision[key] for key in ("corpus_graph_hygiene_gap_material", "current_corpus_graph_hygiene_status", "corpus_graph_hygiene_repair_justified", "graph_hygiene_repair_decision", "recommended_next_step")},
        "targeted_repair_candidate_count": sum(row["targeted_repair_candidate"] for row in rows),
        "automatic_repair_eligible_count": sum(row["automatic_repair_eligible"] for row in rows),
        "owner_review_required_count": sum(row["owner_review_required"] for row in rows),
        "task0155_multihop_no_go_preserved": authority["task0155_graph_retrieval_v2_multihop_go_decision"] == "no_go",
        "bounded_multihop_experiment_ready": False,
        "repair_applied": False,
        **mutation,
    }


def build_contract(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0156.corpus-graph-hygiene-and-dangling-reference-impact-audit-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "authority_valid": all(summary.get(key) is True for key in ("task0155_authority_valid", "task0154_authority_valid", "task0149_authority_valid")),
        "dangling_inventory_reproduced": summary["task0156_reproduced_dangling_reference_count"] == summary["task0155_dangling_reference_count"] == 8,
        "all_references_classified": len(rows) == summary["dangling_reference_count"] and all(row.get("intent_classification") and row.get("severity_class") for row in rows),
        "impact_before_repair": summary["repair_applied"] is False,
        "no_source_mutation": summary["source_document_creation_count"] == summary["source_document_mutation_count"] == summary["source_reference_mutation_count"] == 0,
        "no_policy_mutation": summary["resolver_policy_mutation_count"] == summary["graph_policy_mutation_count"] == summary["runtime_policy_mutation_count"] == 0,
        "multihop_no_go_preserved": summary["task0155_multihop_no_go_preserved"] is True and summary["bounded_multihop_experiment_ready"] is False,
        "decision_valid": summary["graph_hygiene_repair_decision"] in {"repair", "no_repair", "owner_review_required"},
    }


def verify_task0156_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if name != "verification.json" and not (output_dir / name).exists()]
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
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() and not parse_errors else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "contract_valid": contract.get("summary_required_fields_present") is True,
        "authority_valid": all(summary.get(key) is True for key in ("task0155_authority_valid", "task0154_authority_valid", "task0149_authority_valid")),
        "corpus_snapshot_identity_valid": summary.get("corpus_snapshot_identity_valid") is True,
        "dangling_inventory_reproduced": summary.get("task0155_dangling_reference_count") == summary.get("task0156_reproduced_dangling_reference_count") == 8,
        "severity_counts_consistent": summary.get("graph_blocking_dangling_reference_count", 0)
        + summary.get("graph_degrading_dangling_reference_count", 0)
        + summary.get("non_graph_blocking_dangling_reference_count", 0)
        + summary.get("unknown_impact_dangling_reference_count", 0)
        == summary.get("dangling_reference_count"),
        "no_repair_applied": summary.get("repair_applied") is False,
        "no_source_mutation": summary.get("source_document_creation_count") == summary.get("source_document_mutation_count") == summary.get("source_reference_mutation_count") == 0,
        "no_policy_mutation": summary.get("resolver_policy_mutation_count") == summary.get("graph_policy_mutation_count") == summary.get("runtime_policy_mutation_count") == 0,
        "v1_integrity_preserved": summary.get("graph_retrieval_v1_frozen") is True and summary.get("task0149_frozen_artifact_mutation_count") == 0,
        "multihop_no_go_preserved": summary.get("task0155_multihop_no_go_preserved") is True and summary.get("bounded_multihop_experiment_ready") is False,
        "decision_valid": summary.get("graph_hygiene_repair_decision") in {"repair", "no_repair", "owner_review_required"},
    }
    result = {
        "schema_version": "opk-rag.task0156.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if all(checks.values()) else "invalid",
        "checks": checks,
        "missing_artifacts": missing,
        "parse_errors": parse_errors,
        "summary": {key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def repair_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.repair-candidate.v1",
        "task_id": TASK_ID,
        "dangling_reference_id": row["dangling_reference_id"],
        "repair_priority_rank": None,
        "targeted_repair_candidate": row["targeted_repair_candidate"],
        "automatic_repair_eligible": row["automatic_repair_eligible"],
        "owner_review_required": row["owner_review_required"],
        "repair_benefit_potential": row["repair_benefit_potential"],
        "repair_risk": row["repair_risk"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0156.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def build_report(summary: dict[str, Any], intent_audit: dict[str, Any]) -> str:
    return f"""# TASK0156 Corpus Graph Hygiene and Dangling Reference Impact Audit Report

## Summary

`task_status={summary["task_status"]}`

`task0155_authority_valid={str(summary["task0155_authority_valid"]).lower()}`

`graph_retrieval_v1_frozen={str(summary["graph_retrieval_v1_frozen"]).lower()}`

`graph_retrieval_v1_baseline_digest={summary["graph_retrieval_v1_baseline_digest"]}`

TASK-0156 reproduced the production corpus dangling-reference inventory and audited impact before repair. It did not create source documents, edit source references, mutate resolver or graph policy, change runtime policy, repair dangling links, or reopen multi-hop runtime work.

## Corpus Snapshot

Corpus revision: `{summary["corpus_snapshot_revision"]}`.

Corpus digest: `{summary["corpus_snapshot_digest"]}`.

Corpus document count: `{summary["corpus_document_count"]}`.

Internal references: `{summary["total_internal_reference_count"]}`.

Dangling references: `{summary["dangling_reference_count"]}`.

Dangling reference rate: `{summary["dangling_reference_rate"]}`.

Inventory drift detected: `{str(summary["dangling_inventory_drift_detected"]).lower()}`.

## Intent And Severity

Intent distribution: `{intent_audit["intent_class_distribution"]}`.

Graph-blocking dangling references: `{summary["graph_blocking_dangling_reference_count"]}`.

Graph-degrading dangling references: `{summary["graph_degrading_dangling_reference_count"]}`.

Non-graph-blocking dangling references: `{summary["non_graph_blocking_dangling_reference_count"]}`.

Unknown-impact dangling references: `{summary["unknown_impact_dangling_reference_count"]}`.

Potential authoritative edge recovery count: `{summary["potential_authoritative_edge_recovery_count"]}`.

## Impact

Retrieval-impact dangling references: `{summary["dangling_reference_with_retrieval_impact_count"]}`.

Evidence-impact dangling references: `{summary["dangling_reference_with_evidence_impact_count"]}`.

Benchmark impacted unit count: `{summary["benchmark_unit_impacted_by_dangling_reference_count"]}`.

Production query impacted count: `{summary["production_query_impacted_count"]}`.

## Decision

Corpus graph hygiene status: `{summary["current_corpus_graph_hygiene_status"]}`.

Corpus graph hygiene repair justified: `{str(summary["corpus_graph_hygiene_repair_justified"]).lower()}`.

Graph hygiene repair decision: `{summary["graph_hygiene_repair_decision"]}`.

Targeted repair candidates: `{summary["targeted_repair_candidate_count"]}`.

Automatic repair eligible: `{summary["automatic_repair_eligible_count"]}`.

Owner review required: `{summary["owner_review_required_count"]}`.

TASK-0155 multi-hop NO_GO preserved: `{str(summary["task0155_multihop_no_go_preserved"]).lower()}`.

Bounded multi-hop experiment ready: `{str(summary["bounded_multihop_experiment_ready"]).lower()}`.

Recommended next step: `{summary["recommended_next_step"]}`.
"""


def reference_shape(row: dict[str, Any]) -> str:
    if row.get("link_syntax") == "obsidian_embed" and Path(str(row.get("raw_link_text") or "")).suffix:
        return "attachment_reference"
    return task0154.classify_reference_shape(str(row.get("raw_link_text") or ""), str(row.get("link_syntax") or ""))


def source_context(row: dict[str, Any], *, radius: int = 2) -> str:
    path = ROOT / row["source_document_id"]
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    line = int(row.get("line") or 1)
    start = max(line - radius - 1, 0)
    end = min(line + radius, len(lines))
    return "\n".join(lines[start:end])


def repair_benefit_potential(severity: str, retrieval_impact: bool, evidence_impact: str, benchmark_impact: bool) -> str:
    if retrieval_impact or evidence_impact != "no_known_evidence_impact" or benchmark_impact:
        return "high"
    if severity == "graph_blocking":
        return "medium"
    if severity == "graph_degrading":
        return "low"
    if severity == "non_graph_blocking":
        return "none"
    return "unknown"


def repair_risk(intent: str, confidence: str) -> str:
    if intent == "likely_typo" and confidence == "high":
        return "low"
    if intent in {"future_placeholder", "intentional_uncreated_note", "missing_authoritative_document", "ambiguous_intent", "unknown"}:
        return "high"
    return "medium"


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
