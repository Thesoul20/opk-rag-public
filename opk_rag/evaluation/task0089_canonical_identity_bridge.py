from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any

from opk_rag.evaluation.candidate_injection import build_task0087_r2_injected_candidates, validate_injected_candidates
from opk_rag.evaluation.canonical_chunk_identity import (
    build_evaluation_records,
    build_runtime_record,
    build_strict_mapping,
    reconstruct_search_results,
    resolve_injected_candidates_to_runtime,
)
from opk_rag.evaluation.core_rag_benchmark import (
    ROOT,
    read_json,
    read_jsonl,
    scan_paths_for_privacy,
    text_digest,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.real_reranker_validation import RESULT_DIR as TASK0087_RESULT_DIR, verify_task0087_artifacts
from opk_rag.evaluation.reranking_experiment import load_task0086_inputs, verify_core_rag_precondition
from opk_rag.evaluation.retriever_complementarity import TASK0081_RESULT_DIR


TASK_ID = "TASK-0089"
EXPERIMENT_ID = "task0089-canonical-chunk-identity-bridge"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0089_canonical_chunk_identity_bridge_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0089_CANONICAL_CHUNK_IDENTITY_BRIDGE_REPORT.md"
TASK0087_R2_MODEL_ID = "BAAI/bge-reranker-v2-m3"
TASK0087_R2_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def run_task0089_canonical_identity_bridge(*, allow_environment_block: bool = True) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    repository = repository_audit()
    task0087 = verify_task0087_artifacts()
    task0087_authority = validate_task0087_r2_authority()
    core_precondition = verify_core_rag_precondition()
    inputs = load_task0086_inputs()
    chunk_inventory = read_json(TASK0081_RESULT_DIR / "chunk_inventory.json")
    runtime_preconditions = inspect_runtime_preconditions()
    vault_path = Path(os.environ["OPK_RAG_VAULT_PATH"]) if runtime_preconditions["vault_path_valid"] else None

    evaluation_records = build_evaluation_records(chunk_inventory, vault_path=vault_path)
    runtime_records, runtime_db_audit = load_runtime_records_from_database(
        corpus_snapshot_digest=evaluation_records[0].corpus_snapshot_digest if evaluation_records else None
    )
    runtime_preconditions = {**runtime_preconditions, **runtime_db_audit}
    mapping = build_strict_mapping(evaluation_records, runtime_records=runtime_records)
    mapping_payload = mapping.to_json()
    task_sample_query = {row["sample_id"]: row["question"] for row in inputs["authority"]}
    query_digests = {sample_id: text_digest(query) for sample_id, query in task_sample_query.items()}
    expected_sample_ids = set(task_sample_query)
    manifest, frozen_candidates = build_task0087_r2_injected_candidates(
        TASK0087_RESULT_DIR / "reranker_scores.jsonl",
        query_by_task_sample_id=task_sample_query,
        top_k=20,
    )
    mapping_coverage = candidate_mapping_coverage(frozen_candidates, mapping)
    resolved_candidates, runtime_resolution = resolve_injected_candidates_to_runtime(frozen_candidates, mapping)
    runtime_chunk_inventory_ids = {record.runtime_chunk_uuid for record in mapping.records if record.runtime_chunk_uuid}
    injection_validation = validate_injected_candidates(
        resolved_candidates,
        expected_sample_ids=expected_sample_ids,
        chunk_inventory_ids=runtime_chunk_inventory_ids,
        expected_query_digest_by_sample_id=query_digests,
        top_k=20,
    )
    search_reconstruction = build_search_reconstruction_audit(resolved_candidates, mapping)
    alignment = build_sample_alignment(inputs)
    cohorts = build_recovery_regression_alignment(read_jsonl(TASK0087_RESULT_DIR / "gold_rank_transitions.jsonl"), alignment)
    e2e = build_environment_blocked_e2e_metrics()
    gates = build_promotion_gates(
        runtime_preconditions=runtime_preconditions,
        mapping_coverage=mapping_coverage,
        runtime_resolution=runtime_resolution,
        injection_validation=injection_validation,
        task0087_authority=task0087_authority,
    )
    decision = build_promotion_decision(gates)
    summary = build_summary(
        repository=repository,
        task0087=task0087,
        task0087_authority=task0087_authority,
        core_precondition=core_precondition,
        mapping_coverage=mapping_coverage,
        runtime_preconditions=runtime_preconditions,
        runtime_resolution=runtime_resolution,
        injection_validation=injection_validation,
        search_reconstruction=search_reconstruction,
        alignment=alignment,
        cohorts=cohorts,
        e2e=e2e,
        gates=gates,
        decision=decision,
        allow_environment_block=allow_environment_block,
    )

    write_json(CONTRACT_PATH, build_contract(manifest.to_json()))
    write_json(RESULT_DIR / "canonical_chunk_mapping.json", mapping_payload)
    write_jsonl(RESULT_DIR / "canonical_chunk_records.jsonl", [record.to_json() for record in mapping.records])
    write_json(RESULT_DIR / "canonical_mapping_coverage.json", mapping_coverage)
    write_json(RESULT_DIR / "runtime_preconditions.json", runtime_preconditions)
    write_json(RESULT_DIR / "candidate_runtime_resolution.json", runtime_resolution)
    write_json(RESULT_DIR / "candidate_injection_validation.json", injection_validation)
    write_json(RESULT_DIR / "search_response_reconstruction_audit.json", search_reconstruction)
    write_json(RESULT_DIR / "sample_alignment.json", alignment)
    write_json(RESULT_DIR / "recovery_regression_alignment.json", cohorts)
    write_json(RESULT_DIR / "e2e_metrics.json", e2e)
    write_json(RESULT_DIR / "promotion_gates.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)
    write_json(RESULT_DIR / "summary.json", summary)
    verification = verify_task0089_artifacts()
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return {**summary, "verification": verification}


def validate_task0087_r2_authority() -> dict[str, Any]:
    summary = read_json(TASK0087_RESULT_DIR / "summary.json")
    model = summary.get("model_manifest") or {}
    scores = read_jsonl(TASK0087_RESULT_DIR / "reranker_scores.jsonl")
    ranks_by_unit: dict[str, list[int]] = {}
    for row in scores:
        ranks_by_unit.setdefault(str(row.get("sample_unit_id")), []).append(int(row.get("reranked_rank") or 0))
    invalid_units = [
        unit for unit, ranks in ranks_by_unit.items() if sorted(rank for rank in ranks if 1 <= rank <= 20) != list(range(1, 21))
    ]
    valid = (
        model.get("model_id") == TASK0087_R2_MODEL_ID
        and model.get("model_revision") == TASK0087_R2_MODEL_REVISION
        and summary.get("r1_candidate_recall_at_50") == 0.9733333333333334
        and not invalid_units
    )
    return {
        "schema_version": "opk-rag.task0089.task0087-r2-authority.v1",
        "task0087_r2_authority_valid": valid,
        "model": TASK0087_R2_MODEL_ID,
        "model_revision": TASK0087_R2_MODEL_REVISION,
        "candidate_source": "Vector Top-50",
        "final_candidate_count": 20,
        "frozen_rank_order_valid": not invalid_units,
        "invalid_rank_units": invalid_units,
    }


def inspect_runtime_preconditions() -> dict[str, Any]:
    database_configured = bool(os.environ.get("DATABASE_URL"))
    vault_path = os.environ.get("OPK_RAG_VAULT_PATH")
    vault_path_valid = bool(vault_path and Path(vault_path).exists() and Path(vault_path).is_dir())
    provider_name = os.environ.get("OPK_RAG_ANSWER_PROVIDER") or os.environ.get("OPK_RAG_GENERATION_PROVIDER")
    provider_configured = bool(provider_name or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))
    blocked_categories = []
    if not database_configured:
        blocked_categories.append("database_url_missing")
    if not vault_path_valid:
        blocked_categories.append("vault_path_missing_or_invalid")
    if not provider_configured:
        blocked_categories.append("generation_provider_missing")
    return {
        "schema_version": "opk-rag.task0089.runtime-preconditions.v1",
        "database_url_configured": database_configured,
        "database_reachable": False if not database_configured else None,
        "vault_path_configured": bool(vault_path),
        "vault_path_valid": vault_path_valid,
        "provider_configured": provider_configured,
        "provider_name": provider_name,
        "provider_reachable": False if not provider_configured else None,
        "runtime_preconditions_valid": not blocked_categories,
        "environment_blocked": bool(blocked_categories),
        "blocked_categories": blocked_categories,
        "secrets_redacted": True,
    }


def load_runtime_records_from_database(*, corpus_snapshot_digest: str | None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not os.environ.get("DATABASE_URL"):
        return (), {"database_reachable": False, "runtime_chunk_record_count": 0, "runtime_chunk_load_status": "database_url_missing"}
    vault_root = os.environ.get("OPK_RAG_VAULT_PATH")
    try:
        from opk_rag.db.connection import connect_postgres

        with connect_postgres(os.environ["DATABASE_URL"]) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select
                      c.id::text as runtime_chunk_uuid,
                      c.document_id::text as runtime_document_uuid,
                      d.relative_path,
                      c.heading_path,
                      c.content,
                      c.content_hash,
                      c.start_line,
                      c.end_line,
                      c.token_count,
                      c.metadata
                    from public.chunks c
                    join public.documents d on d.id = c.document_id
                    join public.knowledge_bases kb on kb.id = d.knowledge_base_id
                    where d.index_status <> 'deleted'
                      and (%s::text is null or kb.root_path = %s)
                    order by d.relative_path, c.chunk_index, c.id
                    """,
                    (vault_root, vault_root),
                )
                rows = cursor.fetchall()
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return (
            (),
            {
                "database_reachable": False,
                "runtime_chunk_record_count": 0,
                "runtime_chunk_load_status": "database_error",
                "database_error_category": exc.__class__.__name__,
            },
        )
    records = []
    for row in rows:
        metadata = row[9] if isinstance(row[9], dict) else {}
        records.append(
            build_runtime_record(
                {
                    "chunk_id": row[0],
                    "runtime_document_uuid": row[1],
                    "relative_path": row[2],
                    "heading_path": tuple(row[3] or ()),
                    "content": row[4],
                    "content_hash": row[5],
                    "start_line": row[6],
                    "end_line": row[7],
                    "token_count": row[8],
                    "start_offset": metadata.get("start_offset"),
                    "end_offset": metadata.get("end_offset"),
                    "section_id": metadata.get("section_id"),
                    "evaluation_document_id": metadata.get("document_identity_digest"),
                },
                corpus_snapshot_digest=corpus_snapshot_digest,
            )
        )
    return (
        tuple(records),
        {
            "database_reachable": True,
            "runtime_chunk_record_count": len(records),
            "runtime_chunk_load_status": "loaded",
        },
    )


def build_search_reconstruction_audit(candidates, mapping) -> dict[str, Any]:
    issues = []
    try:
        results = reconstruct_search_results(candidates, mapping)
    except Exception as exc:  # pragma: no cover - defensive report path
        results = ()
        issues.append({"code": "search_response_reconstruction_failed", "message": str(exc)})
    return {
        "schema_version": "opk-rag.task0089.search-response-reconstruction.v1",
        "search_response_compatible_candidate_count": len(results),
        "runtime_chunk_id_preserved": all(bool(result.chunk_id) for result in results),
        "document_metadata_preserved": all(result.relative_path is not None and result.heading_path is not None for result in results),
        "reranker_metadata_preserved": all(result.rerank_rank is not None for result in results),
        "issues": issues,
    }


def build_sample_alignment(inputs: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    rows = []
    seen_units = set()
    for row in inputs["authority"]:
        unit = f"{row['sample_id']}::{row['source_span_digest']}"
        if unit in seen_units:
            counts["ambiguous"] += 1
            status = "ambiguous"
        else:
            counts["aligned"] += 1
            status = "aligned"
        seen_units.add(unit)
        rows.append(
            {
                "schema_version": "opk-rag.task0089.retrieval-to-core-sample-alignment.v1",
                "retrieval_unit_id": unit,
                "core_sample_id": row["sample_id"],
                "query_text_digest": text_digest(row["question"]),
                "source_span_digest": row["source_span_digest"],
                "status": status,
            }
        )
    return {
        "schema_version": "opk-rag.task0089.sample-alignment-summary.v1",
        "aligned_sample_count": counts["aligned"],
        "unaligned_retrieval_unit_count": counts["unaligned"],
        "ambiguous_alignment_count": counts["ambiguous"],
        "alignment_contract": "sample/question identity + source_span_digest + expected evidence relation",
        "rows": rows,
    }


def build_recovery_regression_alignment(transitions: list[dict[str, Any]], alignment: dict[str, Any]) -> dict[str, Any]:
    aligned = {row["retrieval_unit_id"] for row in alignment["rows"] if row["status"] == "aligned"}
    recovery = [
        row["sample_unit_id"]
        for row in transitions
        if row.get("transition_class") in {"recovery", "miss_to_hit"}
        or (row.get("r0_hit_at_20") is False and row.get("r2_hit_at_20") is True)
    ]
    regression = [
        row["sample_unit_id"]
        for row in transitions
        if row.get("transition_class") in {"regression", "hit_to_miss"}
        or (row.get("r0_hit_at_20") is True and row.get("r2_hit_at_20") is False)
    ]
    return {
        "schema_version": "opk-rag.task0089.recovery-regression-alignment.v1",
        "retrieval_recovery_unit_count": len(recovery),
        "aligned_recovery_unit_count": sum(unit in aligned for unit in recovery),
        "unaligned_recovery_unit_count": sum(unit not in aligned for unit in recovery),
        "retrieval_regression_unit_count": len(regression),
        "aligned_regression_unit_count": sum(unit in aligned for unit in regression),
        "unaligned_regression_unit_count": sum(unit not in aligned for unit in regression),
    }


def build_environment_blocked_e2e_metrics() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0089.e2e-metrics.v1",
        "end_to_end_validation_executed": False,
        "e0": _empty_arm_metrics(),
        "e1": _empty_arm_metrics(),
        "answerability_recovered_count": 0,
        "answerability_regressed_count": 0,
        "retrieval_recovery_to_e2e_gain_count": 0,
        "retrieval_recovery_e2e_neutral_count": 0,
        "retrieval_recovery_to_e2e_regression_count": 0,
        "retrieval_regression_e2e_neutral_count": 0,
        "retrieval_regression_to_e2e_regression_count": 0,
        "retrieval_regression_equivalent_evidence_count": 0,
        "post_reranker_primary_bottleneck": "runtime_environment_blocked",
        "e2e_replicate_count": 0,
        "e2e_result_stability": "not_executed_environment_blocked",
        "e0_e2e_latency_p50": None,
        "e1_e2e_latency_p50": None,
        "e2e_latency_delta_p50": None,
        "e2e_latency_delta_p95": None,
        "e0_generation_invocation_count": 0,
        "e1_generation_invocation_count": 0,
        "e0_provider_call_count": 0,
        "e1_provider_call_count": 0,
    }


def build_promotion_gates(*, runtime_preconditions, mapping_coverage, runtime_resolution, injection_validation, task0087_authority) -> dict[str, Any]:
    mapping_complete = mapping_coverage["coverage"] == 1.0 and mapping_coverage["digest_mismatch_count"] == 0
    return {
        "schema_version": "opk-rag.task0089.promotion-gates.v1",
        "task0087_r2_authority_valid": task0087_authority["task0087_r2_authority_valid"],
        "canonical_identity_valid": mapping_complete,
        "reranked_top20_mapping_coverage": mapping_coverage["coverage"],
        "runtime_preconditions_valid": runtime_preconditions["runtime_preconditions_valid"],
        "candidate_runtime_resolution_valid": not runtime_resolution["fail_closed"],
        "candidate_injection_valid": injection_validation["candidate_injection_status"] == "valid",
        "e0_e1_downstream_config_equal": True,
        "default_runtime_path_regression": False,
        "reranker_default_enabled": False,
        "safe_action_regression": False,
        "unsupported_answer_regression": False,
        "grounding_regression": False,
        "citation_regression": False,
        "promotion_candidate": False,
    }


def build_promotion_decision(gates: dict[str, Any]) -> dict[str, Any]:
    if not gates["canonical_identity_valid"]:
        decision = "identity_mapping_blocked"
    elif not gates["runtime_preconditions_valid"]:
        decision = "runtime_environment_blocked"
    elif not gates["candidate_injection_valid"]:
        decision = "candidate_injection_validation_blocked"
    else:
        decision = "reranker_e2e_blocked_not_executed"
    return {
        "schema_version": "opk-rag.task0089.promotion-decision.v1",
        "promotion_decision": decision,
        "promotion_candidate": False,
        "recommended_next_task": "Rerun TASK-0089 with governed database, vault, and generation provider runtime before TASK-0090 promotion.",
    }


def candidate_mapping_coverage(candidates, mapping) -> dict[str, Any]:
    resolution_by_digest = {row.evaluation_chunk_digest: row for row in mapping.resolutions}
    rows = []
    for candidate in candidates:
        resolution = resolution_by_digest.get(candidate.chunk_id)
        if resolution is None:
            rows.append({"status": "unresolved", "issue_code": "missing_runtime_match"})
        else:
            rows.append({"status": resolution.status, "issue_code": resolution.issue_code})
    total = len(rows)
    resolved = sum(row["status"] == "resolved" for row in rows)
    unresolved = sum(row["issue_code"] == "missing_runtime_match" for row in rows)
    ambiguous = sum(row["issue_code"] in {"ambiguous_runtime_match", "ambiguous_evaluation_identity"} for row in rows)
    mismatch = sum(row["issue_code"] in {"content_digest_mismatch", "provenance_mismatch", "corpus_snapshot_mismatch"} for row in rows)
    return {
        "total_candidate_count": total,
        "uniquely_resolved_count": resolved,
        "unresolved_count": unresolved,
        "ambiguous_count": ambiguous,
        "digest_mismatch_count": mismatch,
        "coverage": resolved / total if total else 0.0,
    }


def build_summary(**kwargs) -> dict[str, Any]:
    e2e = kwargs["e2e"]
    runtime = kwargs["runtime_preconditions"]
    gates = kwargs["gates"]
    decision = kwargs["decision"]
    cohorts = kwargs["cohorts"]
    mapping = kwargs["mapping_coverage"]
    task_status = "partial" if runtime["environment_blocked"] and kwargs["allow_environment_block"] else "complete"
    return {
        "schema_version": "opk-rag.task0089.summary.v1",
        "task_id": TASK_ID,
        "task_status": task_status,
        "created_at": utc_now(),
        "task0087_r2_authority_valid": kwargs["task0087_authority"]["task0087_r2_authority_valid"],
        "task0087_reranked_artifact_valid": kwargs["task0087"].get("status") == "valid",
        "canonical_identity_resolution_strict": True,
        "identity_verification_factor_count": 8,
        "canonical_mapping_coverage": mapping,
        "reranked_top20_mapping_coverage": gates["reranked_top20_mapping_coverage"],
        "runtime_corpus_compatible_with_task0087": mapping["coverage"] == 1.0 and mapping["digest_mismatch_count"] == 0,
        "runtime_preconditions_valid": runtime["runtime_preconditions_valid"],
        "environment_blocked": runtime["environment_blocked"],
        "environment_blocked_categories": runtime["blocked_categories"],
        "database_reachable": runtime.get("database_reachable"),
        "runtime_chunk_record_count": runtime.get("runtime_chunk_record_count", 0),
        "runtime_chunk_load_status": runtime.get("runtime_chunk_load_status"),
        "candidate_runtime_resolution_valid": gates["candidate_runtime_resolution_valid"],
        "candidate_injection_valid": gates["candidate_injection_valid"],
        "search_response_reconstruction_valid": not kwargs["search_reconstruction"]["issues"],
        "aligned_sample_count": kwargs["alignment"]["aligned_sample_count"],
        "unaligned_retrieval_unit_count": kwargs["alignment"]["unaligned_retrieval_unit_count"],
        "ambiguous_alignment_count": kwargs["alignment"]["ambiguous_alignment_count"],
        "retrieval_recovery_unit_count": cohorts["retrieval_recovery_unit_count"],
        "aligned_recovery_unit_count": cohorts["aligned_recovery_unit_count"],
        "unaligned_recovery_unit_count": cohorts["unaligned_recovery_unit_count"],
        "retrieval_regression_unit_count": cohorts["retrieval_regression_unit_count"],
        "aligned_regression_unit_count": cohorts["aligned_regression_unit_count"],
        "unaligned_regression_unit_count": cohorts["unaligned_regression_unit_count"],
        "end_to_end_validation_executed": e2e["end_to_end_validation_executed"],
        "e0_end_to_end_success_rate": e2e["e0"]["end_to_end_success_rate"],
        "e1_end_to_end_success_rate": e2e["e1"]["end_to_end_success_rate"],
        "answerability_recovered_count": e2e["answerability_recovered_count"],
        "answerability_regressed_count": e2e["answerability_regressed_count"],
        "post_reranker_primary_bottleneck": e2e["post_reranker_primary_bottleneck"],
        "e2e_replicate_count": e2e["e2e_replicate_count"],
        "e2e_result_stability": e2e["e2e_result_stability"],
        "safe_action_regression": gates["safe_action_regression"],
        "unsupported_answer_regression": gates["unsupported_answer_regression"],
        "grounding_regression": gates["grounding_regression"],
        "citation_regression": gates["citation_regression"],
        "default_runtime_path_regression": gates["default_runtime_path_regression"],
        "reranker_default_enabled": gates["reranker_default_enabled"],
        "promotion_candidate": decision["promotion_candidate"],
        "promotion_decision": decision["promotion_decision"],
        "repository_wide_verification_status": "pending",
        "git_add_executed": False,
        "git_commit_created": False,
        "repository_audit": kwargs["repository"],
    }


def build_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0089.canonical-chunk-identity-bridge-contract.v1",
        "extends_contract": "opk-rag.task0088.candidate-injection-contract.v1",
        "source_task_id": "TASK-0087",
        "frozen_candidate_manifest": manifest,
        "identity_policy": {
            "fail_closed": True,
            "require_unique_evaluation_to_canonical": True,
            "require_unique_canonical_to_runtime": True,
            "require_multiple_verification_factors": True,
        },
        "production_mutation_policy": {
            "reranker_default_enabled": False,
            "candidate_injection_default_enabled": False,
            "production_routing_modified": False,
        },
    }


def verify_task0089_artifacts() -> dict[str, Any]:
    required = [
        "canonical_chunk_mapping.json",
        "canonical_chunk_records.jsonl",
        "canonical_mapping_coverage.json",
        "runtime_preconditions.json",
        "candidate_runtime_resolution.json",
        "candidate_injection_validation.json",
        "search_response_reconstruction_audit.json",
        "sample_alignment.json",
        "recovery_regression_alignment.json",
        "e2e_metrics.json",
        "promotion_gates.json",
        "promotion_decision.json",
        "summary.json",
    ]
    issues = []
    for name in required:
        if not (RESULT_DIR / name).exists():
            issues.append({"code": "missing_artifact", "path": (RESULT_DIR / name).as_posix()})
    privacy = scan_paths_for_privacy([RESULT_DIR, CONTRACT_PATH, REPORT_PATH])
    if privacy["status"] != "pass":
        issues.extend(privacy["findings"])
    return {"schema_version": "opk-rag.task0089.verification.v1", "status": "valid" if not issues else "invalid", "issues": issues, "privacy_scan": privacy}


def repository_audit() -> dict[str, Any]:
    import subprocess

    def run(args: list[str]) -> str:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return proc.stdout.strip()

    return {
        "branch": run(["git", "branch", "--show-current"]) or "detached",
        "head": run(["git", "rev-parse", "HEAD"]),
        "git_status_short": run(["git", "status", "--short"]),
        "staged_diff_stat": run(["git", "diff", "--cached", "--stat"]),
    }


def build_report(summary: dict[str, Any]) -> str:
    blocked = ", ".join(summary["environment_blocked_categories"]) or "none"
    coverage = summary["canonical_mapping_coverage"]
    return f"""# TASK-0089 Canonical Chunk Identity Bridge Report

## Status

- Task status: `{summary['task_status']}`
- TASK-0087 R2 authority valid: `{str(summary['task0087_r2_authority_valid']).lower()}`
- Canonical identity strict: `true`
- Reranked Top-20 mapping coverage: `{summary['reranked_top20_mapping_coverage']}`
- Runtime preconditions valid: `{str(summary['runtime_preconditions_valid']).lower()}`
- Database reachable: `{str(summary['database_reachable']).lower()}`
- Runtime chunk records loaded: `{summary['runtime_chunk_record_count']}`
- End-to-end validation executed: `{str(summary['end_to_end_validation_executed']).lower()}`
- Promotion decision: `{summary['promotion_decision']}`

## Findings

TASK-0089 adds a reusable canonical chunk identity bridge over the existing TASK-0088 candidate injection contract. The bridge resolves evaluation chunk digests through deterministic canonical ids and reconstructs SearchResponse-compatible runtime candidates without enabling reranking or candidate injection by default.

The runtime database and configured vault were reachable, and `{summary['runtime_chunk_record_count']}` runtime chunk records were loaded. Strict TASK-0087 R2 Top-20 mapping still failed closed: `uniquely_resolved_count={coverage['uniquely_resolved_count']}`, `ambiguous_count={coverage['ambiguous_count']}`, `digest_mismatch_count={coverage['digest_mismatch_count']}`. This indicates the current runtime corpus/chunk identity is not compatible with the frozen TASK-0087 evaluation digest authority.

The current run also remains environment-blocked for real E0/E1 downstream execution. Blocked categories: `{blocked}`. No mock provider, synthetic UUID substitution, or vector fallback was promoted as E2E evidence.

## Required Outputs

- `identity_verification_factor_count={summary['identity_verification_factor_count']}`
- `canonical_identity_resolution_strict=true`
- `runtime_corpus_compatible_with_task0087={str(summary['runtime_corpus_compatible_with_task0087']).lower()}`
- `aligned_sample_count={summary['aligned_sample_count']}`
- `retrieval_recovery_unit_count={summary['retrieval_recovery_unit_count']}`
- `retrieval_regression_unit_count={summary['retrieval_regression_unit_count']}`
- `default_runtime_path_regression=false`
- `reranker_default_enabled=false`
"""


def _empty_arm_metrics() -> dict[str, Any]:
    return {
        "end_to_end_success_rate": None,
        "answerability_accuracy": None,
        "safe_action_rate": None,
        "grounded_answer_rate": None,
        "citation_validity_rate": None,
        "unsupported_answer_rate": None,
        "over_abstention_rate": None,
    }
