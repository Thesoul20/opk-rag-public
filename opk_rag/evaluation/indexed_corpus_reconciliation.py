from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from psycopg.types.json import Jsonb

from opk_rag.chunking.chunker import chunk_markdown_document
from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.parser import parse_markdown_file
from opk_rag.db.connection import connect_postgres
from opk_rag.evaluation.corpus_snapshot import (
    PRIVATE_MANIFEST_PATH,
    PUBLIC_SNAPSHOT_PATH,
    SNAPSHOT_ID,
    load_private_manifest,
    resolve_reference_vault_path,
    sha256_text,
)
from opk_rag.evaluation.phase2_governance import (
    BASELINE_CONTRACT_PATH,
    stable_hash,
    utc_now,
)
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.vault import normalize_vault_root

PLAN_SCHEMA_VERSION = "opk-rag.index-reconciliation-plan.v1"
PLAN_ID = "task0049-phase2-index-repair-v1"
PLAN_PATH = Path(".private/evaluation/task0049_index_reconciliation_plan.json")
BACKUP_PATH = Path(".private/evaluation/task0049_index_repair_backup.json")
ACTIVE_DOCUMENT_STATUS = "indexed"
NON_ACTIVE_STATUSES = {"failed", "pending", "processing", "deleted"}
EXPECTED_EMBEDDING_DIMENSION = 1024

DRIFT_CONTRACTS: dict[str, dict[str, Any]] = {
    "missing_document": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "reindex_document",
        "requires_manual_confirmation": True,
        "rollback": "Restore document and chunks from backup or rerun single-document indexing.",
        "postcondition": "Frozen source path has one active indexed document.",
    },
    "unexpected_document": {
        "risk_level": "destructive",
        "auto_repairable": True,
        "recommended_action": "delete_unexpected_document",
        "requires_manual_confirmation": True,
        "rollback": "Reinsert backed up document and associated chunks.",
        "postcondition": "Document is absent from the current knowledge base index.",
    },
    "duplicate_document": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "manual_review",
        "requires_manual_confirmation": True,
        "rollback": "No automatic operation is generated.",
        "postcondition": "Only one indexed identity remains for the source path.",
    },
    "stale_document": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "reindex_document",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up document and chunks.",
        "postcondition": "Document digest and size match the frozen source manifest.",
    },
    "non_active_document": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "reindex_or_remove_residual",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up status.",
        "postcondition": "Frozen source documents are active indexed; residual deleted documents are removed.",
    },
    "failed_document": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "reindex_document",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up status and last_error.",
        "postcondition": "Document is indexed or explicitly removed if it is not in source.",
    },
    "pending_document": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "complete_incremental_index",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up status.",
        "postcondition": "Document is indexed.",
    },
    "processing_document": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "retry_or_mark_failed",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up status.",
        "postcondition": "Document is no longer processing.",
    },
    "deleted_document_residual": {
        "risk_level": "destructive",
        "auto_repairable": True,
        "recommended_action": "delete_deleted_document_residual",
        "requires_manual_confirmation": True,
        "rollback": "Reinsert backed up document and associated chunks.",
        "postcondition": "Deleted residual document and chunks are absent.",
    },
    "orphan_chunk": {
        "risk_level": "destructive",
        "auto_repairable": True,
        "recommended_action": "delete_orphan_chunks",
        "requires_manual_confirmation": True,
        "rollback": "Reinsert backed up chunks if necessary.",
        "postcondition": "No chunk references a missing document.",
    },
    "chunk_count_anomaly": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "manual_chunk_boundary_audit",
        "requires_manual_confirmation": True,
        "rollback": "No automatic operation is generated.",
        "postcondition": "Recomputed chunk boundaries match stored chunks.",
    },
    "chunk_boundary_metadata_mismatch": {
        "risk_level": "repairable",
        "auto_repairable": True,
        "recommended_action": "update_chunk_boundary_metadata",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up chunk boundary metadata.",
        "postcondition": "Recomputed chunk boundary metadata matches stored chunks.",
    },
    "content_digest_mismatch": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "reindex_document",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up document and chunks.",
        "postcondition": "Content digest matches frozen manifest.",
    },
    "path_identity_mismatch": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "manual_path_identity_review",
        "requires_manual_confirmation": True,
        "rollback": "No automatic operation is generated.",
        "postcondition": "Path digest maps to the expected source identity.",
    },
    "knowledge_base_mismatch": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "manual_knowledge_base_review",
        "requires_manual_confirmation": True,
        "rollback": "No automatic operation is generated.",
        "postcondition": "All documents belong to the reference knowledge base.",
    },
    "embedding_missing": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "embed_document_chunks",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up chunks.",
        "postcondition": "Indexed chunks have embeddings.",
    },
    "embedding_dimension_mismatch": {
        "risk_level": "repairable",
        "auto_repairable": False,
        "recommended_action": "regenerate_embedding",
        "requires_manual_confirmation": True,
        "rollback": "Restore backed up chunks.",
        "postcondition": "All chunk embeddings have dimension 1024.",
    },
    "unverifiable": {
        "risk_level": "blocked",
        "auto_repairable": False,
        "recommended_action": "collect_more_evidence",
        "requires_manual_confirmation": True,
        "rollback": "No automatic operation is generated.",
        "postcondition": "Missing evidence is available.",
    },
}


@dataclass(frozen=True)
class DocumentAuditRow:
    id: str
    knowledge_base_id: str
    relative_path: str
    content_hash: str
    file_size: int
    index_status: str
    parser_version: str
    chunking_version: str
    chunk_count: int = 0
    embedding_missing_count: int = 0
    embedding_dimension_violation_count: int = 0

    @property
    def path_digest(self) -> str:
        return sha256_text(self.relative_path)

    def redacted(self) -> dict[str, Any]:
        return {
            "document_id": self.id,
            "knowledge_base_id": self.knowledge_base_id,
            "path_digest": self.path_digest,
            "content_digest": self.content_hash,
            "file_size": self.file_size,
            "index_status": self.index_status,
            "chunk_count": self.chunk_count,
            "embedding_missing_count": self.embedding_missing_count,
            "embedding_dimension_violation_count": self.embedding_dimension_violation_count,
        }


def plan_digest(plan: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in plan.items() if key != "plan_digest"}
    return stable_hash(clean)


def source_rows_from_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["path"]): dict(item)
        for item in manifest.get("source_files", [])
        if isinstance(item, dict) and item.get("path")
    }


def classify_document_drift(
    source: Mapping[str, Mapping[str, Any]],
    documents: Sequence[DocumentAuditRow],
    *,
    orphan_chunk_count: int = 0,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    docs_by_path: dict[str, list[DocumentAuditRow]] = {}
    for document in documents:
        docs_by_path.setdefault(document.relative_path, []).append(document)

    for path, source_item in sorted(source.items()):
        matching = docs_by_path.get(path, [])
        if not matching:
            issues.append(_issue("missing_document", path_digest=sha256_text(path)))
            continue
        if len(matching) > 1:
            for document in matching:
                issues.append(_issue("duplicate_document", document=document))
        for document in matching:
            if document.index_status != ACTIVE_DOCUMENT_STATUS:
                status_type = f"{document.index_status}_document" if document.index_status in {"failed", "pending", "processing"} else "non_active_document"
                issues.append(_issue(status_type, document=document))
            if document.content_hash != source_item.get("content_sha256") or int(document.file_size) != int(source_item.get("size_bytes", -1)):
                issues.append(_issue("stale_document", document=document))
            source_size = int(source_item.get("size_bytes", 0))
            if source_size > 0 and document.chunk_count <= 0 and document.index_status == ACTIVE_DOCUMENT_STATUS:
                issues.append(_issue("chunk_count_anomaly", document=document))
            if document.embedding_missing_count:
                issues.append(_issue("embedding_missing", document=document))
            if document.embedding_dimension_violation_count:
                issues.append(_issue("embedding_dimension_mismatch", document=document))

    for path in sorted(set(docs_by_path) - set(source)):
        for document in docs_by_path[path]:
            drift_type = "deleted_document_residual" if document.index_status == "deleted" else "unexpected_document"
            issues.append(_issue(drift_type, document=document))
            if document.index_status in NON_ACTIVE_STATUSES:
                issues.append(_issue("non_active_document", document=document))

    if orphan_chunk_count:
        issues.append(_issue("orphan_chunk", count=orphan_chunk_count))
    return issues


def _issue(
    drift_type: str,
    *,
    document: DocumentAuditRow | None = None,
    path_digest: str | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    contract = DRIFT_CONTRACTS[drift_type]
    payload = {
        "drift_type": drift_type,
        "risk_level": contract["risk_level"],
        "repairable": bool(contract["auto_repairable"]),
        "recommended_action": contract["recommended_action"],
        "requires_manual_confirmation": bool(contract["requires_manual_confirmation"]),
        "rollback": contract["rollback"],
        "postcondition": contract["postcondition"],
    }
    if document is not None:
        payload.update(document.redacted())
    if path_digest is not None:
        payload["path_digest"] = path_digest
    if count is not None:
        payload["count"] = count
    return payload


def inspect_document_chunks(
    vault_path: str | Path,
    source: Mapping[str, Mapping[str, Any]],
    documents: Sequence[DocumentAuditRow],
    chunks_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    root = normalize_vault_root(vault_path)
    checked = 0
    matching = 0
    mismatches: list[dict[str, Any]] = []
    chunks_expected = 0
    chunks_indexed = 0
    doc_by_path = {document.relative_path: document for document in documents}
    for path, source_item in sorted(source.items()):
        document = doc_by_path.get(path)
        if document is None or document.index_status != ACTIVE_DOCUMENT_STATUS:
            continue
        checked += 1
        parsed = parse_markdown_file(root.path / path)
        expected = chunk_markdown_document(parsed, config)
        actual = list(chunks_by_document.get(document.id, ()))
        chunks_expected += len(expected)
        chunks_indexed += len(actual)
        expected_rows = [
            {
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "heading_path": list(chunk.heading_path),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            for chunk in expected
        ]
        actual_rows = [
            {
                "chunk_index": int(chunk.get("chunk_index", -1)),
                "content_hash": chunk.get("content_hash"),
                "heading_path": list(chunk.get("heading_path") or []),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
            }
            for chunk in actual
        ]
        if expected_rows == actual_rows:
            matching += 1
        else:
            diffs = []
            for expected_row, actual_row in zip(expected_rows, actual_rows):
                changed_fields = [
                    field
                    for field in ("content_hash", "heading_path", "start_line", "end_line")
                    if expected_row.get(field) != actual_row.get(field)
                ]
                if changed_fields:
                    diffs.append(
                        {
                            "chunk_index": expected_row["chunk_index"],
                            "fields": changed_fields,
                            "expected": {
                                "content_hash": expected_row["content_hash"],
                                "heading_path_digest": sha256_text(json.dumps(expected_row["heading_path"], ensure_ascii=False)),
                                "start_line": expected_row["start_line"],
                                "end_line": expected_row["end_line"],
                            },
                            "actual": {
                                "content_hash": actual_row["content_hash"],
                                "heading_path_digest": sha256_text(json.dumps(actual_row["heading_path"], ensure_ascii=False)),
                                "start_line": actual_row["start_line"],
                                "end_line": actual_row["end_line"],
                            },
                        }
                    )
            mismatches.append(
                {
                    "document_id": document.id,
                    "path_digest": source_item.get("path_digest") or sha256_text(path),
                    "expected_chunk_count": len(expected_rows),
                    "indexed_chunk_count": len(actual_rows),
                    "diffs": diffs,
                }
            )
    status = "verified" if checked and not mismatches else ("drift" if mismatches else "unavailable")
    return {
        "status": status,
        "documents_checked": checked,
        "documents_matching": matching,
        "documents_mismatching": len(mismatches),
        "chunks_expected": chunks_expected,
        "chunks_indexed": chunks_indexed,
        "mismatches": mismatches,
    }


def inspect_indexed_corpus(
    *,
    vault_path: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    database_url: str | None = None,
    include_chunk_boundary: bool = True,
) -> dict[str, Any]:
    load_project_env(Path(__file__).resolve().parents[2])
    database_url = database_url or os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return {"status": "unavailable", "reason": "DATABASE_URL is not configured.", "drift_summary": {}}
    manifest = dict(manifest or load_private_manifest())
    source = source_rows_from_manifest(manifest)
    root = normalize_vault_root(resolve_reference_vault_path(vault_path=vault_path))

    try:
        with connect_postgres(database_url) as connection:
            db = _read_remote_index_state(connection, root.canonical_path)
    except Exception as exc:
        return {"status": "unavailable", "reason": _sanitize(str(exc)), "drift_summary": {}}
    if db.get("knowledge_base_id") is None:
        return {"status": "unavailable", "reason": "Knowledge base for current Vault was not found.", "drift_summary": {}}

    documents = db["documents"]
    issues = classify_document_drift(source, documents, orphan_chunk_count=db["orphan_chunk_count"])
    chunk_boundary = (
        inspect_document_chunks(root.path, source, documents, db["chunks_by_document"])
        if include_chunk_boundary
        else {"status": "partial", "reason": "chunk boundary parity was not requested"}
    )
    drift_summary = _summarize_issues(issues)
    boundary_repair_issues = _chunk_boundary_repair_issues(chunk_boundary)
    issues.extend(boundary_repair_issues)
    drift_summary = _summarize_issues(issues)
    status = "verified" if _is_verified(issues, chunk_boundary) else "partial"
    indexed_digest = stable_hash([row.redacted() for row in sorted(documents, key=lambda item: item.relative_path)])
    active_documents = [row for row in documents if row.index_status == ACTIVE_DOCUMENT_STATUS]
    return {
        "schema_version": "opk-rag.indexed-corpus-audit.v1",
        "status": status,
        "audited_at": utc_now(),
        "source_snapshot_id": manifest.get("snapshot_id"),
        "source_markdown_files": len(source),
        "knowledge_base": {
            "id": db["knowledge_base_id"],
            "root_path_digest": sha256_text(root.canonical_path),
            "name_digest": sha256_text(str(db.get("knowledge_base_name") or "")),
        },
        "document_count": len(documents),
        "active_document_count": len(active_documents),
        "non_active_document_count": len(documents) - len(active_documents),
        "chunk_count": db["chunk_count"],
        "index_status_counts": db["index_status_counts"],
        "missing_source_documents": [issue for issue in issues if issue["drift_type"] == "missing_document"],
        "unexpected_indexed_documents": [issue for issue in issues if issue["drift_type"] == "unexpected_document"],
        "stale_documents": [issue for issue in issues if issue["drift_type"] == "stale_document"],
        "failed_or_non_active_documents": [
            issue
            for issue in issues
            if issue["drift_type"] in {"failed_document", "pending_document", "processing_document", "non_active_document"}
        ],
        "deleted_document_residuals": [issue for issue in issues if issue["drift_type"] == "deleted_document_residual"],
        "duplicate_documents": [issue for issue in issues if issue["drift_type"] == "duplicate_document"],
        "orphan_chunk_count": db["orphan_chunk_count"],
        "invalid_document_references": db["orphan_chunk_count"],
        "embedding_missing_count": db["embedding_missing_count"],
        "embedding_dimension_violations": db["embedding_dimension_violations"],
        "embedding_contract_parity": db["embedding_contract_parity"],
        "embedding_presence_parity": "verified" if db["embedding_missing_count"] == 0 else "drift",
        "embedding_dimension_parity": "verified" if db["embedding_dimension_violations"] == 0 else "drift",
        "embedding_value_parity": "not_required",
        "chunk_boundary_parity": chunk_boundary,
        "index_configuration": db["index_configuration"],
        "recent_index_run": db["recent_index_run"],
        "unresolved_index_failures": db["unresolved_index_failures"],
        "indexed_document_digest": indexed_digest,
        "drift_summary": drift_summary,
        "drift_issues": issues,
        "retrieval_visibility": {
            "unexpected_documents_visible": len(
                [issue for issue in issues if issue["drift_type"] == "unexpected_document" and issue.get("index_status") == ACTIVE_DOCUMENT_STATUS]
            ),
            "non_active_documents_visible": 0,
        },
    }


def build_reconciliation_plan(audit: Mapping[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for issue in audit.get("drift_issues", []):
        drift_type = issue.get("drift_type")
        action = None
        if drift_type == "deleted_document_residual":
            action = "hard_delete_document"
        elif drift_type == "unexpected_document" and issue.get("index_status") == "indexed":
            action = "hard_delete_document"
        elif drift_type == "orphan_chunk":
            action = "delete_orphan_chunks"
        elif drift_type == "chunk_boundary_metadata_mismatch":
            action = "update_chunk_boundary_metadata"
        if action is None:
            continue
        target_document_id = issue.get("document_id")
        op_id_source = f"{drift_type}:{target_document_id or issue.get('count')}"
        operations.append(
            {
                "operation_id": hashlib.sha256(op_id_source.encode("utf-8")).hexdigest()[:16],
                "drift_type": drift_type,
                "target_document_id": target_document_id,
                "target_path_digest": issue.get("path_digest"),
                "target_chunk_updates": issue.get("chunk_updates", []),
                "current_state": {k: issue.get(k) for k in ("index_status", "chunk_count", "content_digest", "file_size") if k in issue},
                "proposed_action": action,
                "reason_code": drift_type,
                "risk_level": DRIFT_CONTRACTS[str(drift_type)]["risk_level"],
                "preconditions": _preconditions_for_issue(issue),
                "expected_postconditions": _postconditions_for_issue(issue),
                "rollback": {"method": DRIFT_CONTRACTS[str(drift_type)]["rollback"], "backup_required": True},
            }
        )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "source_snapshot_id": audit.get("source_snapshot_id", SNAPSHOT_ID),
        "knowledge_base_id": audit.get("knowledge_base", {}).get("id"),
        "created_at": utc_now(),
        "dry_run": dry_run,
        "audit_status": audit.get("status"),
        "operations": operations,
    }
    plan["plan_digest"] = plan_digest(plan)
    return plan


def validate_reconciliation_plan(plan: Mapping[str, Any], audit: Mapping[str, Any] | None = None) -> tuple[bool, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        issues.append({"code": "plan_schema", "message": "Invalid reconciliation plan schema."})
    if plan.get("plan_id") != PLAN_ID:
        issues.append({"code": "plan_id", "message": "Unexpected reconciliation plan id."})
    expected_digest = plan_digest(plan)
    if plan.get("plan_digest") != expected_digest:
        issues.append({"code": "plan_digest_mismatch", "message": "Plan digest does not match plan content."})
    for operation in plan.get("operations", []):
        if not operation.get("preconditions"):
            issues.append({"code": "missing_preconditions", "message": "Plan operation is missing preconditions."})
        if operation.get("risk_level") == "destructive" and not operation.get("rollback", {}).get("backup_required"):
            issues.append({"code": "backup_required", "message": "Destructive operation requires a backup."})
    if audit is not None and plan.get("knowledge_base_id") != audit.get("knowledge_base", {}).get("id"):
        issues.append({"code": "knowledge_base_changed", "message": "Plan knowledge base does not match current audit."})
    return not issues, issues


def write_plan(plan: Mapping[str, Any], path: Path = PLAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_reconciliation_plan(
    plan: Mapping[str, Any],
    *,
    database_url: str | None = None,
    confirm: bool = False,
    backup_path: Path = BACKUP_PATH,
) -> dict[str, Any]:
    if not confirm:
        return {"status": "rejected", "reason": "confirm_required", "applied": 0, "skipped": 0, "failed": 0, "rolled_back": 0}
    valid, issues = validate_reconciliation_plan(plan)
    if not valid:
        return {"status": "rejected", "reason": "invalid_plan", "issues": issues, "applied": 0, "skipped": 0, "failed": 0, "rolled_back": 0}
    load_project_env(Path(__file__).resolve().parents[2])
    database_url = database_url or os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return {"status": "rejected", "reason": "database_unavailable", "applied": 0, "skipped": 0, "failed": 0, "rolled_back": 0}
    backup: dict[str, Any] = {
        "schema_version": "opk-rag.index-repair-backup.v1",
        "plan_id": plan.get("plan_id"),
        "plan_digest": plan.get("plan_digest"),
        "created_at": utc_now(),
        "contains_chunk_content": False,
        "documents": [],
        "chunks": [],
    }
    applied = skipped = failed = rolled_back = affected_chunks = 0
    try:
        with connect_postgres(database_url) as connection:
            with connection.transaction():
                for operation in plan.get("operations", []):
                    action = operation.get("proposed_action")
                    precondition_issue = _check_operation_preconditions(connection, operation)
                    if precondition_issue is not None:
                        raise RuntimeError(precondition_issue)
                    if action == "hard_delete_document":
                        document_id = operation.get("target_document_id")
                        if not document_id:
                            skipped += 1
                            continue
                        state = _backup_document(connection, document_id)
                        if state is None:
                            skipped += 1
                            continue
                        backup["documents"].append(state["document"])
                        backup["chunks"].extend(state["chunks"])
                        affected_chunks += len(state["chunks"])
                        with connection.cursor() as cursor:
                            cursor.execute("delete from public.documents where id = %s", (document_id,))
                            applied += cursor.rowcount
                    elif action == "delete_orphan_chunks":
                        orphan_backup = _backup_orphan_chunks(connection)
                        backup["chunks"].extend(orphan_backup)
                        affected_chunks += len(orphan_backup)
                        with connection.cursor() as cursor:
                            cursor.execute(
                                """
                                delete from public.chunks c
                                where not exists (select 1 from public.documents d where d.id = c.document_id)
                                """
                            )
                            applied += cursor.rowcount
                    elif action == "update_chunk_boundary_metadata":
                        updates = list(operation.get("target_chunk_updates") or [])
                        if not updates:
                            skipped += 1
                            continue
                        backup["chunks"].extend(_backup_chunks_for_boundary_updates(connection, operation.get("target_document_id"), updates))
                        affected_chunks += len(updates)
                        with connection.cursor() as cursor:
                            for update in updates:
                                cursor.execute(
                                    """
                                    update public.chunks
                                    set start_line = %s,
                                        end_line = %s
                                    where document_id = %s and chunk_index = %s
                                    """,
                                    (
                                        update.get("expected_start_line"),
                                        update.get("expected_end_line"),
                                        operation.get("target_document_id"),
                                        update.get("chunk_index"),
                                    ),
                                )
                                applied += cursor.rowcount
                    else:
                        skipped += 1
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(json.dumps(backup, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        failed += 1
        rolled_back += 1
        return {
            "status": "failed",
            "reason": _sanitize(str(exc)),
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "rolled_back": rolled_back,
            "affected_documents": len(backup["documents"]),
            "affected_chunks": affected_chunks,
            "transaction_used": True,
        }
    return {
        "status": "applied",
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "rolled_back": rolled_back,
        "affected_documents": len(backup["documents"]),
        "affected_chunks": affected_chunks,
        "transaction_used": True,
        "backup_path": backup_path.as_posix(),
        "generated_embeddings": 0,
        "rebuilt_documents": 0,
    }


def verify_reconciliation_result(**kwargs: Any) -> dict[str, Any]:
    audit = inspect_indexed_corpus(**kwargs)
    return {
        "status": "verified" if audit.get("status") == "verified" else "unverified",
        "audit": audit,
    }


def update_public_contracts(audit: Mapping[str, Any]) -> dict[str, Any]:
    public = json.loads(PUBLIC_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if public.get("snapshot_id") != SNAPSHOT_ID:
        raise RuntimeError("Unexpected source snapshot id.")
    if public.get("source_content_digest") != "87d2b0fbb810ae7433741e1a964f62666c8e04e136ab9f69437077e966eb5eb0":
        raise RuntimeError("Source corpus digest drifted; refusing to update indexed corpus contract.")
    public["indexed_corpus_status"] = audit.get("status")
    public["indexed_document_count"] = audit.get("document_count")
    public["indexed_active_document_count"] = audit.get("active_document_count")
    public["indexed_chunk_count"] = audit.get("chunk_count")
    public["indexed_verified_at"] = audit.get("audited_at")
    public["indexed_contract_digest"] = stable_hash(
        {
            "indexed_document_digest": audit.get("indexed_document_digest"),
            "chunk_boundary_parity": audit.get("chunk_boundary_parity", {}).get("status"),
            "embedding_contract_parity": audit.get("embedding_contract_parity"),
            "embedding_presence_parity": audit.get("embedding_presence_parity"),
            "embedding_dimension_parity": audit.get("embedding_dimension_parity"),
        }
    )
    public["indexed_drift_summary"] = audit.get("drift_summary", {})
    PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    baseline = json.loads(BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))
    baseline["corpus_snapshot"]["indexed_corpus_status"] = audit.get("status")
    baseline["corpus_snapshot"]["private_manifest_digest"] = public["private_manifest_digest"]
    baseline["corpus_snapshot"]["source_content_digest"] = public["source_content_digest"]
    reasons = [reason for reason in baseline.get("incomplete_reasons", []) if reason != "indexed_corpus_unverified"]
    if audit.get("status") != "verified":
        reasons.append("indexed_corpus_unverified")
    baseline["incomplete_reasons"] = sorted(set(reasons))
    baseline["status"] = "ready_for_baseline" if not baseline["incomplete_reasons"] else "incomplete"
    if baseline["status"] == "ready_for_baseline":
        baseline["notes"] = "Source and indexed corpus are verified. Sealed holdout remains unrun; soft target thresholds remain pending in the frozen scoring contract."
    else:
        baseline["notes"] = "Status remains incomplete because the indexed corpus is not fully verified. Sealed holdout remains unrun."
    baseline["baseline_contract_digest"] = stable_hash(
        {key: value for key, value in baseline.items() if key != "baseline_contract_digest"}
    )
    BASELINE_CONTRACT_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "public_snapshot_path": PUBLIC_SNAPSHOT_PATH.as_posix(),
        "baseline_contract_path": BASELINE_CONTRACT_PATH.as_posix(),
        "baseline_status": baseline["status"],
        "remaining_incomplete_reasons": baseline["incomplete_reasons"],
    }


def _read_remote_index_state(connection: Any, root_path: str) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("select id, name from public.knowledge_bases where root_path = %s", (root_path,))
        kb = cursor.fetchone()
        if kb is None:
            return {"knowledge_base_id": None}
        kb_id, kb_name = str(kb[0]), str(kb[1])
        cursor.execute(
            """
            select d.id,
                   d.knowledge_base_id,
                   d.relative_path,
                   d.content_hash,
                   d.file_size,
                   d.index_status,
                   d.parser_version,
                   d.chunking_version,
                   count(c.id)::integer,
                   count(*) filter (where c.id is not null and c.embedding is null)::integer,
                   count(*) filter (where c.id is not null and c.embedding_dimension is distinct from %s)::integer
            from public.documents d
            left join public.chunks c on c.document_id = d.id
            where d.knowledge_base_id = %s
            group by d.id
            order by d.relative_path
            """,
            (EXPECTED_EMBEDDING_DIMENSION, kb_id),
        )
        documents = [
            DocumentAuditRow(
                id=str(row[0]),
                knowledge_base_id=str(row[1]),
                relative_path=str(row[2]),
                content_hash=str(row[3]),
                file_size=int(row[4]),
                index_status=str(row[5]),
                parser_version=str(row[6]),
                chunking_version=str(row[7]),
                chunk_count=int(row[8]),
                embedding_missing_count=int(row[9]),
                embedding_dimension_violation_count=int(row[10]),
            )
            for row in cursor.fetchall()
        ]
        cursor.execute(
            """
            select c.document_id,
                   c.id,
                   c.chunk_index,
                   c.content_hash,
                   c.heading_path,
                   c.start_line,
                   c.end_line,
                   c.embedding_model,
                   c.embedding_dimension
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s
            order by c.document_id, c.chunk_index
            """,
            (kb_id,),
        )
        chunks_by_document: dict[str, list[dict[str, Any]]] = {}
        chunk_count = 0
        for row in cursor.fetchall():
            chunk_count += 1
            chunks_by_document.setdefault(str(row[0]), []).append(
                {
                    "chunk_id": str(row[1]),
                    "chunk_index": int(row[2]),
                    "content_hash": str(row[3]),
                    "heading_path": list(row[4] or []),
                    "start_line": row[5],
                    "end_line": row[6],
                    "embedding_model": row[7],
                    "embedding_dimension": row[8],
                }
            )
        cursor.execute("select index_status, count(*) from public.documents where knowledge_base_id = %s group by index_status", (kb_id,))
        status_counts = {str(row[0]): int(row[1]) for row in cursor.fetchall()}
        cursor.execute("select count(*) from public.chunks c left join public.documents d on d.id = c.document_id where d.id is null")
        orphan_chunk_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            select count(*) filter (where c.embedding is null)::integer,
                   count(*) filter (where c.embedding_dimension is distinct from %s)::integer
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s and d.index_status = 'indexed'
            """,
            (EXPECTED_EMBEDDING_DIMENSION, kb_id),
        )
        embedding_missing_count, embedding_dimension_violations = [int(value or 0) for value in cursor.fetchone()]
        cursor.execute(
            """
            select schema_version, parser_version, chunking_version, embedding_provider, embedding_model,
                   model_revision, embedding_dimension, normalize, distance_metric, configuration_fingerprint
            from public.index_configurations
            order by created_at desc
            limit 1
            """
        )
        config_row = cursor.fetchone()
        cursor.execute(
            """
            select id, status, run_type, documents_discovered, documents_created, documents_updated,
                   documents_deleted, documents_failed, chunks_created, chunks_deleted, finished_at is not null
            from public.index_runs
            where knowledge_base_id = %s
            order by started_at desc
            limit 1
            """,
            (kb_id,),
        )
        run_row = cursor.fetchone()
        cursor.execute(
            """
            select count(*)
            from public.index_failures f
            join public.index_runs r on r.id = f.run_id
            where r.knowledge_base_id = %s and coalesce(f.retryable, false) = false
            """,
            (kb_id,),
        )
        unresolved_failures = int(cursor.fetchone()[0])
    index_configuration = {}
    if config_row is not None:
        index_configuration = {
            "schema_version": config_row[0],
            "parser_version": config_row[1],
            "chunking_version": config_row[2],
            "embedding_provider": config_row[3],
            "embedding_model": config_row[4],
            "model_revision": config_row[5],
            "embedding_dimension": config_row[6],
            "normalize": config_row[7],
            "distance_metric": config_row[8],
            "configuration_fingerprint": config_row[9],
        }
    return {
        "knowledge_base_id": kb_id,
        "knowledge_base_name": kb_name,
        "documents": documents,
        "chunks_by_document": chunks_by_document,
        "chunk_count": chunk_count,
        "index_status_counts": status_counts,
        "orphan_chunk_count": orphan_chunk_count,
        "embedding_missing_count": embedding_missing_count,
        "embedding_dimension_violations": embedding_dimension_violations,
        "embedding_contract_parity": _embedding_contract_parity(index_configuration),
        "index_configuration": index_configuration,
        "recent_index_run": _redact_run(run_row),
        "unresolved_index_failures": unresolved_failures,
    }


def _embedding_contract_parity(config: Mapping[str, Any]) -> str:
    if not config:
        return "unavailable"
    if config.get("embedding_dimension") != EXPECTED_EMBEDDING_DIMENSION:
        return "drift"
    if config.get("distance_metric") != "cosine":
        return "drift"
    return "verified"


def _redact_run(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "status": row[1],
        "run_type": row[2],
        "documents_discovered": row[3],
        "documents_created": row[4],
        "documents_updated": row[5],
        "documents_deleted": row[6],
        "documents_failed": row[7],
        "chunks_created": row[8],
        "chunks_deleted": row[9],
        "finished": bool(row[10]),
    }


def _backup_document(connection: Any, document_id: str) -> dict[str, Any] | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select id, knowledge_base_id, relative_path, file_name, title, content_hash, file_size,
                   source_modified_at, parser_version, chunking_version, index_status, last_indexed_at,
                   last_error, metadata, created_at, updated_at
            from public.documents
            where id = %s
            for update
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        document = {
            "id": str(row[0]),
            "knowledge_base_id": str(row[1]),
            "path_digest": sha256_text(str(row[2])),
            "file_name_digest": sha256_text(str(row[3])),
            "title_digest": sha256_text(str(row[4] or "")),
            "content_hash": row[5],
            "file_size": row[6],
            "source_modified_at": _iso(row[7]),
            "parser_version": row[8],
            "chunking_version": row[9],
            "index_status": row[10],
            "last_indexed_at": _iso(row[11]),
            "last_error_type": type(row[12]).__name__ if row[12] else None,
            "metadata": _redact_metadata(row[13]),
            "created_at": _iso(row[14]),
            "updated_at": _iso(row[15]),
            "restore_note": "Private path/file_name/title are redacted; rollback requires restoring from database PITR or rerunning source indexing if exact private path is needed.",
        }
        cursor.execute(
            """
            select id, document_id, index_configuration_id, chunk_index, content_hash, heading_path, start_line,
                   end_line, token_count, embedding is not null, embedding_model, embedding_dimension, metadata,
                   created_at, updated_at
            from public.chunks
            where document_id = %s
            order by chunk_index
            """,
            (document_id,),
        )
        chunks = [
            {
                "id": str(chunk[0]),
                "document_id": str(chunk[1]),
                "index_configuration_id": str(chunk[2]),
                "chunk_index": chunk[3],
                "content_hash": chunk[4],
                "heading_path_digest": sha256_text(json.dumps(list(chunk[5] or []), ensure_ascii=False)),
                "start_line": chunk[6],
                "end_line": chunk[7],
                "token_count": chunk[8],
                "embedding_present": bool(chunk[9]),
                "embedding_model": chunk[10],
                "embedding_dimension": chunk[11],
                "metadata": _redact_metadata(chunk[12]),
                "created_at": _iso(chunk[13]),
                "updated_at": _iso(chunk[14]),
            }
            for chunk in cursor.fetchall()
        ]
    return {"document": document, "chunks": chunks}


def _backup_orphan_chunks(connection: Any) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select id, document_id, index_configuration_id, chunk_index, content_hash, heading_path,
                   start_line, end_line, token_count, embedding is not null, embedding_model,
                   embedding_dimension, metadata, created_at, updated_at
            from public.chunks c
            where not exists (select 1 from public.documents d where d.id = c.document_id)
            """
        )
        return [
            {
                "id": str(row[0]),
                "document_id": str(row[1]),
                "index_configuration_id": str(row[2]),
                "chunk_index": row[3],
                "content_hash": row[4],
                "heading_path_digest": sha256_text(json.dumps(list(row[5] or []), ensure_ascii=False)),
                "start_line": row[6],
                "end_line": row[7],
                "token_count": row[8],
                "embedding_present": bool(row[9]),
                "embedding_model": row[10],
                "embedding_dimension": row[11],
                "metadata": _redact_metadata(row[12]),
                "created_at": _iso(row[13]),
                "updated_at": _iso(row[14]),
            }
            for row in cursor.fetchall()
        ]


def _backup_chunks_for_boundary_updates(connection: Any, document_id: str | None, updates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not document_id or not updates:
        return []
    indexes = [int(update["chunk_index"]) for update in updates]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select id, document_id, index_configuration_id, chunk_index, content_hash, heading_path,
                   start_line, end_line, token_count, embedding is not null, embedding_model,
                   embedding_dimension, metadata, created_at, updated_at
            from public.chunks
            where document_id = %s and chunk_index = any(%s)
            order by chunk_index
            """,
            (document_id, indexes),
        )
        return [
            {
                "id": str(row[0]),
                "document_id": str(row[1]),
                "index_configuration_id": str(row[2]),
                "chunk_index": row[3],
                "content_hash": row[4],
                "heading_path_digest": sha256_text(json.dumps(list(row[5] or []), ensure_ascii=False)),
                "start_line": row[6],
                "end_line": row[7],
                "token_count": row[8],
                "embedding_present": bool(row[9]),
                "embedding_model": row[10],
                "embedding_dimension": row[11],
                "metadata": _redact_metadata(row[12]),
                "created_at": _iso(row[13]),
                "updated_at": _iso(row[14]),
            }
            for row in cursor.fetchall()
        ]


def _check_operation_preconditions(connection: Any, operation: Mapping[str, Any]) -> str | None:
    action = operation.get("proposed_action")
    if action == "hard_delete_document":
        document_id = operation.get("target_document_id")
        expected_digest = operation.get("target_path_digest")
        expected_status = None
        for precondition in operation.get("preconditions", []):
            if precondition.get("field") == "index_status":
                expected_status = precondition.get("equals")
        with connection.cursor() as cursor:
            cursor.execute("select relative_path, index_status from public.documents where id = %s for update", (document_id,))
            row = cursor.fetchone()
        if row is None:
            return "precondition_failed: document_absent"
        if expected_digest and sha256_text(str(row[0])) != expected_digest:
            return "precondition_failed: path_digest_changed"
        if expected_status and row[1] != expected_status:
            return "precondition_failed: index_status_changed"
        return None
    if action == "update_chunk_boundary_metadata":
        document_id = operation.get("target_document_id")
        updates = list(operation.get("target_chunk_updates") or [])
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select chunk_index, start_line, end_line
                from public.chunks
                where document_id = %s and chunk_index = any(%s)
                for update
                """,
                (document_id, [int(update["chunk_index"]) for update in updates]),
            )
            actual = {int(row[0]): {"start_line": row[1], "end_line": row[2]} for row in cursor.fetchall()}
        for update in updates:
            chunk_index = int(update["chunk_index"])
            row = actual.get(chunk_index)
            if row is None:
                return "precondition_failed: chunk_absent"
            if row["start_line"] != update.get("actual_start_line") or row["end_line"] != update.get("actual_end_line"):
                return "precondition_failed: chunk_boundary_changed"
        return None
    if action == "delete_orphan_chunks":
        return None
    return "precondition_failed: unsupported_action"


def _preconditions_for_issue(issue: Mapping[str, Any]) -> list[dict[str, Any]]:
    preconditions = [{"field": "drift_type", "equals": issue.get("drift_type")}]
    if issue.get("document_id"):
        preconditions.append({"field": "document_id", "equals": issue.get("document_id")})
    if issue.get("index_status"):
        preconditions.append({"field": "index_status", "equals": issue.get("index_status")})
    if issue.get("path_digest"):
        preconditions.append({"field": "path_digest", "equals": issue.get("path_digest")})
    return preconditions


def _postconditions_for_issue(issue: Mapping[str, Any]) -> list[dict[str, Any]]:
    if issue.get("drift_type") in {"unexpected_document", "deleted_document_residual"}:
        return [{"field": "document_id", "absent": issue.get("document_id")}]
    if issue.get("drift_type") == "orphan_chunk":
        return [{"field": "orphan_chunk_count", "equals": 0}]
    if issue.get("drift_type") == "chunk_boundary_metadata_mismatch":
        return [{"field": "chunk_boundary_parity", "equals": "verified"}]
    return [{"field": "drift_type", "absent": issue.get("drift_type")}]


def _is_verified(issues: Sequence[Mapping[str, Any]], chunk_boundary: Mapping[str, Any]) -> bool:
    return not issues and chunk_boundary.get("status") == "verified"


def _summarize_issues(issues: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for issue in issues:
        drift_type = str(issue.get("drift_type"))
        summary[drift_type] = summary.get(drift_type, 0) + int(issue.get("count", 1))
    return summary


def _chunk_boundary_repair_issues(chunk_boundary: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for mismatch in chunk_boundary.get("mismatches", []):
        updates = []
        repairable = True
        for diff in mismatch.get("diffs", []):
            fields = set(diff.get("fields", []))
            if not fields <= {"start_line", "end_line"}:
                repairable = False
                break
            expected = diff.get("expected", {})
            actual = diff.get("actual", {})
            if expected.get("content_hash") != actual.get("content_hash"):
                repairable = False
                break
            if expected.get("heading_path_digest") != actual.get("heading_path_digest"):
                repairable = False
                break
            updates.append(
                {
                    "chunk_index": diff.get("chunk_index"),
                    "expected_start_line": expected.get("start_line"),
                    "expected_end_line": expected.get("end_line"),
                    "actual_start_line": actual.get("start_line"),
                    "actual_end_line": actual.get("end_line"),
                }
            )
        if repairable and updates:
            contract = DRIFT_CONTRACTS["chunk_boundary_metadata_mismatch"]
            issues.append(
                {
                    "drift_type": "chunk_boundary_metadata_mismatch",
                    "risk_level": contract["risk_level"],
                    "repairable": True,
                    "recommended_action": contract["recommended_action"],
                    "requires_manual_confirmation": True,
                    "rollback": contract["rollback"],
                    "postcondition": contract["postcondition"],
                    "document_id": mismatch.get("document_id"),
                    "path_digest": mismatch.get("path_digest"),
                    "chunk_updates": updates,
                }
            )
    return issues


def _redact_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    allowed = {}
    for key, item in value.items():
        if key in {"embedding_status", "embedding_provider", "embedding_model_revision", "embedding_input_template_version", "length_unit", "target_size", "max_size", "overlap", "source_chunk_index"}:
            allowed[key] = item
    return allowed


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return str(value)


def _sanitize(message: str) -> str:
    return message.replace(os.environ.get("DATABASE_URL", ""), "<database_url>")
