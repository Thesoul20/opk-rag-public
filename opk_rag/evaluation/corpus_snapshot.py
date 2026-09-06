from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from opk_rag.evaluation.phase2_governance import (
    ROOT,
    SCHEMA_CORPUS,
    stable_hash,
    stable_json_dumps,
    utc_now,
)
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.vault import VaultPathError, VaultScanConfig, VaultScanner, normalize_vault_root

PRIVATE_MANIFEST_SCHEMA = "opk-rag.private-corpus-manifest.v1"
PRIVATE_MANIFEST_PATH = ROOT / ".private" / "evaluation" / "phase2_corpus_manifest.json"
PUBLIC_SNAPSHOT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
BASELINE_CONTRACT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json"
SNAPSHOT_ID = "phase2-corpus-v1"
PATH_NORMALIZATION_RULE = "vault-relative POSIX path, sorted lexicographically, symlinks ignored by default"
SECRET_KEYS = ("api_key", "apikey", "password", "secret", "token", "database_url", "authorization")
SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|postgres(?:ql)?://[^:\s]+:[^@\s]+@|authorization\s*:\s*bearer\s+[A-Za-z0-9_\-.]{12,})",
    re.IGNORECASE,
)
ABSOLUTE_PATH_RE = re.compile(r"(^|[\"'\s])(/(?:Users|home|data|mnt|Volumes|var|private)/|[A-Za-z]:\\)")


@dataclass(frozen=True)
class CorpusIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class SnapshotValidation:
    status: str
    exit_code: int
    issues: tuple[CorpusIssue, ...]
    summary: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "issues": [issue.__dict__ for issue in self.issues],
            "summary": self.summary,
        }


def load_reference_env() -> Mapping[str, str]:
    load_project_env(ROOT)
    return os.environ


def resolve_reference_vault_path(env: Mapping[str, str] | None = None, vault_path: str | Path | None = None) -> Path:
    if vault_path is not None:
        return Path(vault_path).expanduser()
    env = env or load_reference_env()
    configured = str(env.get("OPK_RAG_VAULT_PATH", "")).strip()
    if not configured:
        raise VaultPathError("OPK_RAG_VAULT_PATH is required for Phase 2 corpus snapshot operations.")
    return Path(configured).expanduser()


def scan_vault_snapshot(vault_path: str | Path) -> dict[str, Any]:
    scan = VaultScanner(VaultScanConfig()).scan(vault_path)
    source_files = [
        {
            "path": item.relative_path,
            "path_digest": sha256_text(item.relative_path),
            "content_sha256": item.content_hash,
            "size_bytes": item.size_bytes,
            "file_type": "markdown",
        }
        for item in scan.files
    ]
    total_bytes = sum(item["size_bytes"] for item in source_files)
    return {
        "root": scan.root,
        "source_files": source_files,
        "markdown_file_count": len(source_files),
        "total_byte_count": total_bytes,
        "source_content_digest": source_content_digest(source_files),
        "scan_failures": [
            {
                "path_digest": sha256_text(failure.relative_path),
                "stage": failure.stage,
                "error_type": failure.error_type,
            }
            for failure in scan.failures
        ],
        "scan_policy": {
            "supported_extensions": list(scan.supported_extensions),
            "ignored_dir_names": list(scan.ignored_dir_names),
            "follow_directory_symlinks": scan.follow_directory_symlinks,
            "follow_file_symlinks": scan.follow_file_symlinks,
            "path_normalization": PATH_NORMALIZATION_RULE,
        },
    }


def source_content_digest(source_files: list[dict[str, Any]]) -> str:
    rows = [
        {
            "path_digest": item["path_digest"],
            "content_sha256": item["content_sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in sorted(source_files, key=lambda value: value["path"])
    ]
    return stable_hash(rows)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(manifest).encode("utf-8")).hexdigest()


def build_private_manifest(
    vault_path: str | Path,
    runtime_contract: Mapping[str, Any],
    *,
    snapshot_id: str = SNAPSHOT_ID,
    created_at: str | None = None,
    identity_salt: str | None = None,
    indexed_corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scan = scan_vault_snapshot(vault_path)
    root = scan["root"]
    salt = identity_salt or secrets.token_hex(32)
    embedding_contract = runtime_contract.get("embedding_contract") if isinstance(runtime_contract.get("embedding_contract"), dict) else {}
    chunking_contract_digest = str(runtime_contract.get("chunking_contract_digest") or stable_hash(
        {
            "parser_version": embedding_contract.get("parser_version", "markdown-parser-v1"),
            "chunking_version": embedding_contract.get("chunking_version", "markdown-chunker-v1"),
        }
    ))
    return {
        "schema_version": PRIVATE_MANIFEST_SCHEMA,
        "snapshot_id": snapshot_id,
        "created_at": created_at or utc_now(),
        "vault": {
            "identity_digest": vault_identity_digest(root.canonical_path, salt),
            "identity_salt": salt,
            "absolute_path_stored": False,
            "markdown_file_count": scan["markdown_file_count"],
            "total_byte_count": scan["total_byte_count"],
            "source_content_digest": scan["source_content_digest"],
        },
        "source_files": scan["source_files"],
        "scan_policy": scan["scan_policy"],
        "index_contract": {
            "chunking_contract_digest": chunking_contract_digest,
            "embedding_contract_digest": str(runtime_contract.get("embedding_contract_digest") or ""),
            "retrieval_contract_digest": str(runtime_contract.get("retrieval_contract_digest") or ""),
        },
        "indexed_corpus": dict(indexed_corpus or unchecked_indexed_corpus()),
        "privacy": {
            "contains_document_content": False,
            "contains_absolute_paths": False,
            "contains_secrets": False,
            "git_tracked": False,
        },
    }


def unchecked_indexed_corpus() -> dict[str, Any]:
    return {
        "status": "unchecked",
        "document_count": None,
        "chunk_count": None,
        "indexed_document_digest": None,
        "missing_source_documents": [],
        "unexpected_indexed_documents": [],
        "stale_documents": [],
        "failed_documents": [],
        "unverifiable": [
            "full chunk boundary parity",
            "complete embedding vector content parity without reading embeddings",
        ],
    }


def build_public_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    indexed = manifest.get("indexed_corpus") if isinstance(manifest.get("indexed_corpus"), dict) else {}
    index_contract = manifest.get("index_contract") if isinstance(manifest.get("index_contract"), dict) else {}
    return {
        "schema_version": SCHEMA_CORPUS,
        "snapshot_id": manifest["snapshot_id"],
        "created_at": manifest["created_at"],
        "status": "ready",
        "markdown_file_count": manifest["vault"]["markdown_file_count"],
        "total_byte_count": manifest["vault"]["total_byte_count"],
        "source_content_digest": manifest["vault"]["source_content_digest"],
        "private_manifest_digest": manifest_digest(manifest),
        "private_manifest_required": True,
        "private_manifest_present_at_freeze": True,
        "absolute_paths_redacted": True,
        "relative_paths_redacted": True,
        "document_content_included": False,
        "contains_private_filenames": False,
        "chunking_contract_digest": index_contract.get("chunking_contract_digest", ""),
        "embedding_contract_digest": index_contract.get("embedding_contract_digest", ""),
        "retrieval_contract_digest": index_contract.get("retrieval_contract_digest", ""),
        "indexed_corpus_status": indexed.get("status", "unchecked"),
        "indexed_document_count": indexed.get("document_count"),
        "indexed_chunk_count": indexed.get("chunk_count"),
        "notes": "Public summary binds to a gitignored private per-file manifest. It intentionally excludes file names, paths, titles, content, and secrets.",
    }


def write_private_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_public_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_private_manifest(path: Path = PRIVATE_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_private_manifest(manifest: Mapping[str, Any]) -> list[CorpusIssue]:
    issues: list[CorpusIssue] = []
    if manifest.get("schema_version") != PRIVATE_MANIFEST_SCHEMA:
        issues.append(CorpusIssue("error", "private_manifest_schema", "Invalid private manifest schema."))
    if manifest.get("snapshot_id") != SNAPSHOT_ID:
        issues.append(CorpusIssue("error", "snapshot_id_mismatch", "Unexpected private manifest snapshot_id."))
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        issues.append(CorpusIssue("error", "private_manifest_source_files", "Private manifest source_files must be a list."))
        return issues
    paths = []
    for item in source_files:
        if not isinstance(item, dict):
            issues.append(CorpusIssue("error", "private_manifest_source_file", "Source file entries must be objects."))
            continue
        path = str(item.get("path", ""))
        paths.append(path)
        if not path or Path(path).is_absolute() or "\\" in path:
            issues.append(CorpusIssue("error", "private_manifest_path", "Source file path must be vault-relative POSIX."))
        if item.get("path_digest") != sha256_text(path):
            issues.append(CorpusIssue("error", "path_digest_mismatch", "Source file path digest mismatch."))
        if not _is_sha256(item.get("content_sha256")):
            issues.append(CorpusIssue("error", "content_sha256_invalid", "Source file content_sha256 is invalid."))
        if not isinstance(item.get("size_bytes"), int) or item.get("size_bytes") < 0:
            issues.append(CorpusIssue("error", "size_bytes_invalid", "Source file size_bytes is invalid."))
    if paths != sorted(paths):
        issues.append(CorpusIssue("error", "source_files_not_sorted", "Source files must be sorted by relative path."))
    vault = manifest.get("vault") if isinstance(manifest.get("vault"), dict) else {}
    if vault.get("markdown_file_count") != len(source_files):
        issues.append(CorpusIssue("error", "markdown_file_count_mismatch", "Manifest file count does not match source_files."))
    if vault.get("total_byte_count") != sum(int(item.get("size_bytes", 0)) for item in source_files if isinstance(item, dict)):
        issues.append(CorpusIssue("error", "total_byte_count_mismatch", "Manifest byte count does not match source_files."))
    if vault.get("source_content_digest") != source_content_digest([item for item in source_files if isinstance(item, dict)]):
        issues.append(CorpusIssue("error", "source_digest_mismatch", "Manifest source_content_digest is inconsistent."))
    if vault.get("absolute_path_stored") is not False:
        issues.append(CorpusIssue("error", "absolute_path_stored", "Private manifest must not store absolute Vault paths."))
    privacy = manifest.get("privacy") if isinstance(manifest.get("privacy"), dict) else {}
    for key in ("contains_document_content", "contains_absolute_paths", "contains_secrets", "git_tracked"):
        if privacy.get(key) is not False:
            issues.append(CorpusIssue("error", f"privacy_{key}", f"Privacy flag {key} must be false."))
    issues.extend(validate_no_secret_like_fields(manifest))
    return issues


def validate_no_secret_like_fields(value: Any, path: str = "$") -> list[CorpusIssue]:
    issues: list[CorpusIssue] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                any(secret in normalized for secret in SECRET_KEYS)
                and normalized != "identity_salt"
                and not path.startswith("$.privacy")
            ):
                issues.append(CorpusIssue("error", "secret_like_field", f"Secret-like field is not allowed at {path}.{key}."))
            issues.extend(validate_no_secret_like_fields(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(validate_no_secret_like_fields(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        if SECRET_VALUE_RE.search(value):
            issues.append(CorpusIssue("error", "secret_like_value", f"Secret-like value is not allowed at {path}."))
        if ABSOLUTE_PATH_RE.search(value):
            issues.append(CorpusIssue("error", "absolute_path_value", f"Absolute path-like value is not allowed at {path}."))
    return issues


def compare_source_snapshot(manifest: Mapping[str, Any], vault_path: str | Path) -> tuple[str, list[CorpusIssue], dict[str, Any]]:
    scan = scan_vault_snapshot(vault_path)
    issues: list[CorpusIssue] = []
    current_digest = scan["source_content_digest"]
    frozen_digest = manifest.get("vault", {}).get("source_content_digest")
    if current_digest != frozen_digest:
        issues.append(CorpusIssue("error", "snapshot_drift", "Current Vault source corpus differs from frozen manifest."))
    current_files = {item["path"]: item for item in scan["source_files"]}
    frozen_files = {item["path"]: item for item in manifest.get("source_files", []) if isinstance(item, dict)}
    added = sorted(set(current_files) - set(frozen_files))
    deleted = sorted(set(frozen_files) - set(current_files))
    modified = sorted(
        path
        for path in set(current_files) & set(frozen_files)
        if current_files[path]["content_sha256"] != frozen_files[path].get("content_sha256")
        or current_files[path]["size_bytes"] != frozen_files[path].get("size_bytes")
    )
    summary = {
        "markdown_file_count": scan["markdown_file_count"],
        "total_byte_count": scan["total_byte_count"],
        "source_content_digest": current_digest,
        "drift": current_digest != frozen_digest,
        "added_count": len(added),
        "deleted_count": len(deleted),
        "modified_count": len(modified),
    }
    return ("snapshot_match" if not issues else "snapshot_drift"), issues, summary


def verify_snapshot(
    *,
    vault_path: str | Path | None = None,
    private_manifest_path: Path = PRIVATE_MANIFEST_PATH,
    public_snapshot_path: Path = PUBLIC_SNAPSHOT_PATH,
    check_index: bool = False,
    database_url: str | None = None,
) -> SnapshotValidation:
    issues: list[CorpusIssue] = []
    summary: dict[str, Any] = {
        "private_manifest_path": ".private/evaluation/phase2_corpus_manifest.json",
        "indexed_corpus_check": "skipped",
        "document_count": None,
        "chunk_count": None,
        "missing_source_documents": 0,
        "unexpected_indexed_documents": 0,
        "stale_documents": 0,
        "failed_documents": 0,
        "unverifiable": [],
    }
    if not private_manifest_path.exists():
        issues.append(CorpusIssue("incomplete", "private_manifest_missing", "Private corpus manifest is missing.", ".private/evaluation/phase2_corpus_manifest.json"))
        return SnapshotValidation("private_manifest_missing", 2, tuple(issues), summary)
    if is_git_tracked(private_manifest_path):
        issues.append(CorpusIssue("error", "private_manifest_git_tracked", "Private corpus manifest must not be Git tracked.", ".private/evaluation/phase2_corpus_manifest.json"))

    try:
        manifest = load_private_manifest(private_manifest_path)
    except json.JSONDecodeError:
        issues.append(CorpusIssue("error", "private_manifest_invalid", "Private corpus manifest is not valid JSON.", ".private/evaluation/phase2_corpus_manifest.json"))
        return SnapshotValidation("private_manifest_invalid", 1, tuple(issues), summary)
    issues.extend(validate_private_manifest(manifest))

    if public_snapshot_path.exists():
        try:
            public = json.loads(public_snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            public = {}
            issues.append(CorpusIssue("error", "public_snapshot_invalid", "Public corpus snapshot is not valid JSON.", public_snapshot_path.as_posix()))
        if public.get("private_manifest_digest") != manifest_digest(manifest):
            issues.append(CorpusIssue("error", "public_private_digest_mismatch", "Public snapshot digest does not match private manifest."))
        if public.get("source_content_digest") != manifest.get("vault", {}).get("source_content_digest"):
            issues.append(CorpusIssue("error", "public_source_digest_mismatch", "Public snapshot source digest does not match private manifest."))
        issues.extend(validate_public_snapshot_privacy(public))
    else:
        issues.append(CorpusIssue("error", "public_snapshot_missing", "Public corpus snapshot is missing.", public_snapshot_path.as_posix()))

    try:
        resolved_vault_path = resolve_reference_vault_path(vault_path=vault_path)
        root = normalize_vault_root(resolved_vault_path)
    except VaultPathError as exc:
        issues.append(CorpusIssue("incomplete", "vault_unavailable", sanitize_message(str(exc))))
        return _finish_validation(issues, summary)
    expected_identity = vault_identity_digest(root.canonical_path, str(manifest.get("vault", {}).get("identity_salt", "")))
    if expected_identity != manifest.get("vault", {}).get("identity_digest"):
        issues.append(CorpusIssue("error", "vault_identity_mismatch", "Current Vault identity does not match frozen manifest."))
        summary["vault_identity_match"] = False
    else:
        summary["vault_identity_match"] = True

    source_status, source_issues, source_summary = compare_source_snapshot(manifest, root.path)
    issues.extend(source_issues)
    summary.update(source_summary)
    summary["source_status"] = source_status

    if check_index:
        from opk_rag.evaluation.indexed_corpus_reconciliation import inspect_indexed_corpus as inspect_reconciled_index

        index_result = inspect_reconciled_index(
            vault_path=root.path,
            manifest=manifest,
            database_url=database_url,
            include_chunk_boundary=True,
        )
        summary["indexed_corpus_check"] = index_result["status"]
        summary["document_count"] = index_result.get("document_count")
        summary["chunk_count"] = index_result.get("chunk_count")
        summary["missing_source_documents"] = len(index_result.get("missing_source_documents", []))
        summary["unexpected_indexed_documents"] = len(index_result.get("unexpected_indexed_documents", []))
        summary["stale_documents"] = len(index_result.get("stale_documents", []))
        summary["failed_documents"] = len(index_result.get("failed_or_non_active_documents", []))
        summary["orphan_chunks"] = index_result.get("orphan_chunk_count", 0)
        summary["embedding_dimension_violations"] = index_result.get("embedding_dimension_violations", 0)
        summary["chunk_boundary_status"] = index_result.get("chunk_boundary_parity", {}).get("status")
        summary["unverifiable"] = [
            "embedding vector value parity is not required for indexed corpus verification"
        ] if index_result.get("embedding_value_parity") == "not_required" else []
        if index_result["status"] == "unavailable":
            issues.append(CorpusIssue("incomplete", "index_unavailable", index_result.get("reason", "Indexed corpus check unavailable.")))
        elif index_result["status"] not in {"verified", "unchecked"}:
            issues.append(CorpusIssue("error", "index_drift", "Indexed corpus differs from frozen source corpus."))
    return _finish_validation(issues, summary)


def inspect_indexed_corpus(
    vault_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    database_url: str | None = None,
) -> dict[str, Any]:
    database_url = database_url or os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return {**unchecked_indexed_corpus(), "status": "unavailable", "reason": "DATABASE_URL is not configured."}
    root = normalize_vault_root(vault_path)
    try:
        from opk_rag.db.connection import connect_postgres

        with connect_postgres(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select id from public.knowledge_bases where root_path = %s", (root.canonical_path,))
                kb_row = cursor.fetchone()
                if kb_row is None:
                    return {**unchecked_indexed_corpus(), "status": "unavailable", "reason": "Knowledge base for current Vault was not found."}
                kb_id = kb_row[0]
                cursor.execute(
                    """
                    select id, relative_path, content_hash, file_size, index_status, parser_version, chunking_version
                    from public.documents
                    where knowledge_base_id = %s
                    order by relative_path
                    """,
                    (kb_id,),
                )
                documents = cursor.fetchall()
                cursor.execute(
                    """
                    select count(*)
                    from public.chunks c
                    join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s
                    """,
                    (kb_id,),
                )
                chunk_count = int(cursor.fetchone()[0])
                cursor.execute(
                    """
                    select count(*)
                    from public.chunks c
                    left join public.documents d on d.id = c.document_id
                    where d.id is null
                    """
                )
                orphan_chunks = int(cursor.fetchone()[0])
                cursor.execute(
                    """
                    select count(*)
                    from public.chunks c
                    join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and c.embedding_dimension <> 1024
                    """,
                    (kb_id,),
                )
                bad_embedding_dimension = int(cursor.fetchone()[0])
    except Exception as exc:
        return {**unchecked_indexed_corpus(), "status": "unavailable", "reason": sanitize_message(str(exc))}

    source = {item["path"]: item for item in manifest.get("source_files", []) if isinstance(item, dict)}
    indexed = {
        row[1]: {
            "id": row[0],
            "relative_path": row[1],
            "content_hash": row[2],
            "file_size": row[3],
            "index_status": row[4],
            "parser_version": row[5],
            "chunking_version": row[6],
        }
        for row in documents
    }
    missing = sorted(path for path in source if path not in indexed)
    unexpected = sorted(path for path in indexed if path not in source)
    stale = sorted(
        path
        for path in set(source) & set(indexed)
        if indexed[path]["content_hash"] != source[path]["content_sha256"]
        or int(indexed[path]["file_size"]) != int(source[path]["size_bytes"])
    )
    failed = sorted(path for path, row in indexed.items() if row["index_status"] in {"failed", "pending", "processing", "deleted"})
    status = "verified" if not any((missing, unexpected, stale, failed, orphan_chunks, bad_embedding_dimension)) else "partial"
    digest_rows = [
        {
            "path_digest": sha256_text(path),
            "content_sha256": row["content_hash"],
            "size_bytes": int(row["file_size"]),
            "index_status": row["index_status"],
        }
        for path, row in sorted(indexed.items())
    ]
    return {
        "status": status,
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "indexed_document_digest": stable_hash(digest_rows),
        "missing_source_documents": [sha256_text(path) for path in missing],
        "unexpected_indexed_documents": [sha256_text(path) for path in unexpected],
        "stale_documents": [sha256_text(path) for path in stale],
        "failed_documents": [sha256_text(path) for path in failed],
        "orphan_chunk_count": orphan_chunks,
        "bad_embedding_dimension_chunk_count": bad_embedding_dimension,
        "unverifiable": [
            "chunk boundaries cannot be proven identical without recomputing chunking",
            "embedding vector values are not rehashed by this read-only check",
            "deleted source documents cannot be mapped by title if relative_path is absent",
        ],
    }


def validate_public_snapshot_privacy(public: Mapping[str, Any]) -> list[CorpusIssue]:
    issues: list[CorpusIssue] = []
    if public.get("absolute_paths_redacted") is not True:
        issues.append(CorpusIssue("error", "public_absolute_paths_not_redacted", "Public snapshot must redact absolute paths."))
    if public.get("relative_paths_redacted") is not True:
        issues.append(CorpusIssue("error", "public_relative_paths_not_redacted", "Public snapshot must redact relative paths."))
    if public.get("document_content_included") is not False:
        issues.append(CorpusIssue("error", "public_document_content_included", "Public snapshot must not include document content."))
    issues.extend(validate_no_secret_like_fields(public))
    return issues


def create_snapshot_artifacts(
    *,
    vault_path: str | Path | None = None,
    private_manifest_path: Path = PRIVATE_MANIFEST_PATH,
    public_snapshot_path: Path = PUBLIC_SNAPSHOT_PATH,
    baseline_contract_path: Path = BASELINE_CONTRACT_PATH,
    runtime_contract: Mapping[str, Any] | None = None,
    check_index: bool = False,
) -> dict[str, Any]:
    from opk_rag.evaluation.phase2_governance import build_runtime_contract

    resolved = resolve_reference_vault_path(vault_path=vault_path)
    runtime = dict(runtime_contract or build_runtime_contract())
    preliminary = build_private_manifest(resolved, runtime)
    if check_index:
        indexed = inspect_indexed_corpus(resolved, preliminary)
        preliminary = build_private_manifest(
            resolved,
            runtime,
            created_at=preliminary["created_at"],
            identity_salt=preliminary["vault"]["identity_salt"],
            indexed_corpus=indexed,
        )
    public = build_public_snapshot(preliminary)
    write_private_manifest(private_manifest_path, preliminary)
    write_public_snapshot(public_snapshot_path, public)
    update_baseline_contract(baseline_contract_path, public)
    return {
        "private_manifest_path": ".private/evaluation/phase2_corpus_manifest.json",
        "public_snapshot_path": "evaluation-data/dogfooding/phase2_corpus_snapshot_public.json",
        "private_manifest_digest": public["private_manifest_digest"],
        "source_content_digest": public["source_content_digest"],
        "markdown_file_count": public["markdown_file_count"],
        "total_byte_count": public["total_byte_count"],
        "indexed_corpus_status": public["indexed_corpus_status"],
        "indexed_document_count": public["indexed_document_count"],
        "indexed_chunk_count": public["indexed_chunk_count"],
    }


def update_baseline_contract(path: Path, public_snapshot: Mapping[str, Any]) -> None:
    contract = json.loads(path.read_text(encoding="utf-8"))
    indexed_status = str(public_snapshot.get("indexed_corpus_status") or "unchecked")
    corpus_status = "frozen" if indexed_status in {"verified", "unchecked", "partial", "unavailable"} else "partial"
    contract["corpus_snapshot"] = {
        "status": corpus_status,
        "snapshot_id": public_snapshot["snapshot_id"],
        "public_manifest": "evaluation-data/dogfooding/phase2_corpus_snapshot_public.json",
        "private_manifest_required": True,
        "private_manifest_git_tracked": False,
        "source_content_digest": public_snapshot["source_content_digest"],
        "private_manifest_digest": public_snapshot["private_manifest_digest"],
        "indexed_corpus_status": indexed_status,
    }
    contract["status"] = "incomplete"
    reasons = ["holdout_missing"]
    if indexed_status in {"unchecked", "unavailable", "partial"}:
        reasons.append("indexed_corpus_unverified")
    contract["incomplete_reasons"] = reasons
    contract["notes"] = (
        "Status remains incomplete because sealed_holdout is missing. "
        "The Phase 2 source corpus is frozen with a gitignored private per-file manifest; "
        f"indexed corpus status is {indexed_status}."
    )
    contract["baseline_contract_digest"] = stable_hash(
        {key: value for key, value in contract.items() if key != "baseline_contract_digest"}
    )
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_git_tracked(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    result = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def vault_identity_digest(canonical_path: str, salt: str) -> str:
    return hashlib.sha256(f"opk-rag-vault-identity-v1\0{salt}\0{canonical_path}".encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def sanitize_message(message: str) -> str:
    sanitized = str(message).replace(str(ROOT), "<repo>")
    vault_path = os.environ.get("OPK_RAG_VAULT_PATH", "")
    if vault_path:
        sanitized = sanitized.replace(vault_path, "<vault>")
    return sanitized


def _finish_validation(issues: list[CorpusIssue], summary: dict[str, Any]) -> SnapshotValidation:
    if any(issue.severity == "error" for issue in issues):
        return SnapshotValidation("invalid", 1, tuple(issues), summary)
    if any(issue.severity == "incomplete" for issue in issues):
        return SnapshotValidation("incomplete", 2, tuple(issues), summary)
    if summary.get("source_status") == "snapshot_drift":
        return SnapshotValidation("snapshot_drift", 1, tuple(issues), summary)
    return SnapshotValidation("snapshot_match", 0, tuple(issues), summary)
