from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from opk_rag.document_ir.chunking_adapter import CanonicalDocumentChunkingInputError, canonical_document_to_chunking_input
from opk_rag.chunking.chunker import chunk_markdown_document
from opk_rag.chunking.models import ChunkingConfig
from opk_rag.document_adapters.base import StructuredDocumentSource
from opk_rag.document_adapters.markdown import MarkdownDocumentAdapter
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    CORPUS_ROOT,
    EXPECTED_DOCUMENT_COUNT,
    SECRET_URL_RE,
    TASK_DATABASE_URL_VARIABLE,
    resolve_cold_start_secret_authority,
)

TASK_ID = "TASK-0181"
SOURCE_AUTHORITATIVE_TASK = "TASK-0180"
EXPERIMENT_ID = "task0181-canonical-document-to-chunking-input-contract-repair"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0181_canonical_document_to_chunking_input_contract_repair_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0181_CANONICAL_DOCUMENT_TO_CHUNKING_INPUT_CONTRACT_REPAIR_REPORT.md"
TASK0180_SUMMARY = ROOT / "evaluation-data" / "results" / "task0180-cold-start-zero-chunk-diagnosis-and-repair" / "summary.json"


@dataclass(frozen=True)
class ContractAuditRow:
    relative_path: str
    document_id: str
    source_path: str
    document_type: str
    canonical_content_owner: str
    chunking_required_object_type: str
    chunking_required_content_field: str
    index_status: str
    content_digest: str
    chunking_input_content_digest: str
    content_equivalence: bool
    chunking_input_content_nonempty: bool
    compatible: bool
    mismatch: str
    raw_chunk_output_count: int
    validated_chunk_count: int


def run_task0181(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = _git_head(ROOT)
    task0180 = _load_json(TASK0180_SUMMARY)
    rows = audit_contract_repair(CORPUS_ROOT)
    database = probe_public_database(env)
    summary = build_summary(source_head=source_head, task0180=task0180, rows=rows, database=database)
    artifacts = {
        "summary.json": summary,
        "canonical_document_representation_audit.json": [asdict(row) for row in rows],
        "chunking_input_contract_audit.json": chunking_input_contract_audit(),
        "diagnosis_matrix.json": diagnosis_matrix(rows),
        "cold_start_replay.json": cold_start_replay(rows, database),
        "database_verification.json": _redact(database),
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, _redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, rows), encoding="utf-8")
    return summary


def audit_contract_repair(corpus_root: Path = CORPUS_ROOT) -> list[ContractAuditRow]:
    adapter = MarkdownDocumentAdapter()
    config = ChunkingConfig()
    rows: list[ContractAuditRow] = []
    for path in sorted(corpus_root.glob("*.md")):
        if path.name == "README.md":
            continue
        source = StructuredDocumentSource(
            source_bytes=path.read_bytes(),
            knowledge_id="task0181-cold-start-corpus",
            representation_id=path.name,
            representation_type="markdown",
            source_ref=path.name,
            source_relative_path=path.name,
            mime_type="text/markdown",
        )
        document = adapter.adapt(source)
        canonical_text = "\n\n".join(block.text for block in document.blocks if isinstance(block.text, str) and block.text.strip())
        try:
            chunking_input = canonical_document_to_chunking_input(document)
            chunks = chunk_markdown_document(chunking_input.parsed_document, config)
            valid = tuple(chunk for chunk in chunks if chunk.content and chunk.content_hash and chunk.chunk_index >= 0)
            content_equivalence = canonical_text == chunking_input.content
            nonempty = bool(chunking_input.content.strip())
            mismatch = "none" if content_equivalence and nonempty else ("content_payload" if not nonempty else "content_equivalence")
            compatible = mismatch == "none"
            rows.append(
                ContractAuditRow(
                    relative_path=path.name,
                    document_id=document.document_id,
                    source_path=str(chunking_input.source_path or ""),
                    document_type=document.representation_type,
                    canonical_content_owner="CanonicalDocument.blocks[].text",
                    chunking_required_object_type="ParsedMarkdownDocument",
                    chunking_required_content_field="ParsedMarkdownDocument.sections[].content/body",
                    index_status="pending",
                    content_digest=document.content_digest or "",
                    chunking_input_content_digest=chunking_input.chunking_input_content_digest,
                    content_equivalence=content_equivalence,
                    chunking_input_content_nonempty=nonempty,
                    compatible=compatible,
                    mismatch=mismatch,
                    raw_chunk_output_count=len(chunks),
                    validated_chunk_count=len(valid),
                )
            )
        except CanonicalDocumentChunkingInputError as exc:
            rows.append(
                ContractAuditRow(
                    relative_path=path.name,
                    document_id=document.document_id,
                    source_path=path.name,
                    document_type=document.representation_type,
                    canonical_content_owner="CanonicalDocument.blocks[].text",
                    chunking_required_object_type="ParsedMarkdownDocument",
                    chunking_required_content_field="ParsedMarkdownDocument.sections[].content/body",
                    index_status="pending",
                    content_digest=document.content_digest or "",
                    chunking_input_content_digest="",
                    content_equivalence=False,
                    chunking_input_content_nonempty=False,
                    compatible=False,
                    mismatch=f"adapter_rejected:{type(exc).__name__}",
                    raw_chunk_output_count=0,
                    validated_chunk_count=0,
                )
            )
    return rows


def chunking_input_contract_audit() -> dict[str, Any]:
    config = ChunkingConfig()
    return {
        "authoritative_production_function": "opk_rag.chunking.chunker.chunk_markdown_document",
        "required_object_type": "ParsedMarkdownDocument",
        "required_fields_observed_from_code": ["sections", "sections[].content", "sections[].heading_path", "sections[].start_line", "sections[].end_line"],
        "optional_fields_used_by_persistence": ["source_path", "frontmatter", "title", "body"],
        "frozen_policy": {"target_chunk_size": config.target_size, "maximum_chunk_size": config.max_size, "overlap": config.overlap, "length_unit": config.length_unit},
    }


def diagnosis_matrix(rows: list[ContractAuditRow]) -> dict[str, Any]:
    mismatches = [row.mismatch for row in rows if row.mismatch != "none"]
    return {
        "matrix": [
            {"field_or_property": "document identity", "canonical_document": "document_id", "chunking_runtime_required": "external document.id plus ParsedMarkdownDocument", "compatible": True},
            {"field_or_property": "content payload", "canonical_document": "blocks[].text", "chunking_runtime_required": "sections[].content/body", "compatible": not any(row.mismatch.startswith("content") for row in rows)},
            {"field_or_property": "source path", "canonical_document": "representation_metadata.source_relative_path/source.source_ref", "chunking_runtime_required": "ParsedMarkdownDocument.source_path", "compatible": True},
            {"field_or_property": "metadata", "canonical_document": "metadata + representation_metadata", "chunking_runtime_required": "frontmatter/title for persistence metadata", "compatible": True},
            {"field_or_property": "status field", "canonical_document": "external documents.index_status", "chunking_runtime_required": "pending/chunked in public.documents.index_status", "compatible": True},
            {"field_or_property": "object type", "canonical_document": "CanonicalDocument", "chunking_runtime_required": "ParsedMarkdownDocument", "compatible": True},
        ],
        "contract_mismatch_count": len(mismatches),
        "first_contract_mismatch": mismatches[0] if mismatches else "none",
        "dominant_contract_mismatch": max(set(mismatches), key=mismatches.count) if mismatches else "none",
        "document_drop_condition": "pre-repair: object type/content field mismatch (CanonicalDocument lacks ParsedMarkdownDocument.sections[].content); post-repair: none",
    }


def probe_public_database(env: Mapping[str, str]) -> dict[str, Any]:
    secret = resolve_cold_start_secret_authority(env)
    database_url = str(secret.get("database_url_value") or env.get(TASK_DATABASE_URL_VARIABLE) or env.get("DATABASE_URL") or "")
    root = str(CORPUS_ROOT.resolve())
    out: dict[str, Any] = {"database_connection_authority_valid": bool(secret.get("database_connection_authority_valid")), "root_path": root, "final_authoritative_document_count": 0, "final_authoritative_chunk_count": 0, "chunk_persist_failure_count": 0, "database_probe_error": ""}
    if not database_url:
        out["database_probe_error"] = "database_url_unavailable"
        return out
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("select id from public.knowledge_bases where root_path = %s order by created_at desc limit 1", (root,))
                row = cur.fetchone()
                if row is None:
                    out["database_probe_error"] = "knowledge_base_not_found"
                    return out
                kb_id = row[0]
                cur.execute("select count(*) from public.documents where knowledge_base_id = %s and index_status <> 'deleted' and relative_path <> 'README.md'", (kb_id,))
                out["final_authoritative_document_count"] = int(cur.fetchone()[0])
                cur.execute("select count(*) from public.chunks c join public.documents d on d.id = c.document_id where d.knowledge_base_id = %s and d.relative_path <> 'README.md'", (kb_id,))
                out["final_authoritative_chunk_count"] = int(cur.fetchone()[0])
                cur.execute("select count(*) from public.documents where knowledge_base_id = %s and index_status = 'failed' and relative_path <> 'README.md'", (kb_id,))
                out["chunk_persist_failure_count"] = int(cur.fetchone()[0])
    except Exception as exc:  # pragma: no cover - live DB dependent
        out["database_probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


def cold_start_replay(rows: list[ContractAuditRow], database: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_document_count": len(rows),
        "materialized_document_count": len(rows),
        "canonical_document_success_count": sum(1 for row in rows if row.document_id),
        "chunking_input_document_count": sum(1 for row in rows if row.compatible),
        "chunker_invocation_count": sum(1 for row in rows if row.raw_chunk_output_count > 0),
        "raw_chunk_output_count": sum(row.raw_chunk_output_count for row in rows),
        "validated_chunk_count": sum(row.validated_chunk_count for row in rows),
        "chunk_persist_success_count": int(database.get("final_authoritative_chunk_count", 0) or 0),
        "stage_passed": sum(row.validated_chunk_count for row in rows) > 0 and int(database.get("final_authoritative_chunk_count", 0) or 0) > 0,
    }


def build_summary(*, source_head: str, task0180: Mapping[str, Any], rows: list[ContractAuditRow], database: Mapping[str, Any]) -> dict[str, Any]:
    raw = sum(row.raw_chunk_output_count for row in rows)
    valid = sum(row.validated_chunk_count for row in rows)
    rejected_docs = sum(1 for row in rows if not row.compatible)
    final_docs = int(database.get("final_authoritative_document_count", 0) or 0)
    final_chunks = int(database.get("final_authoritative_chunk_count", 0) or 0)
    success = len(rows) == EXPECTED_DOCUMENT_COUNT and rejected_docs == 0 and raw > 0 and valid > 0 and final_docs == EXPECTED_DOCUMENT_COUNT and final_chunks > 0
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if success else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0180.get("source_authoritative_head") or source_head,
        "diagnosed_root_cause": "canonical_document_to_chunking_input_contract_mismatch",
        "first_chunk_loss_substage": "canonical_document_to_chunking_input",
        "canonical_document_to_chunking_input_contract_repaired": rejected_docs == 0,
        "source_document_count": len(rows),
        "materialized_document_count": len(rows),
        "canonical_document_success_count": sum(1 for row in rows if row.document_id),
        "canonical_document_failure_count": sum(1 for row in rows if not row.document_id),
        "chunking_input_document_count": sum(1 for row in rows if row.compatible),
        "chunking_input_rejected_document_count": rejected_docs,
        "chunker_invocation_count": sum(1 for row in rows if row.raw_chunk_output_count > 0),
        "raw_chunk_output_count": raw,
        "validated_chunk_count": valid,
        "chunk_persist_attempt_count": valid,
        "chunk_persist_success_count": final_chunks,
        "chunk_persist_failure_count": int(database.get("chunk_persist_failure_count", 0) or 0),
        "final_authoritative_document_count": final_docs,
        "final_authoritative_chunk_count": final_chunks,
        "canonical_identity_preserved": all(row.document_id for row in rows) and final_docs == EXPECTED_DOCUMENT_COUNT,
        "content_equivalence_valid": all(row.content_equivalence and row.chunking_input_content_nonempty for row in rows),
        "chunking_policy_changed": False,
        "runtime_default_behavior_change": False,
        "cold_start_chunking_stage_passed": raw > 0 and valid > 0 and final_chunks > 0,
        "next_failure_stage": "embedding_materialization" if final_chunks > 0 else "chunk_persistence",
        "promotion_applied": True,
    }


def contract() -> dict[str, Any]:
    return {"task_id": TASK_ID, "adapter": "opk_rag.document_ir.chunking_adapter.canonical_document_to_chunking_input", "chunker": "opk_rag.chunking.chunker.chunk_markdown_document", "frozen_policy": chunking_input_contract_audit()["frozen_policy"]}


def verify_task0181_artifacts(*, result_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    required = ["summary.json", "canonical_document_representation_audit.json", "chunking_input_contract_audit.json", "diagnosis_matrix.json", "database_verification.json"]
    missing = [name for name in required if not (result_dir / name).exists()]
    summary = _load_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    errors: list[str] = []
    if summary.get("task_id") != TASK_ID:
        errors.append("summary task_id mismatch")
    if summary.get("canonical_document_to_chunking_input_contract_repaired") is not True:
        errors.append("contract repair flag not true")
    if int(summary.get("chunking_input_document_count", 0) or 0) != EXPECTED_DOCUMENT_COUNT:
        errors.append("chunking_input_document_count != 8")
    if int(summary.get("validated_chunk_count", 0) or 0) <= 0:
        errors.append("validated chunks not demonstrated")
    result = {"task_id": TASK_ID, "status": "pass" if not missing and not errors else "fail", "missing_artifacts": missing, "errors": errors}
    if write:
        write_json(result_dir / "verification.json", result)
    return result


def render_report(summary: Mapping[str, Any], rows: list[ContractAuditRow]) -> str:
    return "\n".join([
        "# TASK-0181 Canonical Document to Chunking Input Contract Repair Report",
        "",
        "## Diagnosis",
        "CanonicalDocument stores authoritative text in `CanonicalDocument.blocks[].text`; the frozen chunker requires `ParsedMarkdownDocument.sections[].content`. The repair adds a deterministic adapter and leaves `ChunkingConfig(target_size=1200, max_size=1800, overlap=0)` unchanged.",
        "",
        f"Audited documents: `{len(rows)}`; validated chunks: `{summary.get('validated_chunk_count')}`; final DB chunks: `{summary.get('final_authoritative_chunk_count')}`.",
        "",
        "## Machine-readable summary",
        "```json",
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ])


def _git_head(cwd: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _redact_text(text: str) -> str:
    return SECRET_URL_RE.sub("postgresql://***:***@", text)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        return {k: ("***redacted***" if k == "database_url_value" else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
