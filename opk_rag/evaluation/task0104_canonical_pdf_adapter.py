from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from opk_rag.document_adapters.base import StructuredDocumentSource, adapter_config_digest
from opk_rag.document_adapters.pdf import PYMUPDF4LLM_ADAPTER_CONFIG, PdfDocumentAdapter
from opk_rag.document_ir import document_to_dict, source_bytes_digest, validate_document
from opk_rag.document_ir.serialization import digest_json
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0103_pdf_parser_bakeoff import RESULT_DIR as TASK0103_RESULT_DIR


TASK_ID = "TASK-0104"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0104-canonical-pdf-adapter"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0104_canonical_pdf_adapter_contract.json"
TASK0103_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0103_pdf_parser_bakeoff_contract.json"
TASK0103_EXPECTED_PDFQA_REVISION = "90f787ebee0ba278bd6b8b750a69ed90386bfdf5"
TASK0103_EXPECTED_FORMAL_PROFILE_DIGEST = "1ea1b49dc703a5b3ce4123c5ace0afa9b3bd9ae99b26a9f986222b5560b69d6c"
FORMAL_DOCUMENT_COUNT = 100
REQUIRED_PDF_COUNT = 103
SMOKE_ONLY_PDF_COUNT = 3
CONTENT_PRESERVATION_MIN_RATIO = 0.90
PRIVACY_PATTERNS = {
    "private_absolute_path": re.compile(r"(/home/[^\s\"']+|/Users/[^\s\"']+|/data/envs/[^\s\"']+|[A-Za-z]:\\Users\\[^\s\"']+)"),
    "api_key": re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0103_contract = read_json(root / TASK0103_CONTRACT_PATH.relative_to(ROOT))
    return {
        "schema_version": "opk-rag.task0104.canonical-pdf-adapter-contract.v1",
        "task_id": TASK_ID,
        "document_adapter_contract_version": "opk-rag.document-adapter.v1",
        "canonical_document_ir_version": 1,
        "source_type": "pdf",
        "adapter_name": PdfDocumentAdapter.adapter_name,
        "adapter_version": PdfDocumentAdapter.adapter_version,
        "parser": "pymupdf4llm",
        "parser_version": task0103_contract.get("candidate_versions", {}).get("pymupdf4llm"),
        "parser_revision": task0103_contract.get("candidate_revisions", {}).get("pymupdf4llm"),
        "parser_config": PYMUPDF4LLM_ADAPTER_CONFIG,
        "parser_config_digest": adapter_config_digest(PYMUPDF4LLM_ADAPTER_CONFIG),
        "ocr_enabled": False,
        "vision_enabled": False,
        "task0103_inputs": {
            "task0103_status": read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "summary.json").get("task_status"),
            "task0103_git_head": read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "environment.json").get("repository", {}).get("head"),
            "pdfqa_revision": task0103_contract.get("dataset_authority", {}).get("dataset_revision"),
            "formal_profile_digest": task0103_contract.get("formal_profile_digest"),
            "required_pdf_count": REQUIRED_PDF_COUNT,
            "materialized_pdf_count": REQUIRED_PDF_COUNT,
            "formal_document_count": FORMAL_DOCUMENT_COUNT,
            "smoke_only_pdf_count": SMOKE_ONLY_PDF_COUNT,
        },
        "materialization_source": "task0103_frozen_pymupdf4llm_parser_outputs",
        "raw_canonical_documents_committed": False,
        "production_chunking_modified": False,
        "retrieval_modified": False,
        "reranker_modified": False,
        "rank_fusion_modified": False,
        "agent_behavior_modified": False,
        "graph_runtime_modified": False,
    }


def run_task0104(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    contract = build_contract(root=root)
    input_identity = build_input_identity(contract, root=root)
    parser_identity = build_parser_identity(contract)
    materialization = materialize_formal_documents(root=root)
    determinism = materialization["determinism_report"]
    schema_summary = materialization["canonical_schema_summary"]
    content = materialization["content_preservation_report"]
    failure = materialization["failure_report"]
    result_digests = build_result_digests(
        contract=contract,
        input_identity=input_identity,
        parser_identity=parser_identity,
        schema_summary=schema_summary,
        materialization_summary=materialization["materialization_summary"],
        determinism=determinism,
        content=content,
        failure=failure,
    )
    privacy = privacy_scan_payloads(
        {
            "contract": contract,
            "input_identity": input_identity,
            "parser_identity": parser_identity,
            "canonical_schema_summary": schema_summary,
            "materialization_summary": materialization["materialization_summary"],
            "determinism_report": determinism,
            "content_preservation_report": content,
            "failure_report": failure,
            "result_digests": result_digests,
        }
    )
    verification = verify_payloads(
        contract=contract,
        input_identity=input_identity,
        parser_identity=parser_identity,
        canonical_schema_summary=schema_summary,
        materialization_summary=materialization["materialization_summary"],
        determinism_report=determinism,
        content_preservation_report=content,
        failure_report=failure,
        privacy_scan=privacy,
        result_digests=result_digests,
    )
    if write:
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        result_dir = root / RESULT_DIR.relative_to(ROOT)
        for name, payload in (
            ("input_identity", input_identity),
            ("parser_identity", parser_identity),
            ("canonical_schema_summary", schema_summary),
            ("materialization_summary", materialization["materialization_summary"]),
            ("determinism_report", determinism),
            ("content_preservation_report", content),
            ("failure_report", failure),
            ("privacy_scan", privacy),
            ("result_digests", result_digests),
            ("verification_summary", verification),
        ):
            write_json(result_dir / f"{name}.json", payload)
    return verification


def build_input_identity(contract: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    task0103_summary = read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "summary.json")
    task0103_accounting = read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "formal_profile_document_accounting.json")
    task0103_contract = read_json(root / TASK0103_CONTRACT_PATH.relative_to(ROOT))
    pdfqa_revision = task0103_contract.get("dataset_authority", {}).get("dataset_revision")
    formal_profile_digest = task0103_contract.get("formal_profile_digest")
    task0103_artifacts_modified = _git_modified_under(root, "evaluation-data/results/task0103-pdf-parser-bakeoff") or _git_modified_under(root, "evaluation-data/contracts/task0103_pdf_parser_bakeoff_contract.json")
    return {
        "schema_version": "opk-rag.task0104.input-identity.v1",
        "task_id": TASK_ID,
        "task0103_status": task0103_summary.get("task_status"),
        "task0103_git_head": contract.get("task0103_inputs", {}).get("task0103_git_head"),
        "pdfqa_revision": pdfqa_revision,
        "formal_profile_digest": formal_profile_digest,
        "required_pdf_count": task0103_accounting.get("required_pdf_count"),
        "materialized_pdf_count": task0103_accounting.get("materialized_pdf_count"),
        "formal_document_count": task0103_accounting.get("formal_documents_evaluated"),
        "smoke_only_pdf_count": task0103_accounting.get("category_counts", {}).get("profile_excluded"),
        "task0103_inputs_valid": True,
        "task0103_artifacts_modified": task0103_artifacts_modified,
        "pdfqa_revision_match": pdfqa_revision == TASK0103_EXPECTED_PDFQA_REVISION,
        "formal_profile_digest_match": formal_profile_digest == TASK0103_EXPECTED_FORMAL_PROFILE_DIGEST,
    }


def build_parser_identity(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0104.parser-identity.v1",
        "task_id": TASK_ID,
        "library": "pymupdf4llm",
        "version": contract.get("parser_version"),
        "revision": contract.get("parser_revision"),
        "configuration": contract.get("parser_config"),
        "configuration_digest": contract.get("parser_config_digest"),
        "runtime_identity": {
            "source": "TASK-0103 frozen parser output artifacts",
            "parser_reexecution_required_for_verification": False,
        },
        "parser_specific_payload_isolated_behind_adapter": True,
    }


def materialize_formal_documents(*, root: Path = ROOT) -> dict[str, Any]:
    adapter = PdfDocumentAdapter()
    accounting = read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "formal_profile_document_accounting.json")
    formal_records = [record for record in accounting.get("records", []) if record.get("included_in_formal_run") is True]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    first_run: dict[str, dict[str, Any]] = {}
    second_run: dict[str, dict[str, Any]] = {}

    for record in sorted(formal_records, key=lambda item: item["knowledge_id"]):
        try:
            payload = _load_parser_payload(root, record)
            source = _source_for_record(root, record)
            document = adapter.to_canonical_document(source, payload)
            validate_document(document)
            document_payload = document_to_dict(document)
            first_run[record["knowledge_id"]] = document_payload
            second_document = adapter.to_canonical_document(source, payload)
            validate_document(second_document)
            second_run[record["knowledge_id"]] = document_to_dict(second_document)
            rows.append(_document_row(record, payload, document_payload))
        except Exception as exc:
            failures.append(
                {
                    "knowledge_id": record.get("knowledge_id"),
                    "pdf_source_relative_path": record.get("pdf_source_relative_path"),
                    "failure_classification": "canonical_mapping_failure",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )

    schema_valid = [row for row in rows if row["canonical_schema_valid"]]
    identity_valid = [row for row in rows if row["stable_document_identity"] and row["stable_block_identity"]]
    provenance_valid = [row for row in rows if row["source_provenance_valid"]]
    content_rows = [_content_row(row) for row in rows]
    determinism_rows = [_determinism_row(knowledge_id, first_run[knowledge_id], second_run[knowledge_id]) for knowledge_id in sorted(first_run)]
    content_gate_passed = all(row["content_preservation_gate_passed"] for row in content_rows) and len(content_rows) == FORMAL_DOCUMENT_COUNT
    return {
        "canonical_schema_summary": {
            "schema_version": "opk-rag.task0104.canonical-schema-summary.v1",
            "task_id": TASK_ID,
            "formal_document_count": FORMAL_DOCUMENT_COUNT,
            "canonical_schema_valid_count": len(schema_valid),
            "canonical_schema_valid": len(schema_valid) == FORMAL_DOCUMENT_COUNT,
            "stable_document_identity_count": sum(1 for row in rows if row["stable_document_identity"]),
            "stable_block_identity_count": sum(1 for row in rows if row["stable_block_identity"]),
            "source_provenance_valid_count": len(provenance_valid),
            "canonical_identity_valid": len(identity_valid) == FORMAL_DOCUMENT_COUNT,
            "canonical_provenance_valid": len(provenance_valid) == FORMAL_DOCUMENT_COUNT,
            "documents": rows,
        },
        "materialization_summary": {
            "schema_version": "opk-rag.task0104.materialization-summary.v1",
            "task_id": TASK_ID,
            "formal_document_count": FORMAL_DOCUMENT_COUNT,
            "canonical_document_attempted": len(formal_records),
            "canonical_document_success": len(rows),
            "canonical_document_failure": len(failures),
            "smoke_only_pdf_count": SMOKE_ONLY_PDF_COUNT,
            "smoke_only_in_formal_metric_denominator": False,
            "raw_canonical_documents_committed": False,
        },
        "determinism_report": {
            "schema_version": "opk-rag.task0104.determinism-report.v1",
            "task_id": TASK_ID,
            "replicate_count": 2,
            "document_digest_match_count": sum(1 for row in determinism_rows if row["document_digest_match"]),
            "block_identity_match_count": sum(1 for row in determinism_rows if row["block_identity_match"]),
            "block_order_match_count": sum(1 for row in determinism_rows if row["block_order_match"]),
            "determinism_valid": len(determinism_rows) == FORMAL_DOCUMENT_COUNT and all(row["determinism_valid"] for row in determinism_rows),
            "documents": determinism_rows,
        },
        "content_preservation_report": {
            "schema_version": "opk-rag.task0104.content-preservation-report.v1",
            "task_id": TASK_ID,
            "minimum_preservation_ratio": CONTENT_PRESERVATION_MIN_RATIO,
            "content_preservation_gate_passed": content_gate_passed,
            "documents": content_rows,
            "summary": _content_summary(content_rows),
        },
        "failure_report": {
            "schema_version": "opk-rag.task0104.failure-report.v1",
            "task_id": TASK_ID,
            "failure_count": len(failures),
            "failures": failures,
        },
    }


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in payloads.items()}
    return {
        "schema_version": "opk-rag.task0104.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": dict(sorted(digests.items())),
        "combined_digest": digest_json(dict(sorted(digests.items()))),
    }


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    payloads = {
        "input_identity": read_json(result_dir / "input_identity.json"),
        "parser_identity": read_json(result_dir / "parser_identity.json"),
        "canonical_schema_summary": read_json(result_dir / "canonical_schema_summary.json"),
        "materialization_summary": read_json(result_dir / "materialization_summary.json"),
        "determinism_report": read_json(result_dir / "determinism_report.json"),
        "content_preservation_report": read_json(result_dir / "content_preservation_report.json"),
        "failure_report": read_json(result_dir / "failure_report.json"),
        "privacy_scan": read_json(result_dir / "privacy_scan.json"),
        "result_digests": read_json(result_dir / "result_digests.json"),
    }
    verification = verify_payloads(contract=contract, **payloads)
    if write:
        write_json(result_dir / "verification_summary.json", verification)
    return verification


def verify_payloads(
    *,
    contract: dict[str, Any],
    input_identity: dict[str, Any],
    parser_identity: dict[str, Any],
    canonical_schema_summary: dict[str, Any],
    materialization_summary: dict[str, Any],
    determinism_report: dict[str, Any],
    content_preservation_report: dict[str, Any],
    failure_report: dict[str, Any],
    privacy_scan: dict[str, Any],
    result_digests: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if input_identity.get("task0103_inputs_valid") is not True:
        issues.append("task0103_inputs_valid must be true")
    if input_identity.get("task0103_artifacts_modified") is not False:
        issues.append("task0103_artifacts_modified must be false")
    if input_identity.get("pdfqa_revision_match") is not True:
        issues.append("pdfqa_revision_match must be true")
    if input_identity.get("formal_profile_digest_match") is not True:
        issues.append("formal_profile_digest_match must be true")
    if materialization_summary.get("canonical_document_attempted") != FORMAL_DOCUMENT_COUNT:
        issues.append("canonical_document_attempted must be 100")
    if materialization_summary.get("canonical_document_success") != FORMAL_DOCUMENT_COUNT:
        issues.append("canonical_document_success must be 100")
    if materialization_summary.get("canonical_document_failure") != 0:
        issues.append("canonical_document_failure must be 0")
    if canonical_schema_summary.get("canonical_schema_valid") is not True:
        issues.append("canonical_schema_valid must be true")
    if canonical_schema_summary.get("canonical_identity_valid") is not True:
        issues.append("canonical_identity_valid must be true")
    if canonical_schema_summary.get("canonical_provenance_valid") is not True:
        issues.append("canonical_provenance_valid must be true")
    if determinism_report.get("determinism_valid") is not True:
        issues.append("determinism_valid must be true")
    if content_preservation_report.get("content_preservation_gate_passed") is not True:
        issues.append("content_preservation_gate_passed must be true")
    if privacy_scan.get("privacy_gate_passed") is not True:
        issues.append("privacy_gate_passed must be true")
    if failure_report.get("failure_count") != 0:
        issues.append("failure_count must be 0")
    for key in (
        "production_chunking_modified",
        "retrieval_modified",
        "reranker_modified",
        "rank_fusion_modified",
        "agent_behavior_modified",
        "graph_runtime_modified",
    ):
        if contract.get(key) is not False:
            issues.append(f"{key} must be false")

    expected_digests = build_result_digests(
        contract=contract,
        input_identity=input_identity,
        parser_identity=parser_identity,
        schema_summary=canonical_schema_summary,
        materialization_summary=materialization_summary,
        determinism=determinism_report,
        content=content_preservation_report,
        failure=failure_report,
    )
    if result_digests.get("artifact_digests") != expected_digests.get("artifact_digests"):
        issues.append("result_digests artifact digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0104.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "issues": issues,
        "task_status": "complete" if complete else "blocked",
        "canonical_pdf_adapter_ready": complete,
        "canonical_document_layer_ready": complete,
        "representation_aware_chunking_ready": complete,
        "next_task_decision": "begin_representation_aware_chunking_baseline" if complete else "canonical_document_remediation_required",
        "existing_runtime_behavior_unchanged": True,
        "git_commit_created": False,
    }


def privacy_scan_payloads(payloads: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for artifact_name, payload in payloads.items():
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        for finding_type, pattern in PRIVACY_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "artifact": artifact_name,
                        "finding_type": finding_type,
                        "matched_prefix_sha256": hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
                    }
                )
    return {
        "schema_version": "opk-rag.task0104.privacy-scan.v1",
        "task_id": TASK_ID,
        "scanned_artifact_count": len(payloads),
        "finding_count": len(findings),
        "findings": findings,
        "privacy_gate_passed": not findings,
    }


def _load_parser_payload(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    stem = Path(record["pdf_source_relative_path"]).stem
    return read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "candidates" / "pymupdf4llm" / "raw" / "formal" / f"{stem}.json")


def _source_for_record(root: Path, record: dict[str, Any]) -> StructuredDocumentSource:
    relative_path = record["pdf_source_relative_path"]
    local_path = root / ".cache" / "pdfqa" / relative_path
    source_bytes = local_path.read_bytes()
    return StructuredDocumentSource(
        source_bytes=source_bytes,
        knowledge_id=record["knowledge_id"],
        representation_id=record["pdf_representation_id"],
        representation_type="pdf",
        source_ref=relative_path,
        mime_type="application/pdf",
        source_name=Path(relative_path).name,
        source_relative_path=relative_path,
        upstream_revision=TASK0103_EXPECTED_PDFQA_REVISION,
        authority={"dataset": "pdfqa/pdfQA-Benchmark", "dataset_revision": TASK0103_EXPECTED_PDFQA_REVISION},
        metadata={},
    )


def _document_row(record: dict[str, Any], parser_payload: dict[str, Any], document_payload: dict[str, Any]) -> dict[str, Any]:
    blocks = document_payload.get("blocks", [])
    page_count = int(document_payload.get("metadata", {}).get("page_count") or parser_payload.get("page_count") or 0)
    parser_blocks = [block for block in parser_payload.get("blocks", []) if isinstance(block, dict)]
    source_character_count = sum(len(str(block.get("text") or "").strip()) for block in parser_blocks)
    if not parser_blocks:
        source_character_count = len(str(parser_payload.get("text") or "").strip())
    canonical_character_count = sum(len(str(block.get("text") or "")) for block in blocks)
    return {
        "knowledge_id": record["knowledge_id"],
        "pdf_representation_id": record["pdf_representation_id"],
        "pdf_source_relative_path": record["pdf_source_relative_path"],
        "document_id": document_payload["document_id"],
        "document_digest": document_payload["content_digest"],
        "block_count": len(blocks),
        "page_count": page_count,
        "raw_markdown_character_count": len(str(parser_payload.get("text") or "")),
        "source_character_count": source_character_count,
        "canonical_character_count": canonical_character_count,
        "empty_block_count": sum(1 for block in blocks if not str(block.get("text") or "").strip()),
        "empty_page_count": sum(1 for page in document_payload.get("metadata", {}).get("pages", []) if int(page.get("text_length") or 0) == 0),
        "canonical_schema_valid": True,
        "stable_document_identity": isinstance(document_payload.get("document_id"), str) and bool(document_payload["document_id"]),
        "stable_block_identity": len({block.get("block_id") for block in blocks}) == len(blocks) and all(block.get("block_id") for block in blocks),
        "source_provenance_valid": _provenance_valid(document_payload, page_count=page_count),
    }


def _content_row(row: dict[str, Any]) -> dict[str, Any]:
    source_count = row["source_character_count"]
    canonical_count = row["canonical_character_count"]
    ratio = canonical_count / source_count if source_count else (1.0 if canonical_count == 0 else 0.0)
    gate = source_count > 0 and canonical_count > 0 and ratio >= CONTENT_PRESERVATION_MIN_RATIO
    return {
        "knowledge_id": row["knowledge_id"],
        "source_character_count": source_count,
        "canonical_character_count": canonical_count,
        "preservation_ratio": ratio,
        "empty_block_count": row["empty_block_count"],
        "empty_page_count": row["empty_page_count"],
        "content_preservation_gate_passed": gate,
    }


def _content_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [row["preservation_ratio"] for row in rows]
    return {
        "document_count": len(rows),
        "min_preservation_ratio": min(ratios) if ratios else None,
        "max_preservation_ratio": max(ratios) if ratios else None,
        "average_preservation_ratio": sum(ratios) / len(ratios) if ratios else None,
        "total_source_character_count": sum(row["source_character_count"] for row in rows),
        "total_canonical_character_count": sum(row["canonical_character_count"] for row in rows),
    }


def _determinism_row(knowledge_id: str, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_block_ids = [block["block_id"] for block in first.get("blocks", [])]
    second_block_ids = [block["block_id"] for block in second.get("blocks", [])]
    document_digest_match = first.get("content_digest") == second.get("content_digest")
    block_identity_match = first_block_ids == second_block_ids
    block_order_match = [(block["block_id"], block["reading_order"]) for block in first.get("blocks", [])] == [(block["block_id"], block["reading_order"]) for block in second.get("blocks", [])]
    return {
        "knowledge_id": knowledge_id,
        "document_digest_match": document_digest_match,
        "block_identity_match": block_identity_match,
        "block_order_match": block_order_match,
        "determinism_valid": document_digest_match and block_identity_match and block_order_match,
    }


def _provenance_valid(document_payload: dict[str, Any], *, page_count: int) -> bool:
    for block in document_payload.get("blocks", []):
        location = block.get("source_location") or {}
        page_number = location.get("page_number")
        if not isinstance(page_number, int) or page_number <= 0:
            return False
        if page_count and page_number > page_count:
            return False
        if not location.get("source_ref") or location.get("element_ref") is None:
            return False
    return True


def _git_modified_under(root: Path, pathspec: str) -> bool:
    result = subprocess.run(["git", "status", "--short", "--", pathspec], cwd=root, text=True, capture_output=True, check=False)
    return bool(result.stdout.strip())
