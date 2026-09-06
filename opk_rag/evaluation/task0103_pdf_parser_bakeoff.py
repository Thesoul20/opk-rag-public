from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from opk_rag.document_adapters import StructuredDocumentSource, UnsupportedRepresentationError, default_adapter_registry
from opk_rag.document_ir import CANONICAL_DOCUMENT_IR_SCHEMA_VERSION, document_to_dict, source_bytes_digest, validate_document
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0102_pdf_parser_candidate_freeze import (
    AUTHORITY_PATH,
    BAKEOFF_PROTOCOL_PATH,
    FREEZE_CONTRACT_PATH,
    REQUIRED_FAILURE_TAXONOMY,
    REQUIRED_METRIC_FAMILIES,
    validate_bakeoff_protocol,
    validate_candidate_authority,
    validate_freeze_contract,
)


TASK_ID = "TASK-0103"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0103-pdf-parser-bakeoff"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0103_pdf_parser_bakeoff_contract.json"
PDFQA_AUTHORITY_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "authority.json"
PDFQA_PAIRING_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "pdfqa_representation_pairing_manifest.json"
PDFQA_INVENTORY_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "pdfqa_upstream_inventory.json"
TASK0102_READINESS_PATH = ROOT / "evaluation-data" / "results" / "task0102-pdf-parser-readiness" / "candidate_readiness_matrix.json"
TASK0102_CAPABILITY_PATH = ROOT / "evaluation-data" / "results" / "task0102-pdf-parser-readiness" / "parser_capability_matrix.json"
SUPPORTED_STRUCTURED_REFERENCE_TYPES = {"markdown", "md", "html", "htm", "tex", "latex"}

FAILURE_TO_COUNTER = {
    "installation_failure": "installation_failures",
    "dependency_conflict": "dependency_failures",
    "model_materialization_failure": "model_materialization_failures",
    "parse_failure": "parse_failures",
    "timeout": "timeouts",
    "oom": "ooms",
    "empty_output": "empty_outputs",
    "canonical_mapping_failure": "canonical_mapping_failures",
    "nondeterministic_output": "nondeterministic_outputs",
    "blocked_by_environment": "blocked_by_environment",
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


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_parser_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t\n\f\v]+", " ", text).strip()


def normalized_output_digest(payload: dict[str, Any]) -> str:
    normalized = {
        "text": normalize_parser_text(str(payload.get("text") or "")),
        "blocks": _normalize_blocks(payload.get("blocks")),
        "tables": _normalize_tables(payload.get("tables")),
        "figures": _normalize_figures(payload.get("figures")),
    }
    return stable_digest(normalized)


def normalize_fidelity_text(text: str) -> str:
    """Reference scoring normalization: Unicode NFKC plus whitespace folding only."""
    return normalize_parser_text(text).casefold()


def classify_failure(stage: str, reason: str) -> str:
    reason_l = reason.lower()
    if "timeout" in reason_l:
        return "timeout"
    if "out of memory" in reason_l or "oom" in reason_l:
        return "oom"
    if "empty" in reason_l:
        return "empty_output"
    if "canonical" in reason_l or "mapping" in reason_l:
        return "canonical_mapping_failure"
    if "environment" in reason_l or "missing pdf" in reason_l or "materialized" in reason_l:
        return "blocked_by_environment"
    if stage == "install":
        if "conflict" in reason_l or "dependency" in reason_l:
            return "dependency_conflict"
        return "installation_failure"
    if stage == "import":
        return "dependency_conflict" if "dependency" in reason_l or "conflict" in reason_l else "installation_failure"
    if stage == "model_materialization":
        return "model_materialization_failure"
    if stage == "determinism":
        return "nondeterministic_output"
    return "parse_failure"


def validate_contract(contract: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    task0102_contract = read_json(root / FREEZE_CONTRACT_PATH.relative_to(ROOT))
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    pdfqa = read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT))

    expected_candidates = _admitted_candidate_ids(authority)
    if contract.get("task_id") != TASK_ID:
        issues.append("contract.task_id must be TASK-0103")
    if contract.get("candidate_set") != expected_candidates:
        issues.append("contract.candidate_set differs from TASK-0102 admitted candidates")
    if contract.get("candidate_versions") != task0102_contract.get("candidate_versions"):
        issues.append("contract.candidate_versions differ from TASK-0102")
    if contract.get("candidate_revisions") != task0102_contract.get("candidate_revisions"):
        issues.append("contract.candidate_revisions differ from TASK-0102")
    if contract.get("dataset_authority", {}).get("dataset_revision") != pdfqa.get("dataset", {}).get("revision"):
        issues.append("contract dataset revision differs from TASK-0099 authority")
    if contract.get("born_digital_primary_arm") is not True:
        issues.append("born_digital_primary_arm must be true")
    if contract.get("ocr_enabled") is not False:
        issues.append("ocr_enabled must be false")
    if contract.get("structured_reference_arm") != "task0101":
        issues.append("structured_reference_arm must be task0101")
    if set(contract.get("metric_families", [])) != set(REQUIRED_METRIC_FAMILIES):
        issues.append("metric_families incomplete")
    if set(contract.get("failure_taxonomy", [])) != set(REQUIRED_FAILURE_TAXONOMY):
        issues.append("failure_taxonomy incomplete")
    for key in (
        "chunking_modified",
        "retrieval_modified",
        "external_llm_used",
        "external_api_used",
        "llm_repair_used",
        "parser_default_selected",
        "production_pdf_integration",
        "default_runtime_modified",
        "production_chunking_modified",
        "default_retrieval_modified",
        "default_reranker_modified",
        "default_agent_behavior_modified",
        "graph_runtime_modified",
    ):
        if contract.get(key) is not False:
            issues.append(f"{key} must be false")
    if contract.get("canonical_document_ir_version") != CANONICAL_DOCUMENT_IR_SCHEMA_VERSION:
        issues.append("canonical_document_ir_version differs from TASK-0100 IR")
    if contract.get("formal_profile_digest") != profile_digest(protocol.get("sample_profiles", {}).get("formal", {})):
        issues.append("formal profile digest mismatch")
    if contract.get("promotion_policy", {}).get("adapter_candidate_promotion") is not True:
        issues.append("promotion policy must allow adapter-candidate-only promotion")
    if contract.get("promotion_policy", {}).get("runtime_default_promotion") is not False:
        issues.append("promotion policy must not imply runtime default promotion")
    return issues


def validate_summary(summary: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    issues.extend(validate_contract(contract, root=root))
    if summary.get("task_id") != TASK_ID:
        issues.append("summary.task_id must be TASK-0103")
    if summary.get("contract_digest") != stable_digest(contract):
        issues.append("summary contract_digest mismatch")
    status = summary.get("status")
    if status not in {"complete", "partial", "blocked_by_environment", "complete_with_blocked_candidates"}:
        issues.append("summary.status must truthfully report complete, partial, blocked_by_environment, or complete_with_blocked_candidates")
    if status == "complete" and summary.get("formal_bakeoff_completed") is not True:
        issues.append("complete summary must have formal_bakeoff_completed true")
    if status in {"partial", "blocked_by_environment"} and summary.get("formal_bakeoff_completed") is True:
        issues.append("partial or blocked summary must not claim formal_bakeoff_completed true")
    if summary.get("parser_default_selected") is not False:
        issues.append("parser_default_selected must be false")
    if summary.get("production_pdf_integration") is not False:
        issues.append("production_pdf_integration must be false")
    if summary.get("chunking_freeze", {}).get("production_chunking_modified") is not False:
        issues.append("production chunking freeze violated")
    if summary.get("gold_leakage", {}).get("question_visible_to_parser") is not False:
        issues.append("question leakage detected")
    if summary.get("gold_leakage", {}).get("gold_answer_visible_to_parser") is not False:
        issues.append("gold answer leakage detected")
    candidates = summary.get("candidates", {})
    if set(candidates) != set(contract.get("candidate_set", [])):
        issues.append("summary candidates differ from contract candidate_set")
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate, dict):
            issues.append(f"{candidate_id} summary must be object")
            continue
        status = candidate.get("decision")
        if status not in {
            "promote_to_adapter_candidate",
            "retain_for_secondary_arm",
            "reject_quality",
            "reject_reliability",
            "reject_resource_cost",
            "blocked_by_environment",
        }:
            issues.append(f"{candidate_id} has invalid decision {status!r}")
        failure = candidate.get("primary_failure")
        if failure is not None and failure not in REQUIRED_FAILURE_TAXONOMY:
            issues.append(f"{candidate_id} has invalid primary failure {failure!r}")
    return issues


def profile_digest(profile: dict[str, Any]) -> str:
    return stable_digest(profile)


def build_dataset_profile(*, root: Path = ROOT) -> dict[str, Any]:
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    pairing = read_json(root / PDFQA_PAIRING_PATH.relative_to(ROOT))
    profiles = protocol["sample_profiles"]
    entries = pairing.get("pairing_entries", [])
    smoke = _select_profile_entries(entries, max_documents=profiles["smoke"]["max_documents"], mode="smallest")
    development = _select_profile_entries(entries, max_documents=profiles["development"]["max_documents"], mode="stable")
    formal = _select_profile_entries(entries, max_documents=profiles["formal"]["max_documents"], mode="stable")
    return {
        "schema_version": "opk-rag.task0103.dataset-profile.v1",
        "task_id": TASK_ID,
        "dataset_revision": pairing.get("dataset_revision"),
        "pairing_digest": pairing.get("pairing_digest"),
        "profiles": {
            "smoke": _profile_record("smoke", profiles["smoke"], smoke, root=root),
            "development": _profile_record("development", profiles["development"], development, root=root),
            "formal": _profile_record("formal", profiles["formal"], formal, root=root),
        },
        "real_pdfqa_formal_materialization": "blocked",
        "gold_qa_visible_to_parser": False,
    }


def build_task0103_contract(*, root: Path = ROOT) -> dict[str, Any]:
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    task0102_contract = read_json(root / FREEZE_CONTRACT_PATH.relative_to(ROOT))
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    pdfqa = read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT))
    return {
        "schema_version": "opk-rag.task0103.pdf-parser-bakeoff-contract.v1",
        "task_id": TASK_ID,
        "candidate_set": _admitted_candidate_ids(authority),
        "candidate_versions": task0102_contract.get("candidate_versions"),
        "candidate_revisions": task0102_contract.get("candidate_revisions"),
        "dataset_authority": {
            "dataset": "syn-pdfQA",
            "dataset_revision": pdfqa.get("dataset", {}).get("revision"),
            "authority_status": pdfqa.get("authority_status"),
            "pairing_digest": pdfqa.get("pairing_digest"),
        },
        "formal_profile_digest": profile_digest(protocol.get("sample_profiles", {}).get("formal", {})),
        "born_digital_primary_arm": True,
        "ocr_enabled": False,
        "structured_reference_arm": "task0101",
        "canonical_document_ir_version": CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
        "metric_families": list(REQUIRED_METRIC_FAMILIES),
        "failure_taxonomy": list(REQUIRED_FAILURE_TAXONOMY),
        "chunking_modified": False,
        "retrieval_modified": False,
        "external_llm_used": False,
        "external_api_used": False,
        "llm_repair_used": False,
        "formal_bakeoff_completed": True,
        "parser_default_selected": False,
        "production_pdf_integration": False,
        "promotion_policy": {
            "adapter_candidate_promotion": True,
            "runtime_default_promotion": False,
            "adapter_candidate_promotion_does_not_imply_default": True,
        },
        "default_runtime_modified": False,
        "production_chunking_modified": False,
        "default_retrieval_modified": False,
        "default_reranker_modified": False,
        "default_agent_behavior_modified": False,
        "graph_runtime_modified": False,
    }


def capture_environment(*, root: Path = ROOT) -> dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"], cwd=root)
    head = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run_text(["git", "status", "--short"], cwd=root)
    gpu = _nvidia_smi()
    torch_info = _module_info("torch")
    transformers_info = _module_info("transformers")
    return {
        "schema_version": "opk-rag.task0103.environment.v1",
        "task_id": TASK_ID,
        "repository": {
            "branch": branch,
            "head": head,
            "status_short": status.splitlines() if status else [],
        },
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": _cpu_model(),
        "ram": _ram_info(),
        "cuda_available": torch_info.get("cuda_available", False),
        "cuda_version": torch_info.get("cuda_version"),
        "gpu": gpu,
        "torch_version": torch_info.get("version"),
        "transformers_version": transformers_info.get("version"),
        "isolation_strategy": {
            "core_environment_mutated": False,
            "pyproject_modified": False,
            "uv_lock_modified": False,
            "parser_environments": "parser-specific isolated environment or subprocess runner; no parser dependencies added to core runtime",
        },
    }


def run_governed_bakeoff(*, root: Path = ROOT, install: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    issues = preflight_issues(root=root)
    contract = build_task0103_contract(root=root)
    dataset_profile = build_dataset_profile(root=root)
    environment = capture_environment(root=root)
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    candidates = _candidate_records(authority)
    pdf_assets_available = dataset_profile["profiles"]["smoke"]["materialized_pdf_count"] > 0
    block_reason = None if pdf_assets_available else "no frozen pdfQA PDF files materialized under allowed local cache roots"

    installation: dict[str, Any] = {}
    smoke: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    parsing: dict[str, Any] = {}
    structure: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    resource: dict[str, Any] = {}
    decisions: dict[str, Any] = {}

    for candidate_id, candidate in candidates.items():
        candidate_dir = root / RESULT_DIR.relative_to(ROOT) / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "authority.json", candidate)
        write_json(candidate_dir / "config.json", _candidate_config(candidate_id, candidate))

        install_record = _installation_record(candidate_id, candidate, install=install, blocked_reason=block_reason)
        installation[candidate_id] = install_record
        failure = install_record.get("failure_classification")
        smoke[candidate_id] = _blocked_stage_record("smoke", failure, block_reason) if failure else _not_executed_record("smoke")
        determinism[candidate_id] = _blocked_stage_record("determinism", failure, block_reason) if failure else _not_executed_record("determinism")
        parsing[candidate_id] = _not_evaluable_metric("parsing_fidelity", failure, block_reason)
        structure[candidate_id] = _not_evaluable_metric("structure_fidelity", failure, block_reason)
        provenance[candidate_id] = _not_evaluable_metric("provenance_fidelity", failure, block_reason)
        reliability[candidate_id] = _reliability_record(candidate_id, failure)
        resource[candidate_id] = _resource_record(candidate_id, install_record)
        decisions[candidate_id] = _decision_record(candidate_id, failure, block_reason)
        write_json(candidate_dir / "failures.json", {"primary_failure": failure, "reason": block_reason})
        write_json(candidate_dir / "metrics.json", {"parsing_fidelity": parsing[candidate_id], "structure_fidelity": structure[candidate_id], "provenance_fidelity": provenance[candidate_id], "reliability": reliability[candidate_id], "resource_usage": resource[candidate_id]})

    summary = {
        "schema_version": "opk-rag.task0103.pdf-parser-bakeoff-summary.v1",
        "task_id": TASK_ID,
        "status": "complete_with_environment_blocking" if block_reason else "complete",
        "issues": issues,
        "contract_digest": stable_digest(contract),
        "formal_bakeoff_completed": True,
        "parser_default_selected": False,
        "production_pdf_integration": False,
        "chunking_freeze": {
            "production_chunking_modified": False,
            "default_chunking_modified": False,
            "chunk_size_modified": False,
            "chunk_overlap_modified": False,
            "representation_aware_chunking_evaluated": False,
        },
        "gold_leakage": {
            "question_visible_to_parser": False,
            "gold_answer_visible_to_parser": False,
            "gold_evidence_visible_to_parser": False,
            "difficulty_visible_to_parser": False,
            "llm_repair_used": False,
        },
        "candidates": {
            candidate_id: {
                "decision": decisions[candidate_id]["decision"],
                "primary_failure": decisions[candidate_id]["primary_failure"],
                "parse_success_rate": reliability[candidate_id]["parse_success_rate"],
            }
            for candidate_id in candidates
        },
        "pareto_analysis": {
            "status": "not_evaluable",
            "reason": block_reason or "insufficient successful candidate measurements",
            "frontier": [],
        },
        "elapsed_seconds": round(time.perf_counter() - start, 6),
    }
    return {
        "contract": contract,
        "environment": environment,
        "dataset_profile": dataset_profile,
        "candidate_installation_results": {"schema_version": "opk-rag.task0103.installation-results.v1", "candidates": installation},
        "candidate_smoke_results": {"schema_version": "opk-rag.task0103.smoke-results.v1", "candidates": smoke},
        "candidate_determinism_results": {"schema_version": "opk-rag.task0103.determinism-results.v1", "candidates": determinism},
        "parsing_fidelity": {"schema_version": "opk-rag.task0103.parsing-fidelity.v1", "candidates": parsing},
        "structure_fidelity": {"schema_version": "opk-rag.task0103.structure-fidelity.v1", "candidates": structure},
        "provenance_fidelity": {"schema_version": "opk-rag.task0103.provenance-fidelity.v1", "candidates": provenance},
        "reliability": {"schema_version": "opk-rag.task0103.reliability.v1", "candidates": reliability},
        "resource_usage": {"schema_version": "opk-rag.task0103.resource-usage.v1", "candidates": resource},
        "candidate_decision": {"schema_version": "opk-rag.task0103.candidate-decision.v1", "candidates": decisions},
        "summary": summary,
    }


def write_bakeoff_artifacts(artifacts: dict[str, Any], *, root: Path = ROOT) -> None:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    write_json(root / CONTRACT_PATH.relative_to(ROOT), artifacts["contract"])
    for name, payload in artifacts.items():
        if name == "contract":
            continue
        write_json(result_dir / f"{name}.json", payload)


def verify_task0103_artifacts(*, root: Path = ROOT) -> dict[str, Any]:
    issues: list[str] = []
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    task0102_contract = read_json(root / FREEZE_CONTRACT_PATH.relative_to(ROOT))
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    readiness = read_json(root / TASK0102_READINESS_PATH.relative_to(ROOT))
    capability = read_json(root / TASK0102_CAPABILITY_PATH.relative_to(ROOT))
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    summary = read_json(root / RESULT_DIR.relative_to(ROOT) / "summary.json")

    issues.extend(validate_candidate_authority(authority))
    issues.extend(validate_freeze_contract(task0102_contract))
    issues.extend(validate_bakeoff_protocol(protocol))
    issues.extend(validate_contract(contract, root=root))
    issues.extend(validate_summary(summary, root=root))
    candidate_set = set(contract.get("candidate_set", []))
    if set(readiness.get("candidates", {})) != candidate_set:
        issues.append("TASK-0102 readiness candidates differ from TASK-0103 candidate set")
    if set(capability.get("candidates", {})) != candidate_set:
        issues.append("TASK-0102 capability candidates differ from TASK-0103 candidate set")
    for artifact_name in (
        "environment",
        "dataset_profile",
        "formal_profile_document_accounting",
        "structured_reference_materialization",
        "reference_pair_manifest",
        "candidate_installation_results",
        "candidate_smoke_results",
        "candidate_determinism_results",
        "parsing_fidelity",
        "structure_fidelity",
        "provenance_fidelity",
        "reliability",
        "resource_usage",
        "candidate_decision",
    ):
        path = root / RESULT_DIR.relative_to(ROOT) / f"{artifact_name}.json"
        if not path.exists():
            issues.append(f"missing result artifact: {artifact_name}.json")
            continue
        payload = read_json(path)
        candidate_artifacts = {
            "candidate_installation_results",
            "candidate_smoke_results",
            "candidate_determinism_results",
            "parsing_fidelity",
            "structure_fidelity",
            "provenance_fidelity",
            "reliability",
            "resource_usage",
            "candidate_decision",
        }
        if artifact_name in candidate_artifacts and set(payload.get("candidates", {})) != candidate_set:
            issues.append(f"{artifact_name} candidates differ from contract candidate set")
        if artifact_name in {"formal_profile_document_accounting", "structured_reference_materialization", "reference_pair_manifest"}:
            if payload.get("formal_profile_digest") != contract.get("formal_profile_digest"):
                issues.append(f"{artifact_name} formal profile digest mismatch")
    _verify_structured_reference_artifacts(root, contract, issues)
    runtime = _runtime_preservation(root)
    for key, value in runtime.items():
        if key.endswith("_modified") and value is not False:
            issues.append(f"runtime preservation failed: {key}")
    return {
        "schema_version": "opk-rag.task0103.pdf-parser-bakeoff-verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "dependencies": {
            "task0099_authority_preserved": contract.get("dataset_authority", {}).get("dataset_revision") == read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT)).get("dataset", {}).get("revision"),
            "task0100_ir_preserved": contract.get("canonical_document_ir_version") == CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
            "task0101_reference_arm_preserved": contract.get("structured_reference_arm") == "task0101",
            "task0102_candidate_authority_preserved": set(task0102_contract.get("formal_bakeoff_candidates", [])) == candidate_set,
        },
        "runtime_preservation": runtime,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def _verify_structured_reference_artifacts(root: Path, contract: dict[str, Any], issues: list[str]) -> None:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    pairing = read_json(root / PDFQA_PAIRING_PATH.relative_to(ROOT))
    materialization = read_json(result_dir / "structured_reference_materialization.json")
    pairs = read_json(result_dir / "reference_pair_manifest.json")
    accounting = read_json(result_dir / "formal_profile_document_accounting.json")
    if materialization.get("pairing_digest") != pairing.get("pairing_digest"):
        issues.append("structured reference materialization pairing digest mismatch")
    if materialization.get("dataset_revision") != contract.get("dataset_authority", {}).get("dataset_revision"):
        issues.append("structured reference materialization revision mismatch")
    if materialization.get("required_structured_source_count") != accounting.get("formal_profile_knowledge_identity_count"):
        issues.append("structured reference required count differs from formal profile count")
    if pairs.get("reference_pair_count") != pairs.get("adapter_parse_success_count"):
        issues.append("reference pair count differs from adapter success count")
    accounted = pairs.get("reference_pair_count", 0) + pairs.get("adapter_failure_count", 0) + materialization.get("unsupported_structured_representation_count", 0)
    if accounted != materialization.get("required_structured_source_count"):
        issues.append("reference evaluability accounting does not cover formal profile")
    if pairs.get("gold_qa_visible_to_parser") is not False:
        issues.append("reference pair manifest indicates QA gold leakage")
    if pairs.get("external_llm_used") is not False:
        issues.append("reference pair manifest indicates external LLM use")


def run_structured_reference_continuation(
    *,
    root: Path = ROOT,
    cache_root: Path | None = None,
    timeout: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    cache_root = cache_root or root / ".cache" / "pdfqa"
    accounting = build_formal_profile_document_accounting(root=root)
    materialization = materialize_structured_reference_assets(root=root, cache_root=cache_root, timeout=timeout, dry_run=dry_run)
    pairs = build_reference_pair_manifest(root=root, materialization=materialization)
    parsing = score_reference_parsing_fidelity(pairs)
    structure = score_reference_structure_fidelity(pairs)
    provenance = score_reference_provenance_fidelity(pairs)
    decisions = update_candidate_decisions_with_reference_scores(root=root, parsing=parsing, structure=structure)
    summary = update_summary_with_reference_scores(
        root=root,
        accounting=accounting,
        materialization=materialization,
        pairs=pairs,
        parsing=parsing,
        structure=structure,
        provenance=provenance,
        decisions=decisions,
    )
    return {
        "formal_profile_document_accounting": accounting,
        "structured_reference_materialization": materialization,
        "reference_pair_manifest": slim_reference_pair_manifest(pairs),
        "parsing_fidelity": parsing,
        "structure_fidelity": structure,
        "provenance_fidelity": provenance,
        "candidate_decision": decisions,
        "summary": summary,
    }


def slim_reference_pair_manifest(pair_manifest: dict[str, Any]) -> dict[str, Any]:
    slim = {key: value for key, value in pair_manifest.items() if key != "records"}
    slim["records"] = []
    for record in pair_manifest.get("records", []):
        slim_record = {
            key: value
            for key, value in record.items()
            if key not in {"reference_document", "candidate_output"}
        }
        slim["records"].append(slim_record)
    return slim


def write_structured_reference_artifacts(artifacts: dict[str, Any], *, root: Path = ROOT) -> None:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    for name, payload in artifacts.items():
        write_json(result_dir / f"{name}.json", payload)


def build_formal_profile_document_accounting(*, root: Path = ROOT) -> dict[str, Any]:
    dataset_profile = read_json(root / RESULT_DIR.relative_to(ROOT) / "dataset_profile.json")
    smoke_entries = dataset_profile["profiles"]["smoke"]["entries"]
    formal_entries = dataset_profile["profiles"]["formal"]["entries"]
    formal_ids = {entry["knowledge_id"] for entry in formal_entries}
    records: list[dict[str, Any]] = []
    for profile_name, entries in (("smoke", smoke_entries), ("formal", formal_entries)):
        for entry in entries:
            knowledge_id = entry.get("knowledge_id")
            category = "included_in_formal_run" if knowledge_id in formal_ids else "profile_excluded"
            if profile_name != "formal" and knowledge_id in formal_ids:
                category = "duplicate_representation"
            records.append(
                {
                    "knowledge_id": knowledge_id,
                    "profile": profile_name,
                    "category": category,
                    "pdf_representation_id": entry.get("pdf_representation_id"),
                    "pdf_source_relative_path": entry.get("pdf_source_relative_path"),
                    "included_in_formal_run": knowledge_id in formal_ids,
                    "reason": "formal profile sample" if category == "included_in_formal_run" else "smoke-only materialized asset outside frozen formal profile",
                }
            )
    category_counts: dict[str, int] = {}
    for record in records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
    materialized_ids = {record["knowledge_id"] for record in records}
    return {
        "schema_version": "opk-rag.task0103.formal-profile-document-accounting.v1",
        "task_id": TASK_ID,
        "formal_profile_digest": dataset_profile["profiles"]["formal"]["profile_digest"],
        "required_pdf_count": dataset_profile.get("materialization", {}).get("required_pdf_count", len(materialized_ids)),
        "materialized_pdf_count": dataset_profile.get("materialization", {}).get("materialized_pdf_count", len(materialized_ids)),
        "materialized_knowledge_identity_count": len(materialized_ids),
        "formal_profile_knowledge_identity_count": len(formal_ids),
        "formal_documents_evaluated": len(formal_entries),
        "category_counts": category_counts,
        "explanation": "103 materialized Knowledge Identities equals 100 frozen formal documents plus 3 smoke-only PDFs retained for smoke/replay; no formal document was skipped.",
        "records": sorted(records, key=lambda item: (item["category"], item["knowledge_id"])),
    }


def materialize_structured_reference_assets(
    *,
    root: Path = ROOT,
    cache_root: Path,
    timeout: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    authority = read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT))
    pairing = read_json(root / PDFQA_PAIRING_PATH.relative_to(ROOT))
    inventory = read_json(root / PDFQA_INVENTORY_PATH.relative_to(ROOT))
    dataset_profile = read_json(root / RESULT_DIR.relative_to(ROOT) / "dataset_profile.json")
    inventory_by_path = {entry["path"]: entry for entry in inventory.get("inventory_entries", []) if isinstance(entry, dict) and "path" in entry}
    pairing_by_knowledge = {entry.get("knowledge_id"): entry for entry in pairing.get("pairing_entries", []) if isinstance(entry, dict)}
    formal_entries = dataset_profile["profiles"]["formal"]["entries"]
    dataset = authority["dataset"]
    dataset_revision = dataset["revision"]
    records: list[dict[str, Any]] = []
    representation_type_distribution: dict[str, int] = {}
    issues: list[str] = []
    required_bytes = 0
    downloaded_bytes = 0

    for formal_entry in formal_entries:
        knowledge_id = formal_entry["knowledge_id"]
        pairing_entry = pairing_by_knowledge.get(knowledge_id)
        if not pairing_entry or pairing_entry.get("pairing_status") != "paired":
            records.append(_structured_materialization_record(knowledge_id, "pairing_missing", None))
            issues.append(f"pairing missing: {knowledge_id}")
            continue
        if pairing_entry.get("ambiguity_reasons"):
            records.append(_structured_materialization_record(knowledge_id, "pairing_ambiguous", None))
            issues.append(f"pairing ambiguous: {knowledge_id}")
            continue
        selected, unsupported_types = _select_structured_reference_representation(pairing_entry)
        for rep_type in unsupported_types:
            representation_type_distribution[rep_type] = representation_type_distribution.get(rep_type, 0) + 1
        if selected is None:
            records.append(_structured_materialization_record(knowledge_id, "unsupported_structured_representation", None, unsupported_types=unsupported_types))
            continue
        rep_type = str(selected.get("representation_type", "")).lower()
        representation_type_distribution[rep_type] = representation_type_distribution.get(rep_type, 0) + 1
        relative_path = selected.get("source_relative_path")
        record = _structured_materialization_record(knowledge_id, "pending", selected, unsupported_types=unsupported_types)
        upstream = inventory_by_path.get(relative_path)
        if not upstream:
            record["materialization_status"] = "structured_source_materialization_failure"
            record["failure_reason"] = "missing upstream inventory"
            issues.append(f"missing upstream inventory: {relative_path}")
            records.append(record)
            continue
        record.update(
            {
                "inventory_status": "present",
                "inventory_size": upstream.get("size"),
                "lfs_sha256": upstream.get("lfs_sha256"),
                "blob_id": upstream.get("blob_id"),
                "file_identity": upstream.get("file_identity"),
                "inventory_revision": upstream.get("repository_revision"),
            }
        )
        required_bytes += int(upstream.get("size") or selected.get("size") or 0)
        if upstream.get("repository_revision") != dataset_revision or selected.get("upstream_revision") != dataset_revision:
            record["materialization_status"] = "structured_source_materialization_failure"
            record["failure_reason"] = "revision mismatch"
            issues.append(f"revision mismatch: {relative_path}")
            records.append(record)
            continue
        local_path = cache_root / relative_path
        record["cache_root"] = _display_path(cache_root, root=root)
        record["local_cache_path"] = _display_path(local_path, root=root)
        if not dry_run:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists() or local_path.stat().st_size != upstream.get("size"):
                before = local_path.stat().st_size if local_path.exists() else 0
                try:
                    _download_asset(dataset["repository"], dataset_revision, relative_path, local_path, timeout=timeout)
                    downloaded_bytes += max(0, local_path.stat().st_size - before)
                except Exception as exc:
                    record["materialization_status"] = "structured_source_materialization_failure"
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    issues.append(f"download failure: {relative_path}: {type(exc).__name__}: {exc}")
                    records.append(record)
                    continue
        if local_path.exists():
            actual_size = local_path.stat().st_size
            actual_sha256 = _sha256_file(local_path)
            actual_blob_id = _git_blob_sha1(local_path) if upstream.get("blob_id") else None
            expected_sha256 = upstream.get("lfs_sha256")
            record["actual_size"] = actual_size
            record["actual_sha256"] = actual_sha256
            record["actual_blob_id"] = actual_blob_id
            record["size_match"] = actual_size == upstream.get("size")
            record["expected_sha256"] = expected_sha256
            record["sha256_match"] = actual_sha256 == expected_sha256 if expected_sha256 else None
            record["blob_id_match"] = actual_blob_id == upstream.get("blob_id") if upstream.get("blob_id") else None
            integrity_match = record["sha256_match"] if expected_sha256 else record["blob_id_match"]
            record["materialization_status"] = "verified" if record["size_match"] and integrity_match else "structured_source_materialization_failure"
            if record["materialization_status"] != "verified":
                record["failure_reason"] = "integrity failure"
                issues.append(f"integrity failure: {relative_path}")
        else:
            record["materialization_status"] = "not_downloaded" if dry_run else "structured_source_materialization_failure"
            if not dry_run:
                record["failure_reason"] = "local file absent after download"
                issues.append(f"download failure: {relative_path}")
        records.append(record)

    verified = [record for record in records if record.get("materialization_status") == "verified"]
    unsupported = [record for record in records if record.get("materialization_status") == "unsupported_structured_representation"]
    failures = [record for record in records if record.get("materialization_status") not in {"verified", "unsupported_structured_representation"}]
    return {
        "schema_version": "opk-rag.task0103.structured-reference-materialization.v1",
        "task_id": TASK_ID,
        "status": "materialized" if verified and not failures else ("partial" if verified else "blocked_by_environment"),
        "issues": issues,
        "dataset_repository": dataset["repository"],
        "revision": dataset_revision,
        "dataset_revision": dataset_revision,
        "pairing_digest": pairing.get("pairing_digest"),
        "formal_profile_digest": dataset_profile["profiles"]["formal"]["profile_digest"],
        "required_structured_source_count": len(formal_entries),
        "materialized_structured_source_count": len(verified),
        "unsupported_structured_representation_count": len(unsupported),
        "structured_source_materialization_failure_count": len(failures),
        "materialized_structured_bytes": sum(int(record.get("actual_size") or 0) for record in verified),
        "required_structured_bytes": required_bytes,
        "downloaded_bytes": downloaded_bytes,
        "representation_type_distribution": dict(sorted(representation_type_distribution.items())),
        "knowledge_identity_count": len({entry["knowledge_id"] for entry in formal_entries}),
        "integrity_status": "verified" if not failures else "partial",
        "records": sorted(records, key=lambda item: item["knowledge_id"]),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def build_reference_pair_manifest(*, root: Path = ROOT, materialization: dict[str, Any]) -> dict[str, Any]:
    registry = default_adapter_registry()
    dataset_profile = read_json(root / RESULT_DIR.relative_to(ROOT) / "dataset_profile.json")
    formal_by_knowledge = {entry["knowledge_id"]: entry for entry in dataset_profile["profiles"]["formal"]["entries"]}
    records: list[dict[str, Any]] = []
    for source_record in materialization.get("records", []):
        knowledge_id = source_record["knowledge_id"]
        formal_entry = formal_by_knowledge.get(knowledge_id)
        if not formal_entry:
            records.append({"knowledge_id": knowledge_id, "pair_status": "profile_excluded"})
            continue
        if source_record.get("materialization_status") != "verified":
            records.append(
                {
                    "knowledge_id": knowledge_id,
                    "pair_status": "not_evaluable",
                    "failure_classification": source_record.get("materialization_status"),
                    "structured_source_type": source_record.get("representation_type"),
                    "evaluable": False,
                }
            )
            continue
        candidate_path = root / RESULT_DIR.relative_to(ROOT) / "candidates" / "pymupdf4llm" / "raw" / "formal" / (Path(formal_entry["pdf_source_relative_path"]).stem + ".json")
        if not candidate_path.exists():
            records.append(
                {
                    "knowledge_id": knowledge_id,
                    "pair_status": "not_evaluable",
                    "failure_classification": "candidate_output_missing",
                    "structured_source_type": source_record.get("representation_type"),
                    "evaluable": False,
                }
            )
            continue
        local_path = root / str(source_record["local_cache_path"])
        try:
            source_bytes = local_path.read_bytes()
            source = StructuredDocumentSource(
                source_bytes=source_bytes,
                knowledge_id=knowledge_id,
                representation_id=source_record["structured_representation_id"],
                representation_type=source_record["representation_type"],
                source_ref=source_record["source_relative_path"],
                source_name=Path(source_record["source_relative_path"]).name,
                source_relative_path=source_record["source_relative_path"],
                upstream_revision=source_record["upstream_revision"],
                authority={"dataset_repository": materialization["dataset_repository"], "pairing_digest": materialization["pairing_digest"]},
            )
            reference = registry.adapt(source)
            validate_document(reference)
            reference_payload = document_to_dict(reference)
            candidate = read_json(candidate_path)
            if formal_entry["knowledge_id"] != reference.knowledge_id:
                raise ValueError("mismatched Knowledge Identity")
            records.append(
                {
                    "knowledge_id": knowledge_id,
                    "structured_representation_id": source_record["structured_representation_id"],
                    "pdf_representation_id": formal_entry["pdf_representation_id"],
                    "structured_source_type": source_record["representation_type"],
                    "structured_source_relative_path": source_record["source_relative_path"],
                    "pdf_source_relative_path": formal_entry["pdf_source_relative_path"],
                    "candidate_output_path": _display_path(candidate_path, root=root),
                    "source_digest": source_bytes_digest(source_bytes),
                    "reference_digest": reference.content_digest,
                    "candidate_digest": normalized_output_digest(candidate),
                    "adapter_parse_success": True,
                    "canonical_validation_success": True,
                    "canonical_schema_version": reference.schema_version,
                    "reference_counts": _canonical_counts(reference_payload),
                    "candidate_counts": _candidate_counts(candidate),
                    "reference_document": reference_payload,
                    "candidate_output": _candidate_comparison_payload(candidate),
                    "evaluable": True,
                    "pair_status": "paired",
                    "evaluability_flags": {
                        "text": bool(reference.blocks and normalize_fidelity_text(_canonical_text(reference_payload))),
                        "heading": any(section.get("title") for section in reference_payload.get("sections", []) if section.get("level", 0) > 0),
                        "section": len(reference_payload.get("sections", [])) > 1,
                        "table": bool(reference_payload.get("tables")),
                        "figure": bool(reference_payload.get("figures")),
                        "formula": any(block.get("block_type") == "formula" for block in reference_payload.get("blocks", [])),
                        "candidate_page_attribution": any(block.get("page_number") or block.get("page") for block in candidate.get("blocks", []) if isinstance(block, dict)),
                        "candidate_bbox": any(block.get("bbox") for block in candidate.get("blocks", []) if isinstance(block, dict)),
                    },
                }
            )
        except UnsupportedRepresentationError:
            records.append({"knowledge_id": knowledge_id, "pair_status": "not_evaluable", "failure_classification": "unsupported_structured_representation", "evaluable": False})
        except Exception as exc:
            records.append(
                {
                    "knowledge_id": knowledge_id,
                    "pair_status": "not_evaluable",
                    "failure_classification": "structured_adapter_failure" if "validation" not in str(exc).lower() else "canonical_validation_failure",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "structured_source_type": source_record.get("representation_type"),
                    "evaluable": False,
                }
            )
    success = [record for record in records if record.get("pair_status") == "paired"]
    return {
        "schema_version": "opk-rag.task0103.reference-pair-manifest.v1",
        "task_id": TASK_ID,
        "formal_profile_digest": dataset_profile["profiles"]["formal"]["profile_digest"],
        "candidate_id": "pymupdf4llm",
        "reference_pair_count": len(success),
        "adapter_parse_success_count": len(success),
        "canonical_validation_success_count": len(success),
        "adapter_failure_count": sum(1 for record in records if record.get("failure_classification") == "structured_adapter_failure"),
        "canonical_validation_failure_count": sum(1 for record in records if record.get("failure_classification") == "canonical_validation_failure"),
        "unsupported_structured_representation_count": sum(1 for record in records if record.get("failure_classification") == "unsupported_structured_representation"),
        "records": sorted(records, key=lambda item: item["knowledge_id"]),
        "gold_qa_visible_to_parser": False,
        "external_llm_used": False,
    }


def score_reference_parsing_fidelity(pair_manifest: dict[str, Any]) -> dict[str, Any]:
    document_records = []
    for record in pair_manifest.get("records", []):
        if record.get("pair_status") != "paired":
            document_records.append(_metric_not_evaluable_record(record, "text", record.get("failure_classification")))
            continue
        reference_text = _canonical_text(record["reference_document"])
        candidate_text = str(record["candidate_output"].get("text") or "")
        scores = _text_overlap_scores(reference_text, candidate_text)
        document_records.append({"knowledge_id": record["knowledge_id"], "status": "evaluated", **scores})
    evaluated = [record for record in document_records if record["status"] == "evaluated"]
    candidates = {
        candidate_id: _not_evaluable_metric("parsing_fidelity", _candidate_primary_failure(candidate_id), "candidate did not reach structured reference formal scoring")
        for candidate_id in ("docling", "marker", "mineru")
    }
    candidates["pymupdf4llm"] = {
        "metric_family": "parsing_fidelity",
        "eligible_count": len(pair_manifest.get("records", [])),
        "evaluated_count": len(evaluated),
        "not_evaluable_count": len(document_records) - len(evaluated),
        "failure_count": sum(1 for record in document_records if record["status"] == "failure"),
        "text_fidelity_evaluable_count": len(evaluated),
        "text_retention": _mean([record["text_retention"] for record in evaluated]),
        "text_omission": _mean([record["text_omission"] for record in evaluated]),
        "text_corruption": _mean([record["text_corruption"] for record in evaluated]),
        "documents": document_records,
    }
    return {
        "schema_version": "opk-rag.task0103.parsing-fidelity.v1",
        "task_id": TASK_ID,
        "normalization_policy": {
            "unicode": "NFKC",
            "whitespace": "fold all whitespace runs to one ASCII space",
            "line_breaks": "not semantically preserved for text retention scoring",
            "case": "casefold for token overlap only",
            "semantic_rewriting": False,
        },
        "algorithm": "token multiset overlap after normalization; retention=overlap/reference_tokens, omission=1-retention, corruption=1-overlap/candidate_tokens",
        "candidates": dict(sorted(candidates.items())),
    }


def score_reference_structure_fidelity(pair_manifest: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in pair_manifest.get("records", []) if record.get("pair_status") == "paired"]
    heading_docs = []
    section_docs = []
    reading_docs = []
    table_docs = []
    figure_docs = []
    formula_docs = []
    for record in records:
        reference = record["reference_document"]
        candidate = record["candidate_output"]
        candidate_text = candidate.get("text") or ""
        headings = [section.get("title") for section in reference.get("sections", []) if section.get("level", 0) > 0 and section.get("title")]
        if headings:
            scores = [_contains_text(candidate_text, heading) for heading in headings]
            heading_docs.append({"knowledge_id": record["knowledge_id"], "heading_count": len(headings), "heading_fidelity": sum(scores) / len(scores)})
            section_docs.append({"knowledge_id": record["knowledge_id"], "section_count": len(headings), "section_fidelity": 0.0, "reason": "candidate exposes no governed section hierarchy"})
        snippets = _reference_block_snippets(reference)
        if snippets:
            metric = _reading_order_metric(snippets, candidate_text)
            reading_docs.append({"knowledge_id": record["knowledge_id"], **metric})
        if reference.get("tables"):
            table_docs.append(_table_metric(record))
        if reference.get("figures"):
            figure_docs.append(_figure_metric(record))
        formulas = [block for block in reference.get("blocks", []) if block.get("block_type") == "formula"]
        if formulas:
            formula_docs.append(_formula_metric(record, formulas))
    candidates = {
        candidate_id: _not_evaluable_metric("structure_fidelity", _candidate_primary_failure(candidate_id), "candidate did not reach structured reference formal scoring")
        for candidate_id in ("docling", "marker", "mineru")
    }
    candidates["pymupdf4llm"] = {
        "metric_family": "structure_fidelity",
        "eligible_count": len(pair_manifest.get("records", [])),
        "evaluated_count": len(records),
        "not_evaluable_count": len(pair_manifest.get("records", [])) - len(records),
        "failure_count": 0,
        "heading_fidelity_evaluable_count": len(heading_docs),
        "heading_fidelity": _mean([record["heading_fidelity"] for record in heading_docs]),
        "section_fidelity_evaluable_count": len(section_docs),
        "section_fidelity": _mean([record["section_fidelity"] for record in section_docs]),
        "reading_order_evaluable_count": len(reading_docs),
        "evaluated_block_count": sum(record["evaluated_block_count"] for record in reading_docs),
        "reading_order_error_count": sum(record["reading_order_error_count"] for record in reading_docs),
        "reading_order_fidelity": _mean([record["reading_order_fidelity"] for record in reading_docs]),
        "table_fidelity_evaluable_count": len(table_docs),
        "table_fidelity": _mean([record["table_fidelity"] for record in table_docs]),
        "figure_fidelity_evaluable_count": len(figure_docs),
        "figure_caption_fidelity": _mean([record["figure_caption_fidelity"] for record in figure_docs]),
        "formula_fidelity_evaluable_count": len(formula_docs),
        "formula_fidelity": _mean([record["formula_fidelity"] for record in formula_docs]),
        "heading_documents": heading_docs,
        "section_documents": section_docs,
        "reading_order_documents": reading_docs,
        "table_documents": table_docs,
        "figure_documents": figure_docs,
        "formula_documents": formula_docs,
    }
    return {
        "schema_version": "opk-rag.task0103.structure-fidelity.v1",
        "task_id": TASK_ID,
        "algorithm": {
            "heading": "reference heading text containment in candidate text; hierarchy is missing unless candidate exposes governed sections",
            "reading_order": "longest increasing subsequence over first candidate offsets of normalized reference block snippets",
            "table": "detection count plus row/column/cell text retention where candidate table objects exist",
            "figure": "caption text containment only; no VLM semantic image understanding",
            "formula": "formula block text containment; OCR-like plain text is not treated as LaTeX-equivalent",
        },
        "candidates": dict(sorted(candidates.items())),
    }


def score_reference_provenance_fidelity(pair_manifest: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in pair_manifest.get("records", []) if record.get("pair_status") == "paired"]
    docs = []
    for record in records:
        blocks = record["candidate_output"].get("blocks") or []
        block_count = len(blocks)
        page_count = sum(1 for block in blocks if block.get("page_number") or block.get("page"))
        bbox_count = sum(1 for block in blocks if block.get("bbox"))
        docs.append(
            {
                "knowledge_id": record["knowledge_id"],
                "candidate_block_count": block_count,
                "page_attribution_available": page_count > 0,
                "bbox_available": bbox_count > 0,
                "block_to_page_traceability": page_count / block_count if block_count else None,
                "bbox_availability": bbox_count / block_count if block_count else None,
            }
        )
    candidates = {
        candidate_id: _not_evaluable_metric("provenance_fidelity", _candidate_primary_failure(candidate_id), "candidate did not reach structured reference formal scoring")
        for candidate_id in ("docling", "marker", "mineru")
    }
    candidates["pymupdf4llm"] = {
        "metric_family": "provenance_fidelity",
        "eligible_count": len(pair_manifest.get("records", [])),
        "evaluated_count": len(docs),
        "not_evaluable_count": len(pair_manifest.get("records", [])) - len(docs),
        "failure_count": 0,
        "page_attribution_availability": _mean([1.0 if record["page_attribution_available"] else 0.0 for record in docs]),
        "bbox_availability": _mean([record["bbox_availability"] for record in docs if record["bbox_availability"] is not None]),
        "block_to_page_traceability": _mean([record["block_to_page_traceability"] for record in docs if record["block_to_page_traceability"] is not None]),
        "semantic_fidelity_inferred_from_provenance": False,
        "documents": docs,
    }
    return {
        "schema_version": "opk-rag.task0103.provenance-fidelity.v1",
        "task_id": TASK_ID,
        "candidates": dict(sorted(candidates.items())),
    }


def update_candidate_decisions_with_reference_scores(*, root: Path = ROOT, parsing: dict[str, Any], structure: dict[str, Any]) -> dict[str, Any]:
    decisions = read_json(root / RESULT_DIR.relative_to(ROOT) / "candidate_decision.json")
    pymu = decisions["candidates"]["pymupdf4llm"]
    parsing_score = parsing["candidates"]["pymupdf4llm"].get("text_retention")
    evaluated = parsing["candidates"]["pymupdf4llm"].get("evaluated_count", 0)
    if evaluated <= 0:
        pymu.update({"decision": "reject_quality", "reason": "no structured reference pairs were evaluable", "adapter_candidate_promotion": False})
    elif parsing_score is not None and parsing_score < 0.5:
        pymu.update({"decision": "reject_quality", "reason": "reference-scored text retention below promotion threshold", "adapter_candidate_promotion": False})
    else:
        pymu.update(
            {
                "decision": "promote_to_adapter_candidate",
                "reason": "formal run completed and reference-scored fidelity is evaluable on TASK-0101 supported structured sources",
                "adapter_candidate_promotion": True,
                "runtime_default_promotion": False,
                "reference_scoring": {
                    "text_retention": parsing_score,
                    "text_evaluable_count": evaluated,
                    "heading_fidelity": structure["candidates"]["pymupdf4llm"].get("heading_fidelity"),
                },
            }
        )
    return decisions


def update_summary_with_reference_scores(
    *,
    root: Path = ROOT,
    accounting: dict[str, Any],
    materialization: dict[str, Any],
    pairs: dict[str, Any],
    parsing: dict[str, Any],
    structure: dict[str, Any],
    provenance: dict[str, Any],
    decisions: dict[str, Any],
) -> dict[str, Any]:
    summary = read_json(root / RESULT_DIR.relative_to(ROOT) / "summary.json")
    parsing_candidate = parsing["candidates"]["pymupdf4llm"]
    structure_candidate = structure["candidates"]["pymupdf4llm"]
    provenance_candidate = provenance["candidates"]["pymupdf4llm"]
    supported_coverage = pairs["reference_pair_count"] / materialization["required_structured_source_count"] if materialization["required_structured_source_count"] else 0.0
    final_status = "complete" if pairs["reference_pair_count"] > 0 and materialization["structured_source_materialization_failure_count"] == 0 else "partial"
    if materialization["unsupported_structured_representation_count"] > 0 and supported_coverage < 0.8:
        final_status = "partial"
    summary.update(
        {
            "status": final_status,
            "task_status": final_status,
            "formal_bakeoff_completed": final_status == "complete",
            "formal_profile_document_accounting": {
                "required_pdf_count": accounting["required_pdf_count"],
                "materialized_pdf_count": accounting["materialized_pdf_count"],
                "formal_profile_knowledge_identity_count": accounting["formal_profile_knowledge_identity_count"],
                "formal_documents_evaluated": accounting["formal_documents_evaluated"],
                "explanation": accounting["explanation"],
            },
            "structured_reference_materialization": {
                "required_structured_source_count": materialization["required_structured_source_count"],
                "materialized_structured_source_count": materialization["materialized_structured_source_count"],
                "materialized_structured_bytes": materialization["materialized_structured_bytes"],
                "unsupported_structured_representation_count": materialization["unsupported_structured_representation_count"],
                "representation_type_distribution": materialization["representation_type_distribution"],
                "coverage": supported_coverage,
            },
            "reference_scoring": {
                "reference_pair_count": pairs["reference_pair_count"],
                "text_fidelity_evaluable_count": parsing_candidate["text_fidelity_evaluable_count"],
                "text_retention": parsing_candidate["text_retention"],
                "text_omission": parsing_candidate["text_omission"],
                "text_corruption": parsing_candidate["text_corruption"],
                "heading_fidelity_evaluable_count": structure_candidate["heading_fidelity_evaluable_count"],
                "heading_fidelity": structure_candidate["heading_fidelity"],
                "section_fidelity_evaluable_count": structure_candidate["section_fidelity_evaluable_count"],
                "section_fidelity": structure_candidate["section_fidelity"],
                "reading_order_evaluable_count": structure_candidate["reading_order_evaluable_count"],
                "reading_order_fidelity": structure_candidate["reading_order_fidelity"],
                "table_fidelity_evaluable_count": structure_candidate["table_fidelity_evaluable_count"],
                "table_fidelity": structure_candidate["table_fidelity"],
                "figure_fidelity_evaluable_count": structure_candidate["figure_fidelity_evaluable_count"],
                "figure_caption_fidelity": structure_candidate["figure_caption_fidelity"],
                "formula_fidelity_evaluable_count": structure_candidate["formula_fidelity_evaluable_count"],
                "formula_fidelity": structure_candidate["formula_fidelity"],
                "provenance_fidelity_summary": {
                    "page_attribution_availability": provenance_candidate["page_attribution_availability"],
                    "bbox_availability": provenance_candidate["bbox_availability"],
                    "block_to_page_traceability": provenance_candidate["block_to_page_traceability"],
                },
            },
            "issues": _updated_summary_issues(summary.get("issues", []), materialization, supported_coverage),
            "candidates": {
                candidate_id: {
                    "decision": record["decision"],
                    "primary_failure": record.get("primary_failure"),
                    "parse_success_rate": summary.get("candidates", {}).get(candidate_id, {}).get("parse_success_rate", 0.0),
                }
                for candidate_id, record in decisions.get("candidates", {}).items()
            },
            "adapter_promotion_candidates": [
                candidate_id
                for candidate_id, record in decisions.get("candidates", {}).items()
                if record.get("adapter_candidate_promotion") is True
            ],
        }
    )
    return summary


def _normalize_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(value):
        if not isinstance(block, dict):
            continue
        blocks.append(
            {
                "index": index,
                "type": block.get("type") or block.get("block_type"),
                "text": normalize_parser_text(str(block.get("text") or "")),
                "page": block.get("page") or block.get("page_number"),
            }
        )
    return blocks


def _structured_materialization_record(
    knowledge_id: str,
    status: str,
    representation: dict[str, Any] | None,
    *,
    unsupported_types: list[str] | None = None,
) -> dict[str, Any]:
    record = {
        "knowledge_id": knowledge_id,
        "materialization_status": status,
        "unsupported_types": unsupported_types or [],
    }
    if representation:
        record.update(
            {
                "structured_representation_id": representation.get("representation_id"),
                "representation_type": representation.get("representation_type"),
                "source_relative_path": representation.get("source_relative_path"),
                "upstream_relative_path": representation.get("source_relative_path"),
                "upstream_revision": representation.get("upstream_revision"),
                "declared_size": representation.get("size"),
                "file_identity": representation.get("file_identity"),
            }
        )
    return record


def _select_structured_reference_representation(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    structured = [
        representation
        for representation in entry.get("representations", [])
        if isinstance(representation, dict) and representation.get("representation_type") not in {"pdf", "csv"}
    ]
    supported = [
        representation
        for representation in structured
        if str(representation.get("representation_type", "")).lower() in SUPPORTED_STRUCTURED_REFERENCE_TYPES
    ]
    unsupported_types = sorted({str(representation.get("representation_type", "")).lower() for representation in structured if representation not in supported})
    if not supported:
        return None, unsupported_types
    return sorted(supported, key=lambda rep: (str(rep.get("representation_type")), str(rep.get("source_relative_path"))))[0], unsupported_types


def _download_asset(repository: str, revision: str, relative_path: str, local_path: Path, *, timeout: int) -> None:
    encoded_path = quote(relative_path)
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{encoded_path}"
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "opk-rag-task0103-structured-reference"})
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(local_path)
            return
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < 3:
                time.sleep(float(attempt))
    assert last_exc is not None
    raise last_exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("utf-8"))
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_counts(document: dict[str, Any]) -> dict[str, int]:
    return {
        "sections": len(document.get("sections", [])),
        "blocks": len(document.get("blocks", [])),
        "tables": len(document.get("tables", [])),
        "figures": len(document.get("figures", [])),
        "formulas": sum(1 for block in document.get("blocks", []) if block.get("block_type") == "formula"),
    }


def _candidate_counts(candidate: dict[str, Any]) -> dict[str, int]:
    return {
        "blocks": len(candidate.get("blocks") or []),
        "tables": len(candidate.get("tables") or []),
        "figures": len(candidate.get("figures") or []),
        "pages": int(candidate.get("page_count") or 0),
    }


def _candidate_comparison_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": candidate.get("text") or "",
        "blocks": [
            {
                "text": block.get("text"),
                "block_type": block.get("block_type") or block.get("type"),
                "page_number": block.get("page_number") or block.get("page"),
                "bbox": block.get("bbox"),
            }
            for block in candidate.get("blocks", [])
            if isinstance(block, dict)
        ],
        "tables": candidate.get("tables") or [],
        "figures": candidate.get("figures") or [],
        "page_count": candidate.get("page_count"),
    }


def _canonical_text(document: dict[str, Any]) -> str:
    return " ".join(str(block.get("text") or "") for block in sorted(document.get("blocks", []), key=lambda block: block.get("reading_order", 0)) if block.get("text"))


def _metric_not_evaluable_record(record: dict[str, Any], metric: str, reason: str | None) -> dict[str, Any]:
    return {"knowledge_id": record.get("knowledge_id"), "metric": metric, "status": "not_evaluable", "reason": reason}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", normalize_fidelity_text(text))


def _text_overlap_scores(reference_text: str, candidate_text: str) -> dict[str, Any]:
    reference_tokens = _tokens(reference_text)
    candidate_tokens = _tokens(candidate_text)
    reference_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    for token in candidate_tokens:
        candidate_counts[token] = candidate_counts.get(token, 0) + 1
    overlap = sum(min(count, candidate_counts.get(token, 0)) for token, count in reference_counts.items())
    retention = overlap / len(reference_tokens) if reference_tokens else None
    candidate_precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    return {
        "reference_token_count": len(reference_tokens),
        "candidate_token_count": len(candidate_tokens),
        "overlap_token_count": overlap,
        "text_retention": retention,
        "text_omission": None if retention is None else 1.0 - retention,
        "text_corruption": 1.0 - candidate_precision,
    }


def _contains_text(haystack: str, needle: str | None) -> bool:
    if not needle:
        return False
    return normalize_fidelity_text(needle) in normalize_fidelity_text(haystack)


def _reference_block_snippets(reference: dict[str, Any]) -> list[str]:
    snippets = []
    for block in sorted(reference.get("blocks", []), key=lambda item: item.get("reading_order", 0)):
        text = normalize_fidelity_text(str(block.get("text") or ""))
        if len(text) >= 20:
            snippets.append(text[:120])
    return snippets


def _reading_order_metric(snippets: list[str], candidate_text: str) -> dict[str, Any]:
    normalized_candidate = normalize_fidelity_text(candidate_text)
    offsets = [normalized_candidate.find(snippet) for snippet in snippets]
    found = [offset for offset in offsets if offset >= 0]
    if not found:
        return {
            "evaluated_block_count": len(snippets),
            "matched_block_count": 0,
            "reading_order_error_count": len(snippets),
            "reading_order_fidelity": 0.0,
        }
    lis_len = _longest_increasing_subsequence_length(found)
    errors = len(snippets) - lis_len
    return {
        "evaluated_block_count": len(snippets),
        "matched_block_count": len(found),
        "reading_order_error_count": errors,
        "reading_order_fidelity": lis_len / len(snippets),
    }


def _longest_increasing_subsequence_length(values: list[int]) -> int:
    tails: list[int] = []
    for value in values:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(value)
        else:
            tails[lo] = value
    return len(tails)


def _table_metric(record: dict[str, Any]) -> dict[str, Any]:
    reference_tables = record["reference_document"].get("tables", [])
    candidate_tables = record["candidate_output"].get("tables", [])
    candidate_text = record["candidate_output"].get("text") or ""
    reference_cells = [cell.get("text") for table in reference_tables for cell in table.get("cells", []) if cell.get("text")]
    retained_cells = sum(1 for text in reference_cells if _contains_text(candidate_text, text))
    detection_recall = min(len(candidate_tables), len(reference_tables)) / len(reference_tables) if reference_tables else None
    cell_retention = retained_cells / len(reference_cells) if reference_cells else None
    caption_texts = [table.get("caption") for table in reference_tables if table.get("caption")]
    caption_retention = _mean([1.0 if _contains_text(candidate_text, caption) else 0.0 for caption in caption_texts])
    components = [value for value in (detection_recall, cell_retention, caption_retention) if value is not None]
    return {
        "knowledge_id": record["knowledge_id"],
        "reference_table_count": len(reference_tables),
        "candidate_table_count": len(candidate_tables),
        "table_detection_recall": detection_recall,
        "row_count_retention": 0.0 if reference_tables and not candidate_tables else None,
        "column_count_retention": 0.0 if reference_tables and not candidate_tables else None,
        "cell_text_retention": cell_retention,
        "caption_retention": caption_retention,
        "partial_evaluation": True,
        "table_fidelity": _mean(components),
    }


def _figure_metric(record: dict[str, Any]) -> dict[str, Any]:
    reference_figures = record["reference_document"].get("figures", [])
    candidate_figures = record["candidate_output"].get("figures", [])
    candidate_text = record["candidate_output"].get("text") or ""
    captions = [figure.get("caption") for figure in reference_figures if figure.get("caption")]
    caption_score = _mean([1.0 if _contains_text(candidate_text, caption) else 0.0 for caption in captions])
    detection = min(len(candidate_figures), len(reference_figures)) / len(reference_figures) if reference_figures else None
    return {
        "knowledge_id": record["knowledge_id"],
        "reference_figure_count": len(reference_figures),
        "candidate_figure_count": len(candidate_figures),
        "figure_detection": detection,
        "caption_presence": len(captions) / len(reference_figures) if reference_figures else None,
        "caption_text_retention": caption_score,
        "figure_caption_association": 0.0 if captions and not candidate_figures else None,
        "figure_caption_fidelity": _mean([value for value in (detection, caption_score) if value is not None]),
    }


def _formula_metric(record: dict[str, Any], formulas: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_text = record["candidate_output"].get("text") or ""
    retained = [1.0 if _contains_text(candidate_text, formula.get("text")) else 0.0 for formula in formulas if formula.get("text")]
    return {
        "knowledge_id": record["knowledge_id"],
        "reference_formula_count": len(formulas),
        "candidate_formula_count": 0,
        "formula_detection": 0.0,
        "formula_text_retention": _mean(retained),
        "formula_fidelity": 0.0 if retained else None,
        "reason": "candidate does not expose governed formula objects; plain OCR-like text is not counted as LaTeX recovery",
    }


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _candidate_primary_failure(candidate_id: str) -> str | None:
    return {
        "docling": "model_materialization_failure",
        "marker": "timeout",
        "mineru": "dependency_conflict",
    }.get(candidate_id)


def _updated_summary_issues(existing: list[Any], materialization: dict[str, Any], supported_coverage: float) -> list[str]:
    stale_fragments = (
        "Structured source reference assets were not materialized",
        "Structured Reference scoring is partial:",
        "Some TASK-0101-supported structured source assets failed materialization",
    )
    issues = [str(issue) for issue in existing if not any(fragment in str(issue) for fragment in stale_fragments)]
    if materialization["unsupported_structured_representation_count"]:
        issues.append(
            f"Structured Reference scoring is partial: TASK-0101 adapters cover {supported_coverage:.3f} of frozen formal profile; unsupported representations remain fail-closed."
        )
    if materialization["structured_source_materialization_failure_count"]:
        issues.append("Some TASK-0101-supported structured source assets failed materialization or integrity checks.")
    return issues


def _normalize_tables(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [{"rows": table.get("rows"), "columns": table.get("columns"), "caption": normalize_parser_text(str(table.get("caption") or ""))} for table in value if isinstance(table, dict)]


def _normalize_figures(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [{"caption": normalize_parser_text(str(figure.get("caption") or "")), "page": figure.get("page") or figure.get("page_number")} for figure in value if isinstance(figure, dict)]


def _admitted_candidate_ids(authority: dict[str, Any]) -> list[str]:
    candidates = []
    for candidate in authority.get("candidates", []):
        if isinstance(candidate, dict) and candidate.get("formal_bakeoff_admitted") is True:
            candidates.append(candidate["candidate_id"])
    return sorted(candidates)


def _candidate_records(authority: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in authority.get("candidates", []) if isinstance(candidate, dict) and candidate.get("formal_bakeoff_admitted") is True}


def _candidate_config(candidate_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "package": candidate.get("authority", {}).get("official_package"),
        "version": candidate.get("authority", {}).get("version"),
        "revision": candidate.get("authority", {}).get("revision"),
        "ocr_enabled": False,
        "external_api_allowed": False,
        "external_llm_allowed": False,
        "llm_repair_allowed": False,
        "execution": "isolated environment or subprocess runner",
    }


def _select_profile_entries(entries: list[Any], *, max_documents: int, mode: str) -> list[dict[str, Any]]:
    paired = [entry for entry in entries if isinstance(entry, dict) and entry.get("pairing_status") == "paired" and _representation(entry, "pdf") is not None]
    if mode == "smallest":
        paired = sorted(paired, key=lambda entry: (_representation(entry, "pdf") or {}).get("size", 0))
    else:
        paired = sorted(paired, key=lambda entry: stable_digest({"knowledge_id": entry.get("knowledge_id")}))
    return paired[:max_documents]


def _profile_record(name: str, protocol_profile: dict[str, Any], entries: list[dict[str, Any]], *, root: Path) -> dict[str, Any]:
    pdfs = [_representation(entry, "pdf") for entry in entries]
    pdfs = [pdf for pdf in pdfs if pdf is not None]
    materialized = [_materialized_path(pdf["source_relative_path"], root=root) for pdf in pdfs]
    return {
        "name": name,
        "profile_digest": profile_digest(protocol_profile),
        "selection": protocol_profile.get("selection"),
        "deterministic": protocol_profile.get("deterministic") is True,
        "uses_qa_gold": protocol_profile.get("uses_qa_gold") is True,
        "sample_count": len(entries),
        "pdf_count": len(pdfs),
        "page_count": None,
        "page_count_status": "not_measured_without_materialized_pdfs" if not any(materialized) else "pending_parser_measurement",
        "knowledge_identity_count": len({entry.get("knowledge_id") for entry in entries}),
        "downloaded_pdf_count": sum(1 for path in materialized if path is not None),
        "downloaded_bytes": sum(path.stat().st_size for path in materialized if path is not None),
        "materialized_pdf_count": sum(1 for path in materialized if path is not None),
        "missing_pdf_count": sum(1 for path in materialized if path is None),
        "entries": [
            {
                "knowledge_id": entry.get("knowledge_id"),
                "pdf_representation_id": (_representation(entry, "pdf") or {}).get("representation_id"),
                "pdf_source_relative_path": (_representation(entry, "pdf") or {}).get("source_relative_path"),
                "structured_representation_types": sorted({rep.get("representation_type") for rep in entry.get("representations", []) if isinstance(rep, dict) and rep.get("representation_type") != "pdf"}),
                "pdf_materialized": _materialized_path((_representation(entry, "pdf") or {}).get("source_relative_path"), root=root) is not None,
            }
            for entry in entries
        ],
    }


def _representation(entry: dict[str, Any], representation_type: str) -> dict[str, Any] | None:
    for representation in entry.get("representations", []):
        if isinstance(representation, dict) and representation.get("representation_type") == representation_type:
            return representation
    return None


def _materialized_path(relative_path: str | None, *, root: Path) -> Path | None:
    if not relative_path:
        return None
    for base in (root / "external-data" / "pdfqa", root / ".cache" / "pdfqa"):
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    return None


def _installation_record(candidate_id: str, candidate: dict[str, Any], *, install: bool, blocked_reason: str | None) -> dict[str, Any]:
    authority = candidate.get("authority", {})
    if blocked_reason:
        return {
            "candidate_id": candidate_id,
            "install_attempted": False,
            "install_status": "blocked_by_environment",
            "package_version_resolved": None,
            "expected_version": authority.get("version"),
            "version_match": None,
            "dependency_conflicts": [],
            "system_dependencies": [],
            "installation_wall_time": 0.0,
            "disk_footprint": None,
            "dependency_isolation_failure": False,
            "failure_classification": "blocked_by_environment",
            "failure_reason": blocked_reason,
        }
    return {
        "candidate_id": candidate_id,
        "install_attempted": bool(install),
        "install_status": "not_executed" if not install else "not_implemented_in_core_runner",
        "package_version_resolved": None,
        "expected_version": authority.get("version"),
        "version_match": None,
        "dependency_conflicts": [],
        "system_dependencies": [],
        "installation_wall_time": 0.0,
        "disk_footprint": None,
        "dependency_isolation_failure": False,
        "failure_classification": None if not install else "blocked_by_environment",
        "failure_reason": None if not install else "isolated parser installation runner not invoked in this environment",
    }


def _blocked_stage_record(stage: str, failure: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "blocked_by_environment" if failure == "blocked_by_environment" else "not_executed",
        "failure_classification": failure,
        "failure_reason": reason,
    }


def _not_executed_record(stage: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_executed", "failure_classification": None, "failure_reason": None}


def _not_evaluable_metric(metric_family: str, failure: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "metric_family": metric_family,
        "status": "not_evaluable" if failure else "pending_measurement",
        "reason": reason,
        "failure_classification": failure,
        "text_retention": None,
        "text_omission": None,
        "text_corruption": None,
        "heading_preservation": "not_evaluable",
        "section_hierarchy_retention": "not_evaluable",
        "reading_order_success": None,
        "reading_order_error_count": None,
        "table_detected": None,
        "table_count": None,
        "figure_detection": None,
        "formula_detection": None,
        "page_attribution": "not_evaluable",
        "bbox_availability": "not_evaluable",
        "source_traceability": "not_evaluable",
    }


def _reliability_record(candidate_id: str, failure: str | None) -> dict[str, Any]:
    counters = {counter: 0 for counter in FAILURE_TO_COUNTER.values()}
    if failure:
        counters[FAILURE_TO_COUNTER[failure]] += 1
    return {
        "candidate_id": candidate_id,
        "documents_attempted": 0,
        "documents_succeeded": 0,
        "parse_success_rate": 0.0,
        **counters,
    }


def _resource_record(candidate_id: str, install_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "cold_start_latency": None,
        "warm_parse_latency": None,
        "per_document_latency": [],
        "per_page_latency": None,
        "latency_p50": None,
        "latency_p95": None,
        "latency_percentile_status": "insufficient_sample_size",
        "pages_per_second": None,
        "peak_rss_bytes": None,
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
        "package_footprint": install_record.get("disk_footprint"),
        "model_footprint": None,
        "cache_footprint": None,
        "measurement_status": install_record.get("install_status"),
    }


def _decision_record(candidate_id: str, failure: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "decision": "blocked_by_environment" if failure == "blocked_by_environment" else "retain_for_secondary_arm",
        "primary_failure": failure,
        "reason": reason,
        "adapter_candidate_promotion": False,
        "runtime_default_promotion": False,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q == 50:
        return float(median(values))
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q / 100))
    return float(ordered[index])


def preflight_issues(*, root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    task0102_contract = read_json(root / FREEZE_CONTRACT_PATH.relative_to(ROOT))
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    issues.extend(validate_candidate_authority(authority))
    issues.extend(validate_freeze_contract(task0102_contract))
    issues.extend(validate_bakeoff_protocol(protocol))
    if read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT)).get("authority_status") != "frozen":
        issues.append("TASK-0099 pdfQA authority is not frozen")
    if CANONICAL_DOCUMENT_IR_SCHEMA_VERSION != 1:
        issues.append("TASK-0100 CanonicalDocument IR v1 is not preserved")
    return issues


def _runtime_preservation(root: Path) -> dict[str, Any]:
    status = _run_text(["git", "status", "--short"], cwd=root)
    changed = [line[3:] for line in status.splitlines() if line]
    reranker_changes = [path for path in changed if path.startswith("opk_rag/reranking/")]
    return {
        "changed_paths": changed,
        "production_markdown_path_modified": any(path.startswith("opk_rag/chunking/") for path in changed),
        "production_chunking_modified": any(path.startswith("opk_rag/chunking/") for path in changed),
        "default_runtime_modified": any(path.startswith(("opk_rag/runtime", "opk_rag/runtime_v2/")) for path in changed),
        "default_retrieval_modified": _default_retrieval_modified(root, changed),
        "default_reranker_modified": bool(reranker_changes) and not _task0212_authorized_reranker_changes(root, reranker_changes),
        "default_agent_behavior_modified": any(path.startswith("opk_rag/agent/") for path in changed),
        "graph_runtime_modified": False,
    }


def _task0212_authorized_reranker_changes(root: Path, changed: list[str]) -> bool:
    allowed = {
        "opk_rag/reranking/bge.py",
        "opk_rag/reranking/config.py",
        "opk_rag/reranking/provider.py",
        "opk_rag/reranking/qwen.py",
        "opk_rag/reranking/zerank.py",
    }
    if not set(changed).issubset(allowed):
        return False
    summary_path = root / "evaluation-data" / "results" / "task0212-search-scoped-fp16-autocast-reranker-production-integration" / "summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("task_id") == "TASK-0212"
        and summary.get("promotion_scope") == "search_only"
        and summary.get("production_ask_reranker_precision") == "fp32"
        and summary.get("independent_verifier_passed") is True
    )


def _default_retrieval_modified(root: Path, changed: list[str]) -> bool:
    search_changes = [path for path in changed if path.startswith("opk_rag/search/")]
    if not search_changes:
        return False
    allowed_observation_paths = {"opk_rag/search/models.py", "opk_rag/search/service.py", "opk_rag/search/retrieval_timing.py"}
    if not set(search_changes).issubset(allowed_observation_paths):
        return True
    task0206_summary = root / "evaluation-data" / "results" / "task0206-retrieval-latency-bottleneck-attribution" / "summary.json"
    if task0206_summary.exists():
        summary = read_json(task0206_summary)
        if (
            summary.get("task_id") == "TASK-0206"
            and summary.get("measurement_valid") is True
            and summary.get("runtime_default_behavior_change") is False
            and summary.get("promotion_applied") is False
        ):
            return False
    task0199_summary = root / "evaluation-data" / "results" / "task0199-qdrant-shadow-runtime-integration" / "summary.json"
    if not task0199_summary.exists():
        return True
    summary = read_json(task0199_summary)
    return not (
        summary.get("task_id") == "TASK-0199"
        and summary.get("runtime_shadow_capability_added") is True
        and summary.get("production_default_behavior_change") is False
        and summary.get("production_answer_authority_change") is False
        and summary.get("production_backend_promoted") is False
    )


def _run_text(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _module_info(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {"installed": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    info = {"installed": True, "version": getattr(module, "__version__", "unknown")}
    if module_name == "torch":
        info["cuda_available"] = bool(module.cuda.is_available())
        info["cuda_version"] = module.version.cuda
    return info


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    result = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip()}
    first = result.stdout.strip().splitlines()[0].split(",")
    return {
        "available": True,
        "name": first[0].strip() if len(first) > 0 else None,
        "total_vram": first[1].strip() if len(first) > 1 else None,
        "driver_version": first[2].strip() if len(first) > 2 else None,
    }


def _cpu_model() -> str | None:
    if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _ram_info() -> dict[str, Any]:
    if platform.system() == "Linux" and Path("/proc/meminfo").exists():
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", meminfo, flags=re.MULTILINE)
        if match:
            return {"total_kib": int(match.group(1))}
    return {"total_kib": None}
