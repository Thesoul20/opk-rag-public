from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from opk_rag.document_ir import CANONICAL_DOCUMENT_IR_SCHEMA_VERSION
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT


TASK_ID = "TASK-0102"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0102-pdf-parser-readiness"
AUTHORITY_PATH = ROOT / "evaluation-data" / "external" / "pdf-parsers" / "candidate_authority.json"
FREEZE_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0102_pdf_parser_candidate_freeze_contract.json"
BAKEOFF_PROTOCOL_PATH = ROOT / "evaluation-data" / "contracts" / "task0102_pdf_parser_bakeoff_protocol.json"
PDFQA_AUTHORITY_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "authority.json"
PDFQA_PAIRING_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "pdfqa_representation_pairing_manifest.json"

IMMUTABLE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FLOATING_REFS = {"latest", "main", "master", "head", "nightly", "dev"}

REQUIRED_FAILURE_TAXONOMY = (
    "installation_failure",
    "dependency_conflict",
    "model_materialization_failure",
    "unsupported_pdf",
    "parse_failure",
    "timeout",
    "oom",
    "empty_output",
    "canonical_mapping_failure",
    "nondeterministic_output",
    "blocked_by_environment",
)

REQUIRED_METRIC_FAMILIES = (
    "parsing_fidelity",
    "structure_fidelity",
    "provenance_fidelity",
    "reliability",
    "resource_cost",
)

RUNTIME_PRESERVATION_KEYS = (
    "production_markdown_path_modified",
    "production_chunking_modified",
    "default_runtime_modified",
    "default_retrieval_modified",
    "default_reranker_modified",
    "default_agent_behavior_modified",
    "graph_runtime_modified",
)


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


def validate_pinned_version(label: str, value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{label} must be a non-empty version pin"]
    if value.lower() in FLOATING_REFS:
        return [f"{label} must not use floating ref {value!r}"]
    if any(token in value.lower() for token in ("latest", "main", "master")):
        return [f"{label} contains a floating token: {value!r}"]
    return []


def validate_revision_pin(label: str, value: Any, *, allow_unavailable: bool = False) -> list[str]:
    if allow_unavailable and value == "unavailable":
        return []
    if not isinstance(value, str):
        return [f"{label} must be a revision string"]
    if value.lower() in FLOATING_REFS:
        return [f"{label} must not use floating ref {value!r}"]
    if IMMUTABLE_SHA_RE.fullmatch(value) is None:
        return [f"{label} must be a 40-character lowercase git SHA or explicit unavailable"]
    return []


def validate_candidate_authority(authority: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    candidates = authority.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ["candidate authority must contain non-empty candidates list"]

    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            issues.append(f"candidate {index} must be an object")
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            issues.append(f"candidate {index} missing candidate_id")
        elif candidate_id in seen:
            issues.append(f"duplicate candidate_id: {candidate_id}")
        else:
            seen.add(candidate_id)

        authority_block = candidate.get("authority", {})
        if not isinstance(authority_block, dict):
            issues.append(f"{candidate_id}.authority must be an object")
            authority_block = {}
        issues.extend(validate_pinned_version(f"{candidate_id}.authority.version", authority_block.get("version")))
        issues.extend(
            validate_revision_pin(
                f"{candidate_id}.authority.revision",
                authority_block.get("revision"),
                allow_unavailable=True,
            )
        )
        for key in ("official_repository", "official_package", "license"):
            if not isinstance(authority_block.get(key), str) or not authority_block.get(key):
                issues.append(f"{candidate_id}.authority.{key} must be recorded")

        gates = candidate.get("gates", {})
        if not isinstance(gates, dict):
            issues.append(f"{candidate_id}.gates must be an object")
            gates = {}
        if gates.get("mandatory_cloud_api") is True:
            issues.append(f"{candidate_id} requires mandatory cloud API")
        if gates.get("mandatory_external_llm") is True:
            issues.append(f"{candidate_id} requires mandatory external LLM")

        license_status = candidate.get("license_status")
        admission = candidate.get("formal_bakeoff_admitted")
        if license_status != "resolved" and admission is True:
            issues.append(f"{candidate_id} cannot be admitted with unresolved license_status={license_status!r}")
        if candidate.get("canonical_mapping_feasibility") not in {"direct_mapping", "derived_mapping", "unsupported", "ambiguous"}:
            issues.append(f"{candidate_id}.canonical_mapping_feasibility has invalid classification")
    return issues


def validate_freeze_contract(contract: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_values: dict[str, Any] = {
        "task_id": TASK_ID,
        "born_digital_primary_arm": True,
        "ocr_primary_arm_enabled": False,
        "external_llm_allowed": False,
        "external_api_allowed": False,
        "canonical_document_ir_version": 1,
        "structured_source_reference_arm": "task0101",
        "formal_profile_defined": True,
        "development_profile_defined": True,
        "smoke_profile_defined": True,
        "chunking_modified": False,
        "retrieval_modified": False,
        "candidate_auto_upgrade_allowed": False,
        "pdf_parser_default_selected": False,
        "pdf_parser_production_integrated": False,
        "default_runtime_modified": False,
    }
    for key, value in expected_values.items():
        if contract.get(key) != value:
            issues.append(f"contract.{key} expected {value!r}, got {contract.get(key)!r}")
    for key in RUNTIME_PRESERVATION_KEYS:
        if contract.get(key) is not False:
            issues.append(f"contract.{key} must be false")
    if contract.get("canonical_document_ir_version") != CANONICAL_DOCUMENT_IR_SCHEMA_VERSION:
        issues.append("contract canonical IR version does not match code")
    if set(contract.get("metric_families", [])) != set(REQUIRED_METRIC_FAMILIES):
        issues.append("contract.metric_families incomplete")
    if set(contract.get("failure_taxonomy", [])) != set(REQUIRED_FAILURE_TAXONOMY):
        issues.append("contract.failure_taxonomy incomplete")
    if not isinstance(contract.get("formal_bakeoff_candidates"), list) or not contract["formal_bakeoff_candidates"]:
        issues.append("contract.formal_bakeoff_candidates must be non-empty")
    if not isinstance(contract.get("candidate_versions"), dict):
        issues.append("contract.candidate_versions must be an object")
    else:
        for candidate_id, version in contract["candidate_versions"].items():
            issues.extend(validate_pinned_version(f"candidate_versions.{candidate_id}", version))
    if not isinstance(contract.get("candidate_revisions"), dict):
        issues.append("contract.candidate_revisions must be an object")
    else:
        for candidate_id, revision in contract["candidate_revisions"].items():
            issues.extend(validate_revision_pin(f"candidate_revisions.{candidate_id}", revision, allow_unavailable=True))
    return issues


def validate_bakeoff_protocol(protocol: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if protocol.get("task_id") != TASK_ID:
        issues.append("protocol.task_id must be TASK-0102")
    if protocol.get("dataset_authority", {}).get("dataset") != "syn-pdfQA":
        issues.append("protocol must use syn-pdfQA")
    profiles = protocol.get("sample_profiles", {})
    if not isinstance(profiles, dict):
        return issues + ["protocol.sample_profiles must be an object"]
    for profile_name in ("smoke", "development", "formal"):
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            issues.append(f"protocol.sample_profiles.{profile_name} missing")
            continue
        if profile.get("deterministic") is not True:
            issues.append(f"protocol.sample_profiles.{profile_name}.deterministic must be true")
        if profile.get("uses_qa_gold") is not False:
            issues.append(f"protocol.sample_profiles.{profile_name}.uses_qa_gold must be false")
    if set(protocol.get("failure_taxonomy", [])) != set(REQUIRED_FAILURE_TAXONOMY):
        issues.append("protocol.failure_taxonomy incomplete")
    if set(protocol.get("metric_families", [])) != set(REQUIRED_METRIC_FAMILIES):
        issues.append("protocol.metric_families incomplete")
    if protocol.get("ocr_policy", {}).get("primary_arm_enabled") is not False:
        issues.append("protocol OCR primary arm must be disabled")
    if protocol.get("chunking_freeze_policy", {}).get("production_chunking_modified") is not False:
        issues.append("protocol must freeze production chunking")
    timeout = protocol.get("timeout_policy", {})
    if not isinstance(timeout.get("per_document_seconds"), int) or timeout["per_document_seconds"] <= 0:
        issues.append("protocol timeout_policy.per_document_seconds must be positive integer")
    if not isinstance(timeout.get("global_seconds"), int) or timeout["global_seconds"] <= timeout.get("per_document_seconds", 0):
        issues.append("protocol timeout_policy.global_seconds must exceed per-document timeout")
    return issues


def pdfqa_protocol_summary(root: Path = ROOT) -> dict[str, Any]:
    authority = read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT))
    pairing = read_json(root / PDFQA_PAIRING_PATH.relative_to(ROOT))
    return {
        "authority_revision": authority.get("dataset", {}).get("revision"),
        "authority_status": authority.get("authority_status"),
        "syn_pdfqa_enabled": authority.get("benchmark_arms", {}).get("syn_pdfqa", {}).get("enabled"),
        "paired_knowledge_units": pairing.get("paired_unit_count"),
        "pairing_digest": pairing.get("pairing_digest"),
        "missing_pdf_count": pairing.get("missing_pdf_count"),
        "missing_structured_source_count": pairing.get("missing_structured_source_count"),
    }


def git_status_paths(root: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if line:
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            paths.append(path)
    return paths


def runtime_preservation(root: Path = ROOT) -> dict[str, Any]:
    changed = git_status_paths(root)
    return {
        "changed_paths": changed,
        "production_markdown_path_modified": any(path.startswith("opk_rag/chunking/") for path in changed),
        "production_chunking_modified": any(path.startswith("opk_rag/chunking/") for path in changed),
        "default_runtime_modified": any(path.startswith(("opk_rag/runtime", "opk_rag/runtime_v2/")) for path in changed),
        "default_retrieval_modified": any(path.startswith("opk_rag/search/") for path in changed),
        "default_reranker_modified": any(path.startswith("opk_rag/reranking/") for path in changed),
        "default_agent_behavior_modified": any(path.startswith("opk_rag/agent/") for path in changed),
        "graph_runtime_modified": False,
    }


def verify_task0102_artifacts(root: Path = ROOT) -> dict[str, Any]:
    issues: list[str] = []
    authority = read_json(root / AUTHORITY_PATH.relative_to(ROOT))
    contract = read_json(root / FREEZE_CONTRACT_PATH.relative_to(ROOT))
    protocol = read_json(root / BAKEOFF_PROTOCOL_PATH.relative_to(ROOT))
    readiness = read_json(root / RESULT_DIR.relative_to(ROOT) / "candidate_readiness_matrix.json")
    capability = read_json(root / RESULT_DIR.relative_to(ROOT) / "parser_capability_matrix.json")

    issues.extend(validate_candidate_authority(authority))
    issues.extend(validate_freeze_contract(contract))
    issues.extend(validate_bakeoff_protocol(protocol))

    candidates = authority.get("candidates", [])
    authority_ids = {candidate.get("candidate_id") for candidate in candidates if isinstance(candidate, dict)}
    if set(readiness.get("candidates", {}).keys()) != authority_ids:
        issues.append("readiness matrix candidate IDs differ from authority")
    if set(capability.get("candidates", {}).keys()) != authority_ids:
        issues.append("capability matrix candidate IDs differ from authority")

    runtime = runtime_preservation(root)
    for key in RUNTIME_PRESERVATION_KEYS:
        if runtime.get(key) is not False:
            issues.append(f"runtime preservation failed: {key}")

    pdfqa = pdfqa_protocol_summary(root)
    if pdfqa.get("authority_status") != "frozen" or not pdfqa.get("syn_pdfqa_enabled"):
        issues.append("pdfQA syn-pdfQA authority is not frozen/enabled")
    if contract.get("pdfqa_authority_revision") != pdfqa.get("authority_revision"):
        issues.append("contract pdfQA revision differs from authority")

    return {
        "schema_version": "opk-rag.task0102.pdf-parser-candidate-freeze.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "candidate_ids": sorted(authority_ids),
        "formal_bakeoff_candidates": contract.get("formal_bakeoff_candidates"),
        "pdfqa": pdfqa,
        "runtime_preservation": runtime,
        "git_add_executed": False,
        "git_commit_created": False,
    }
