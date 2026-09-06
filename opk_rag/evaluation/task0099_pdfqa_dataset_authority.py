from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0099"
EXTERNAL_DIR = ROOT / "evaluation-data" / "external" / "pdfqa"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0099_pdfqa_dataset_authority_contract.json"
AUTHORITY_PATH = EXTERNAL_DIR / "authority.json"
INVENTORY_PATH = EXTERNAL_DIR / "pdfqa_upstream_inventory.json"
SCHEMA_PATH = EXTERNAL_DIR / "pdfqa_schema_snapshot.json"
PAIRING_PATH = EXTERNAL_DIR / "pdfqa_representation_pairing_manifest.json"
LICENSE_PATH = EXTERNAL_DIR / "pdfqa_license_provenance.json"
TASK0098_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0098_project_rebaseline_contract.json"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0099-pdfqa-dataset-authority"

IMMUTABLE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_TASK0098_VALUES: dict[str, Any] = {
    "primary_document_benchmark": "pdfQA",
    "primary_real_world_graph_benchmark": "WildGraphBench",
    "primary_graph_necessity_benchmark": "GraphRAG-Bench",
    "new_large_internal_benchmark_default_allowed": False,
    "graph_replaces_agent_rag": False,
    "final_architecture_target": "adaptive_agentic_rag",
}

EXPECTED_CONTRACT_VALUES: dict[str, Any] = {
    "task_id": TASK_ID,
    "authority_status": "frozen",
    "dataset_name": "pdfQA",
    "dataset_provider": "huggingface",
    "dataset_repository": "pdfqa/pdfQA-Benchmark",
    "annotation_repository": "pdfqa/pdfQA-Annotations",
    "benchmark_primary_role": "heterogeneous_document_agent_rag",
    "syn_pdfqa_enabled": True,
    "syn_pdfqa_role": "cross_representation_primary",
    "real_pdfqa_enabled": True,
    "real_pdfqa_role": "real_world_human_annotated_holdout",
    "full_authority_frozen": True,
    "knowledge_identity_defined": True,
    "representation_identity_defined": True,
    "question_identity_defined": True,
    "evidence_identity_defined": True,
    "representation_pairing_audited": True,
    "gold_answer_runtime_visible": False,
    "gold_evidence_runtime_visible": False,
    "gold_metadata_runtime_visible": False,
    "oracle_policy_defined": True,
    "cross_representation_split_leakage_allowed": False,
    "dataset_revision_auto_update_allowed": False,
    "external_dataset_fully_downloaded": False,
    "pdf_parser_implemented": False,
    "canonical_document_runtime_implemented": False,
    "default_runtime_modified": False,
    "default_retrieval_modified": False,
    "default_reranker_modified": False,
    "default_agent_behavior_modified": False,
}

ALLOWED_CHANGED_PREFIXES = (
    "README.md",
    "CHANGELOG.md",
    ".gitignore",
    "docs/HETEROGENEOUS_DOCUMENT_AND_ADAPTIVE_GRAPHRAG_ROADMAP.md",
    "docs/TASK0099_PDFQA_DATASET_AUTHORITY_FREEZE_REPORT.md",
    "evaluation-data/contracts/task0099_pdfqa_dataset_authority_contract.json",
    "evaluation-data/external/",
    "evaluation-data/external/pdfqa/",
    "evaluation-data/results/task0099-pdfqa-dataset-authority/",
    "opk_rag/evaluation/task0099_pdfqa_dataset_authority.py",
    "scripts/verify_task0099_pdfqa_dataset_authority.py",
    "tasks/TASK-0099-",
    "tests/test_task0099_pdfqa_dataset_authority.py",
)


class Task0099ValidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise Task0099ValidationError(f"expected JSON object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def is_immutable_sha(value: Any) -> bool:
    return isinstance(value, str) and IMMUTABLE_SHA_RE.fullmatch(value) is not None


def validate_frozen_revision(label: str, value: Any) -> list[str]:
    if value in {"main", "master", "latest"}:
        return [f"{label} must be an immutable SHA, got mutable ref {value!r}"]
    if not is_immutable_sha(value):
        return [f"{label} must be a 40-character lowercase hex SHA"]
    return []


def validate_inventory(inventory: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    issues.extend(validate_frozen_revision("inventory.dataset_revision", inventory.get("dataset_revision")))
    entries = inventory.get("inventory_entries")
    if not isinstance(entries, list) or not entries:
        return issues + ["inventory.inventory_entries must be a non-empty list"]

    canonical_entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"inventory entry {index} must be an object")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            issues.append(f"inventory entry {index} missing path")
        elif path in seen_paths:
            issues.append(f"duplicate inventory path: {path}")
        else:
            seen_paths.add(path)
        if "size" not in entry:
            issues.append(f"inventory entry {index} missing size")
        if not isinstance(entry.get("file_identity"), str) or not entry.get("file_identity"):
            issues.append(f"inventory entry {index} missing file_identity")
        canonical_entries.append(
            {
                "path": entry.get("path"),
                "file_identity": entry.get("file_identity"),
                "size": entry.get("size"),
            }
        )

    expected_digest = sha256_hex(sorted(canonical_entries, key=lambda item: str(item["path"])))
    if inventory.get("inventory_digest") != expected_digest:
        issues.append(
            f"inventory.inventory_digest mismatch: expected {expected_digest}, got {inventory.get('inventory_digest')!r}"
        )
    if inventory.get("file_count") != len(entries):
        issues.append(f"inventory.file_count expected {len(entries)}, got {inventory.get('file_count')!r}")
    return issues


def validate_schema_snapshot(schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required_categories = ("syn-pdfQA", "real-pdfQA")
    for category in required_categories:
        arm = schema.get("schemas", {}).get(category)
        if not isinstance(arm, dict):
            issues.append(f"schema missing {category}")
            continue
        fields = arm.get("fields")
        if not isinstance(fields, dict):
            issues.append(f"schema.{category}.fields missing")
            continue
        for field in ("question", "answer", "sources", "file_name"):
            if fields.get(field, {}).get("status") != "present":
                issues.append(f"schema.{category}.{field} must be present")
        if fields.get("source_text", {}).get("status") not in {"present", "not_present"}:
            issues.append(f"schema.{category}.source_text must be audited")
    return issues


def validate_pairing_manifest(pairing: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    entries = pairing.get("pairing_entries")
    if not isinstance(entries, list) or not entries:
        return ["pairing.pairing_entries must be a non-empty list"]
    allowed_statuses = {"paired", "ambiguous_pairing", "missing_pdf", "missing_structured_source"}
    canonical_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"pairing entry {index} must be an object")
            continue
        status = entry.get("pairing_status")
        if status not in allowed_statuses:
            issues.append(f"pairing entry {index} has unclassified status {status!r}")
        reasons = entry.get("ambiguity_reasons")
        if status == "ambiguous_pairing" and not reasons:
            issues.append(f"ambiguous pairing entry {index} must include ambiguity_reasons")
        canonical_entries.append(
            {
                "knowledge_id": entry.get("knowledge_id"),
                "pairing_status": status,
                "ambiguity_reasons": reasons or [],
                "representations": entry.get("representations") or [],
            }
        )
    expected_digest = sha256_hex(canonical_entries)
    if pairing.get("pairing_digest") != expected_digest:
        issues.append(f"pairing.pairing_digest mismatch: expected {expected_digest}")
    if pairing.get("ambiguous_pair_count", 0) < 0:
        issues.append("pairing.ambiguous_pair_count cannot be negative")
    return issues


def validate_question_identities(manifest: dict[str, Any]) -> list[str]:
    question_identity = manifest.get("question_identity")
    if not isinstance(question_identity, dict):
        question_identity = manifest.get("identity_policy", {}).get("question_identity", {})
    ids = question_identity.get("sample_question_ids", [])
    duplicate_count = question_identity.get("duplicate_question_identity_count")
    if not isinstance(ids, list):
        return ["manifest.question_identity.sample_question_ids must be a list"]
    issues: list[str] = []
    if len(ids) != len(set(ids)):
        issues.append("duplicate Question Identity detected")
    if duplicate_count != 0:
        issues.append(f"duplicate Question Identity count must be 0, got {duplicate_count!r}")
    return issues


def validate_split_policy(authority: dict[str, Any]) -> list[str]:
    split_policy = authority.get("split_policy", {})
    if not isinstance(split_policy, dict):
        return ["authority.split_policy must be an object"]
    if split_policy.get("cross_representation_split_leakage_allowed") is not False:
        return ["same Knowledge Identity cross-split leakage must be disallowed"]
    assignments = split_policy.get("assignments")
    if assignments is None:
        return []
    seen: dict[str, str] = {}
    issues: list[str] = []
    for item in assignments:
        if not isinstance(item, dict):
            issues.append("split assignment must be an object")
            continue
        knowledge_id = item.get("knowledge_id")
        split = item.get("split")
        if not isinstance(knowledge_id, str) or not isinstance(split, str):
            issues.append("split assignment requires knowledge_id and split")
            continue
        previous = seen.setdefault(knowledge_id, split)
        if previous != split:
            issues.append(f"Knowledge Identity crosses splits: {knowledge_id}")
    return issues


def validate_contract(contract: dict[str, Any], task0098_contract: dict[str, Any] | None = None) -> list[str]:
    issues: list[str] = []
    for key, expected in EXPECTED_CONTRACT_VALUES.items():
        if contract.get(key) != expected:
            issues.append(f"contract.{key} expected {expected!r}, got {contract.get(key)!r}")
    issues.extend(validate_frozen_revision("contract.dataset_revision", contract.get("dataset_revision")))
    issues.extend(validate_frozen_revision("contract.annotation_revision", contract.get("annotation_revision")))
    if contract.get("dataset_revision") == contract.get("annotation_revision"):
        issues.append("dataset_revision and annotation_revision must remain separate authorities")
    for key in ("schema_snapshot_path", "inventory_path", "pairing_manifest_path", "license_provenance_path"):
        if not isinstance(contract.get(key), str) or not contract.get(key):
            issues.append(f"contract.{key} must be set")
    if task0098_contract is not None:
        for key, expected in EXPECTED_TASK0098_VALUES.items():
            if task0098_contract.get(key) != expected:
                issues.append(f"TASK-0098 regression: {key} expected {expected!r}, got {task0098_contract.get(key)!r}")
    return issues


def validate_changed_paths(paths: list[str]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        if not path:
            continue
        if path.startswith("evaluation-data/external/pdfqa/materialization/") and not path.endswith("README.md"):
            issues.append(f"external benchmark materialization is not allowed in TASK-0099: {path}")
            continue
        if path.startswith(("opk_rag/runtime", "opk_rag/runtime_v2", "opk_rag/search", "opk_rag/agent")):
            issues.append(f"default runtime path modified: {path}")
            continue
        if "canonical_document" in path.lower() or "pdf_parser" in path.lower():
            issues.append(f"PDF parser or CanonicalDocument runtime path introduced: {path}")
            continue
        if path.startswith(ALLOWED_CHANGED_PREFIXES):
            continue
        if path.startswith("tmp_pdfqa_audit/"):
            continue
        issues.append(f"unexpected TASK-0099 changed path: {path}")
    return issues


def git_status_paths(root: Path = ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def verify_pdfqa_dataset_authority(
    *,
    root: Path = ROOT,
    contract_path: Path | None = None,
    authority_path: Path | None = None,
    inventory_path: Path | None = None,
    schema_path: Path | None = None,
    pairing_path: Path | None = None,
    manifest_path: Path | None = None,
    task0098_contract_path: Path | None = None,
    changed_paths: list[str] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    contract_path = contract_path or root / CONTRACT_PATH.relative_to(ROOT)
    authority_path = authority_path or root / AUTHORITY_PATH.relative_to(ROOT)
    inventory_path = inventory_path or root / INVENTORY_PATH.relative_to(ROOT)
    schema_path = schema_path or root / SCHEMA_PATH.relative_to(ROOT)
    pairing_path = pairing_path or root / PAIRING_PATH.relative_to(ROOT)
    manifest_path = manifest_path or authority_path
    task0098_contract_path = task0098_contract_path or root / TASK0098_CONTRACT_PATH.relative_to(ROOT)

    issues: list[str] = []
    contract = _read_optional(contract_path, "contract", issues)
    authority = _read_optional(authority_path, "authority", issues)
    inventory = _read_optional(inventory_path, "inventory", issues)
    schema = _read_optional(schema_path, "schema", issues)
    pairing = _read_optional(pairing_path, "pairing", issues)
    manifest = _read_optional(manifest_path, "manifest", issues)
    task0098_contract = _read_optional(task0098_contract_path, "TASK-0098 contract", issues)

    if contract:
        issues.extend(validate_contract(contract, task0098_contract or None))
    if authority:
        if authority.get("authority_status") != "frozen":
            issues.append("authority.authority_status must be frozen")
        issues.extend(validate_frozen_revision("authority.dataset.revision", authority.get("dataset", {}).get("revision")))
        issues.extend(
            validate_frozen_revision(
                "authority.annotations.revision", authority.get("annotations", {}).get("revision")
            )
        )
        if authority.get("dataset_update_policy", {}).get("auto_update_allowed") is not False:
            issues.append("authority.dataset_update_policy.auto_update_allowed must be false")
        issues.extend(validate_split_policy(authority))
    if inventory:
        issues.extend(validate_inventory(inventory))
    if schema:
        issues.extend(validate_schema_snapshot(schema))
    if pairing:
        issues.extend(validate_pairing_manifest(pairing))
    if manifest:
        issues.extend(validate_question_identities(manifest))

    if changed_paths is None:
        changed_paths = git_status_paths(root)
    issues.extend(validate_changed_paths(changed_paths))

    report = {
        "schema_version": "opk-rag.task0099.pdfqa-dataset-authority-verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "checked_paths": {
            "contract": relative(contract_path, root),
            "authority": relative(authority_path, root),
            "inventory": relative(inventory_path, root),
            "schema": relative(schema_path, root),
            "pairing": relative(pairing_path, root),
            "task0098_contract": relative(task0098_contract_path, root),
        },
        "changed_paths": changed_paths,
    }
    if write:
        write_json(root / RESULT_DIR.relative_to(ROOT) / "verification.json", report)
    return report


def _read_optional(path: Path, label: str, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"{label} missing: {relative(path)}")
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, Task0099ValidationError) as exc:
        issues.append(f"{label} unreadable: {relative(path)}: {exc}")
        return {}


def relative(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
