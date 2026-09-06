from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0098"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0098-project-rebaseline"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0098_project_rebaseline_contract.json"
ROADMAP_PATH = ROOT / "docs" / "HETEROGENEOUS_DOCUMENT_AND_ADAPTIVE_GRAPHRAG_ROADMAP.md"
REPORT_PATH = ROOT / "docs" / "TASK0098_HETEROGENEOUS_DOCUMENT_ADAPTIVE_GRAPHRAG_REBASELINE_REPORT.md"
README_PATH = ROOT / "README.md"
SHOWCASE_AUTHORITY_PATH = ROOT / "docs" / "PROJECT_SHOWCASE_AUTHORITY.md"

EXPECTED_CONTRACT_VALUES: dict[str, Any] = {
    "task_id": TASK_ID,
    "canonical_document_required": True,
    "knowledge_identity_required": True,
    "representation_identity_required": True,
    "primary_document_benchmark": "pdfQA",
    "primary_real_world_graph_benchmark": "WildGraphBench",
    "primary_graph_necessity_benchmark": "GraphRAG-Bench",
    "new_large_internal_benchmark_default_allowed": False,
    "legacy_benchmark_role": "regression",
    "graph_replaces_agent_rag": False,
    "graph_is_optional_agent_retrieval_capability": True,
    "final_architecture_target": "adaptive_agentic_rag",
    "runtime_behavior_modified": False,
    "retrieval_behavior_modified": False,
    "reranker_behavior_modified": False,
    "graph_runtime_modified": False,
    "external_dataset_downloaded": False,
}

REQUIRED_ROADMAP_TERMS = (
    "Heterogeneous Document Intelligence",
    "Adaptive Agentic RAG",
    "CanonicalDocument",
    "Knowledge Identity",
    "Representation Identity",
    "pdfQA",
    "WildGraphBench",
    "GraphRAG-Bench",
    "Legacy OPK Regression Authority",
    "Graph Necessity",
    "GraphRAG is not a replacement for Agent RAG",
)

REQUIRED_README_TERMS = (
    "Heterogeneous Document Intelligence",
    "Adaptive Agentic RAG",
    "CanonicalDocument",
    "pdfQA",
    "WildGraphBench",
    "GraphRAG-Bench",
)

ALLOWED_CHANGED_PREFIXES = (
    "README.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "PROJECT_STATE.md",
    "docs/",
    "evaluation-data/contracts/task0098_project_rebaseline_contract.json",
    "evaluation-data/results/task0098-project-rebaseline/",
    "opk_rag/evaluation/task0098_project_rebaseline.py",
    "scripts/verify_task0098_project_rebaseline.py",
    "tasks/TASK-0098-",
    "tests/test_task0098_project_rebaseline.py",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


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


def validate_contract(contract: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key, expected in EXPECTED_CONTRACT_VALUES.items():
        if contract.get(key) != expected:
            issues.append(f"contract.{key} expected {expected!r}, got {contract.get(key)!r}")

    benchmark_authority = contract.get("benchmark_authority", {})
    expected_authority = {
        "pdfQA": "heterogeneous_document_intelligence_agent_rag_primary",
        "WildGraphBench": "real_world_cross_document_graphrag",
        "GraphRAG-Bench": "graph_necessity_controlled_graphrag",
        "Legacy OPK": "legacy_regression_authority",
    }
    for key, expected in expected_authority.items():
        if benchmark_authority.get(key) != expected:
            issues.append(f"contract.benchmark_authority.{key} expected {expected!r}")

    if contract.get("roadmap_authority_document") != "docs/HETEROGENEOUS_DOCUMENT_AND_ADAPTIVE_GRAPHRAG_ROADMAP.md":
        issues.append("contract.roadmap_authority_document must point to the TASK-0098 roadmap authority")

    if contract.get("task_scope") != "documentation_governance_roadmap_rebaseline":
        issues.append("contract.task_scope must remain documentation_governance_roadmap_rebaseline")
    return issues


def validate_text(path: Path, required_terms: tuple[str, ...], label: str) -> list[str]:
    if not path.exists():
        return [f"{label} missing: {relative(path)}"]
    text = path.read_text(encoding="utf-8")
    return [f"{label} missing required term: {term}" for term in required_terms if term not in text]


def validate_readme_authority(readme_path: Path, root: Path) -> list[str]:
    direct_issues = validate_text(readme_path, REQUIRED_README_TERMS, "README")
    if not direct_issues:
        return []
    if not readme_path.exists():
        return direct_issues
    text = readme_path.read_text(encoding="utf-8")
    showcase_path = root / SHOWCASE_AUTHORITY_PATH.relative_to(ROOT)
    if "docs/PROJECT_SHOWCASE_AUTHORITY.md" in text and showcase_path.exists():
        return []
    return direct_issues


def validate_changed_paths(paths: list[str]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        if not path:
            continue
        if path.startswith("evaluation-data/external/") and not path.endswith("README.md"):
            issues.append(f"external benchmark corpus materialization is not allowed: {path}")
            continue
        if path.startswith("evaluation-data/results/task0097-") or path.startswith("evaluation-data/contracts/task0097_"):
            issues.append(f"TASK-0097 artifact modified: {path}")
            continue
        if path.startswith(ALLOWED_CHANGED_PREFIXES):
            continue
        issues.append(f"unexpected runtime or non-governance path modified: {path}")
    return issues


def verify_project_rebaseline(
    *,
    root: Path = ROOT,
    contract_path: Path | None = None,
    roadmap_path: Path | None = None,
    readme_path: Path | None = None,
    changed_paths: list[str] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    contract_path = contract_path or root / CONTRACT_PATH.relative_to(ROOT)
    roadmap_path = roadmap_path or root / ROADMAP_PATH.relative_to(ROOT)
    readme_path = readme_path or root / README_PATH.relative_to(ROOT)
    result_dir = root / RESULT_DIR.relative_to(ROOT)

    issues: list[str] = []
    if not contract_path.exists():
        issues.append(f"contract missing: {relative(contract_path, root)}")
        contract: dict[str, Any] = {}
    else:
        contract = read_json(contract_path)
        issues.extend(validate_contract(contract))

    issues.extend(validate_text(roadmap_path, REQUIRED_ROADMAP_TERMS, "roadmap"))
    issues.extend(validate_readme_authority(readme_path, root))

    if changed_paths is None:
        changed_paths = git_status_paths(root)
    issues.extend(validate_changed_paths(changed_paths))

    report = {
        "schema_version": "opk-rag.task0098.project-rebaseline-verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "checked_paths": {
            "contract": relative(contract_path, root),
            "roadmap": relative(roadmap_path, root),
            "readme": relative(readme_path, root),
        },
        "changed_paths": changed_paths,
    }
    if write:
        write_json(result_dir / "verification.json", report)
    return report


def relative(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
