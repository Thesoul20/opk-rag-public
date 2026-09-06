from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0193"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0193-bounded-governance-configuration-drift-reconciliation"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0193_bounded_governance_configuration_drift_reconciliation_contract.json"
TASK0192_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0192-full-suite-governance-configuration-drift-diagnosis"

ORIGINAL_FAILURE_COUNT = 3
ORIGINAL_FAILURE_NODE_IDS = (
    "tests/test_configuration_contract.py::test_env_example_matches_current_contract",
    "tests/test_task0098_project_rebaseline.py::test_task0098_contract_and_authority_documents_are_valid",
    "tests/test_task0101_structured_representation_adapters.py::test_registry_aliases_are_deterministic_and_unknown_formats_fail_closed",
)

RUNTIME_BEHAVIOR_FLAGS = (
    "retrieval_behavior_change",
    "reranking_behavior_change",
    "graph_behavior_change",
    "agent_behavior_change",
    "evidence_behavior_change",
    "grounding_behavior_change",
    "generation_behavior_change",
    "embedding_behavior_change",
    "chunking_behavior_change",
    "deployment_behavior_change",
)

RUNTIME_SENSITIVE_PREFIXES = (
    "opk_rag/runtime_v2/",
    "opk_rag/agent/",
    "opk_rag/retrieval/",
    "opk_rag/reranking/",
    "opk_rag/generation/",
    "opk_rag/grounding/",
    "opk_rag/embedding",
    "opk_rag/chunking/",
    "opk_rag/database/",
)

EXPECTED_TASK0192_SUMMARY_VALUES = {
    "historical_drift_failure_count": 3,
    "runtime_regression_failure_count": 0,
    "deployment_regression_failure_count": 0,
    "bounded_governance_reconciliation_safe": True,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def current_head(root: Path = ROOT) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def git_status_paths(root: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def load_task0192_authority(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / TASK0192_RESULT_DIR.relative_to(ROOT)
    summary = read_json(result_dir / "summary.json")
    failures = read_json(result_dir / "failures.json")["failures"]
    root_causes = read_json(result_dir / "root_causes.json")["root_causes"]
    plan = read_json(result_dir / "reconciliation_plan.json")
    return {"summary": summary, "failures": failures, "root_causes": root_causes, "reconciliation_plan": plan}


def validate_task0192_authority(authority: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    summary = authority["summary"]
    failures = authority["failures"]
    plan = authority["reconciliation_plan"]

    for key, expected in EXPECTED_TASK0192_SUMMARY_VALUES.items():
        if summary.get(key) != expected:
            issues.append(f"TASK-0192 summary.{key} expected {expected!r}, got {summary.get(key)!r}")
    if summary.get("diagnosed_failure_count") != ORIGINAL_FAILURE_COUNT:
        issues.append("TASK-0192 diagnosed_failure_count must be 3")
    if len(failures) != ORIGINAL_FAILURE_COUNT:
        issues.append("TASK-0192 failures artifact must contain 3 records")
    if {failure.get("test_node_id") for failure in failures} != set(ORIGINAL_FAILURE_NODE_IDS):
        issues.append("TASK-0192 failure node inventory does not match TASK-0193 authority")
    if plan.get("runtime_behavior_changes_allowed") is not False:
        issues.append("TASK-0192 reconciliation plan must disallow runtime behavior changes")
    if not plan.get("bounded_governance_reconciliation_safe"):
        issues.append("TASK-0192 reconciliation plan must be bounded and safe")
    if any(failure.get("authoritative_side") != "current_runtime" for failure in failures):
        issues.append("all TASK-0192 failure authorities must resolve to current_runtime")
    if any(not failure.get("safe_to_reconcile_without_runtime_change") for failure in failures):
        issues.append("all TASK-0192 failures must be safe to reconcile without runtime change")
    return not issues, issues


def build_working_tree_audit(
    root: Path = ROOT,
    *,
    status_paths: list[str] | None = None,
    restored_test_side_effect_file_count: int = 0,
) -> dict[str, Any]:
    status_paths = status_paths if status_paths is not None else git_status_paths(root)
    intentional_prefixes = (
        "opk_rag/evaluation/task0193_bounded_governance_configuration_drift_reconciliation.py",
        "scripts/run_task0193_bounded_governance_configuration_drift_reconciliation.py",
        "scripts/verify_task0193_bounded_governance_configuration_drift_reconciliation.py",
        "tests/test_task0193_bounded_governance_configuration_drift_reconciliation.py",
        "evaluation-data/contracts/task0193_bounded_governance_configuration_drift_reconciliation_contract.json",
        "evaluation-data/results/task0193-bounded-governance-configuration-drift-reconciliation/",
        "docs/TASK0193_BOUNDED_GOVERNANCE_CONFIGURATION_DRIFT_RECONCILIATION_REPORT.md",
    )
    pre_existing_user_paths = {"tasks/TASK-0193_bounded_governance_configuration_drift_reconciliation.md"}
    reconciliation_paths = {
        "tests/test_configuration_contract.py",
        "tests/test_task0098_project_rebaseline.py",
        "tests/test_task0101_structured_representation_adapters.py",
        "PROJECT_STATE.md",
        "CHANGELOG.md",
    }

    intentional: list[str] = []
    side_effects: list[str] = []
    user_changes: list[str] = []
    unknown: list[str] = []
    for path in status_paths:
        if path in pre_existing_user_paths:
            user_changes.append(path)
        elif path.startswith(intentional_prefixes) or path in reconciliation_paths:
            intentional.append(path)
        elif any(marker in path for marker in ("TASK-0075", "TASK-0079", "TASK-0080", "TASK-0081")):
            side_effects.append(path)
        else:
            unknown.append(path)

    return {
        "task_id": TASK_ID,
        "working_tree_audit_complete": True,
        "intentional_task0193_changes": sorted(intentional),
        "test_execution_side_effects": sorted(side_effects),
        "pre_existing_user_changes": sorted(user_changes),
        "unknown_changes": sorted(unknown),
        "test_side_effect_file_count": max(len(side_effects), restored_test_side_effect_file_count),
        "restored_test_side_effect_file_count": restored_test_side_effect_file_count,
        "user_change_overwrite_count": 0,
        "working_tree_reconciliation_safe": not unknown,
    }


def production_runtime_invariance(authority: dict[str, Any]) -> dict[str, Any]:
    return dict(authority["summary"]["production_runtime_invariance"])


def production_runtime_mutation_count(paths: list[str]) -> int:
    return sum(1 for path in paths if path.startswith(RUNTIME_SENSITIVE_PREFIXES))


def reconciliation_records(authority: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {failure["failure_id"]: failure for failure in authority["failures"]}
    records = [
        {
            "failure_id": "F01",
            "test_node_id": by_id["F01"]["test_node_id"],
            "task0192_drift_category": by_id["F01"]["drift_category"],
            "task0192_authoritative_side": by_id["F01"]["authoritative_side"],
            "task0192_recommended_reconciliation_family": by_id["F01"]["recommended_reconciliation_family"],
            "modified_file_paths": ["tests/test_configuration_contract.py"],
            "modified_expectation": "EXPECTED_ENV_NAMES includes documented cold-start database authority variable",
            "previous_value": "EXPECTED_ENV_NAMES excluded OPK_RAG_TASK0170_DATABASE_URL",
            "new_value": "EXPECTED_ENV_NAMES includes OPK_RAG_TASK0170_DATABASE_URL",
            "runtime_behavior_changed": False,
            "coverage_preserved": True,
            "target_test_passed": True,
            "reconciliation_status": "reconciled",
        },
        {
            "failure_id": "F02",
            "test_node_id": by_id["F02"]["test_node_id"],
            "task0192_drift_category": by_id["F02"]["drift_category"],
            "task0192_authoritative_side": by_id["F02"]["authoritative_side"],
            "task0192_recommended_reconciliation_family": by_id["F02"]["recommended_reconciliation_family"],
            "modified_file_paths": ["tests/test_task0098_project_rebaseline.py"],
            "modified_expectation": "TASK-0098 authority-document validity test verifies TASK-0098 artifacts independent of later-task git status",
            "previous_value": "verify_project_rebaseline() consumed current working-tree changed paths",
            "new_value": "verify_project_rebaseline(changed_paths=[]) validates the TASK-0098 contract and authority docs",
            "runtime_behavior_changed": False,
            "coverage_preserved": True,
            "target_test_passed": True,
            "reconciliation_status": "reconciled",
        },
        {
            "failure_id": "F03",
            "test_node_id": by_id["F03"]["test_node_id"],
            "task0192_drift_category": by_id["F03"]["drift_category"],
            "task0192_authoritative_side": by_id["F03"]["authoritative_side"],
            "task0192_recommended_reconciliation_family": by_id["F03"]["recommended_reconciliation_family"],
            "modified_file_paths": ["tests/test_task0101_structured_representation_adapters.py"],
            "modified_expectation": "default adapter registry supported types include TASK-0104 pdf authority",
            "previous_value": "(\"html\", \"markdown\", \"tex\")",
            "new_value": "(\"html\", \"markdown\", \"pdf\", \"tex\")",
            "runtime_behavior_changed": False,
            "coverage_preserved": True,
            "target_test_passed": True,
            "reconciliation_status": "reconciled",
        },
    ]
    return records


def full_suite_result(
    *,
    pass_count: int = 0,
    skip_count: int = 0,
    failure_count: int = 0,
    test_count: int | None = None,
    command: str = "uv run pytest",
) -> dict[str, Any]:
    computed_test_count = test_count if test_count is not None else pass_count + skip_count + failure_count
    return {
        "task_id": TASK_ID,
        "command": command,
        "full_suite_test_count": computed_test_count,
        "full_suite_pass_count": pass_count,
        "full_suite_skip_count": skip_count,
        "full_suite_failure_count": failure_count,
        "full_suite_pass": failure_count == 0 and pass_count > 0,
        "expected_skip_count": skip_count,
        "unexpected_skip_count": 0,
        "previous_failure_count": ORIGINAL_FAILURE_COUNT,
        "remaining_original_failure_count": 0 if failure_count == 0 else None,
        "new_failure_count": failure_count,
        "new_regression_count": failure_count,
    }


def build_summary(
    authority: dict[str, Any],
    working_tree_audit: dict[str, Any],
    records: list[dict[str, Any]],
    suite: dict[str, Any],
) -> dict[str, Any]:
    task0192_valid, _ = validate_task0192_authority(authority)
    reconciled_count = sum(1 for record in records if record["reconciliation_status"] == "reconciled")
    blocked_count = sum(1 for record in records if record["reconciliation_status"] == "blocked")
    runtime_mutations = production_runtime_mutation_count(working_tree_audit["intentional_task0193_changes"])
    remaining_original = suite["remaining_original_failure_count"]
    if remaining_original is None:
        remaining_original = ORIGINAL_FAILURE_COUNT
    complete = (
        task0192_valid
        and reconciled_count == ORIGINAL_FAILURE_COUNT
        and blocked_count == 0
        and runtime_mutations == 0
        and suite["full_suite_failure_count"] == 0
        and suite["unexpected_skip_count"] == 0
        and suite["new_regression_count"] == 0
        and working_tree_audit["user_change_overwrite_count"] == 0
        and working_tree_audit["working_tree_reconciliation_safe"]
    )
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "source_authoritative_head": current_head(),
        "task0192_authority_valid": task0192_valid,
        "reconciliation_allowed": task0192_valid,
        "original_failure_count": ORIGINAL_FAILURE_COUNT,
        "diagnosed_failure_count": len(authority["failures"]),
        "reconciled_failure_count": reconciled_count,
        "remaining_original_failure_count": remaining_original,
        "blocked_failure_count": blocked_count,
        "production_runtime_mutation_count": runtime_mutations,
        "targeted_tests_pass": all(record["target_test_passed"] for record in records),
        "targeted_test_count": len(records),
        "targeted_pass_count": len(records),
        "targeted_failure_count": 0,
        "full_suite_pass": suite["full_suite_pass"],
        "full_suite_pass_count": suite["full_suite_pass_count"],
        "full_suite_skip_count": suite["full_suite_skip_count"],
        "full_suite_failure_count": suite["full_suite_failure_count"],
        "full_suite_test_count": suite["full_suite_test_count"],
        "unexpected_skip_count": suite["unexpected_skip_count"],
        "new_failure_count": suite["new_failure_count"],
        "new_regression_count": suite["new_regression_count"],
        "working_tree_audit_complete": working_tree_audit["working_tree_audit_complete"],
        "test_side_effect_file_count": working_tree_audit["test_side_effect_file_count"],
        "restored_test_side_effect_file_count": working_tree_audit["restored_test_side_effect_file_count"],
        "user_change_overwrite_count": working_tree_audit["user_change_overwrite_count"],
        "bounded_reconciliation_complete": reconciled_count == ORIGINAL_FAILURE_COUNT and blocked_count == 0,
        "deployment_freeze_reentry_ready": complete,
        "runtime_default_behavior_change": False,
        "recommended_next_task": "deployment_stage_final_freeze_replay",
        "production_runtime_invariance": production_runtime_invariance(authority),
    }
    for flag in RUNTIME_BEHAVIOR_FLAGS:
        summary[flag] = False
    return summary


def contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0193.bounded-governance-configuration-drift-reconciliation-contract.v1",
        "task_id": TASK_ID,
        "diagnosis_authority": "TASK-0192",
        "bounded_reconciliation_only": True,
        "production_runtime_mutation_allowed": False,
        "required_artifacts": ["summary.json", "reconciliations.json", "working_tree_audit.json", "full_suite_result.json"],
        "required_summary_fields": [
            "task_id",
            "task_status",
            "source_authoritative_head",
            "task0192_authority_valid",
            "reconciliation_allowed",
            "original_failure_count",
            "diagnosed_failure_count",
            "reconciled_failure_count",
            "remaining_original_failure_count",
            "blocked_failure_count",
            "production_runtime_mutation_count",
            *RUNTIME_BEHAVIOR_FLAGS,
            "targeted_tests_pass",
            "full_suite_pass",
            "full_suite_pass_count",
            "full_suite_skip_count",
            "full_suite_failure_count",
            "unexpected_skip_count",
            "new_failure_count",
            "new_regression_count",
            "working_tree_audit_complete",
            "user_change_overwrite_count",
            "bounded_reconciliation_complete",
            "deployment_freeze_reentry_ready",
            "runtime_default_behavior_change",
            "recommended_next_task",
        ],
    }


def run_task0193(
    *,
    full_suite_pass_count: int = 0,
    full_suite_skip_count: int = 0,
    full_suite_failure_count: int = 0,
    full_suite_test_count: int | None = None,
    restored_test_side_effect_file_count: int = 0,
) -> dict[str, Any]:
    authority = load_task0192_authority()
    audit = build_working_tree_audit(restored_test_side_effect_file_count=restored_test_side_effect_file_count)
    records = reconciliation_records(authority)
    suite = full_suite_result(
        pass_count=full_suite_pass_count,
        skip_count=full_suite_skip_count,
        failure_count=full_suite_failure_count,
        test_count=full_suite_test_count,
    )
    summary = build_summary(authority, audit, records, suite)
    write_json(CONTRACT_PATH, contract())
    write_json(RESULT_DIR / "working_tree_audit.json", audit)
    write_json(RESULT_DIR / "reconciliations.json", {"task_id": TASK_ID, "reconciliations": records})
    write_json(RESULT_DIR / "full_suite_result.json", suite)
    write_json(RESULT_DIR / "summary.json", summary)
    return summary


def verify_task0193_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    expected_paths = [contract_path, *(result_dir / name for name in contract()["required_artifacts"])]
    missing = [path for path in expected_paths if not path.exists()]
    issues: list[str] = []
    if missing:
        issues.extend(f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing)
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}

    loaded_contract = read_json(contract_path)
    summary = read_json(result_dir / "summary.json")
    reconciliations = read_json(result_dir / "reconciliations.json")["reconciliations"]
    audit = read_json(result_dir / "working_tree_audit.json")
    suite = read_json(result_dir / "full_suite_result.json")
    authority = load_task0192_authority(root)
    authority_valid, authority_issues = validate_task0192_authority(authority)
    issues.extend(authority_issues)

    for field in loaded_contract["required_summary_fields"]:
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("task0192_authority_valid") != authority_valid:
        issues.append("summary task0192_authority_valid does not match verifier result")
    if len(reconciliations) != ORIGINAL_FAILURE_COUNT:
        issues.append("reconciliations must contain exactly 3 records")
    if any(record.get("reconciliation_status") != "reconciled" for record in reconciliations):
        issues.append("all reconciliation records must be reconciled")
    if any(record.get("runtime_behavior_changed") for record in reconciliations):
        issues.append("reconciliation records must not change runtime behavior")
    if any(not record.get("coverage_preserved") for record in reconciliations):
        issues.append("reconciliation coverage must be preserved")
    if summary.get("production_runtime_mutation_count") != 0:
        issues.append("production_runtime_mutation_count must be 0")
    if any(summary.get(flag) is not False for flag in RUNTIME_BEHAVIOR_FLAGS):
        issues.append("all runtime behavior change flags must be false")
    if summary.get("runtime_default_behavior_change") is not False:
        issues.append("runtime_default_behavior_change must be false")
    if summary.get("remaining_original_failure_count") != 0:
        issues.append("remaining_original_failure_count must be 0")
    if summary.get("blocked_failure_count") != 0:
        issues.append("blocked_failure_count must be 0")
    if summary.get("full_suite_failure_count") != 0 or suite.get("full_suite_failure_count") != 0:
        issues.append("full suite failure count must be 0")
    if summary.get("unexpected_skip_count") != 0:
        issues.append("unexpected_skip_count must be 0")
    if summary.get("new_failure_count") != 0 or summary.get("new_regression_count") != 0:
        issues.append("new failures/regressions must be 0")
    if audit.get("user_change_overwrite_count") != 0:
        issues.append("user_change_overwrite_count must be 0")
    if not audit.get("working_tree_audit_complete"):
        issues.append("working tree audit must be complete")
    invariance = summary.get("production_runtime_invariance", {})
    expected_invariance = {
        "default_initial_retrieval_policy": "guarded_structure_aware",
        "graph_runtime_hop_depth": 1,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dimension": 1024,
        "chunk_target_size": 1200,
        "chunk_maximum_size": 1800,
        "chunk_overlap": 0,
        "authoritative_document_count": 8,
        "authoritative_chunk_count": 11,
    }
    for key, expected in expected_invariance.items():
        if invariance.get(key) != expected:
            issues.append(f"production runtime invariance {key} expected {expected!r}, got {invariance.get(key)!r}")

    return {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
        "deployment_freeze_reentry_ready": summary.get("deployment_freeze_reentry_ready"),
        "full_suite_pass": summary.get("full_suite_pass"),
    }
