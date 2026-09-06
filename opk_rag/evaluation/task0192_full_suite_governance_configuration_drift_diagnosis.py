from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0192"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0192-full-suite-governance-configuration-drift-diagnosis"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0192_full_suite_governance_configuration_drift_diagnosis_contract.json"
TASK0191_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0191-cold-start-deployment-stage-closeout"

DRIFT_CATEGORIES = (
    "test_expectation_drift",
    "contract_drift",
    "configuration_drift",
    "default_policy_drift",
    "project_state_drift",
    "documentation_drift",
    "fixture_drift",
    "environment_expectation_drift",
    "obsolete_compatibility_expectation",
    "runtime_regression",
    "unknown",
)

RECONCILIATION_FAMILIES = (
    "update_test_expectation",
    "update_evaluation_contract",
    "update_configuration_authority",
    "update_project_state",
    "update_documentation",
    "remove_obsolete_test",
    "environment_gate_correction",
    "production_runtime_fix",
    "needs_further_diagnosis",
)

FULL_SUITE_INVENTORY = {
    "command": "uv run pytest",
    "executed_at_local_date": "2026-08-27",
    "full_suite_test_count": 1949,
    "full_suite_pass_count": 1858,
    "full_suite_skip_count": 88,
    "full_suite_failure_count": 3,
    "full_suite_pass": False,
    "duration_seconds": 240.39,
}


def current_head(root: Path = ROOT) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def production_runtime_invariance() -> dict[str, Any]:
    summary = read_json(TASK0191_RESULT_DIR / "summary.json")
    baseline = read_json(TASK0191_RESULT_DIR / "deployment_baseline.json")
    return {
        "default_initial_retrieval_policy": baseline["retrieval_policy"],
        "graph_runtime_hop_depth": 1,
        "embedding_model": summary["embedding_model"],
        "embedding_dimension": summary["embedding_dimension"],
        "chunk_target_size": summary["target_chunk_size"],
        "chunk_maximum_size": summary["maximum_chunk_size"],
        "chunk_overlap": summary["overlap"],
        "authoritative_document_count": summary["authoritative_document_count"],
        "authoritative_chunk_count": summary["authoritative_chunk_count"],
        "q01_q07_e2e_pass": summary["q01_q07_e2e_pass"],
        "known_deployment_blocker_count": summary["known_deployment_blocker_count"],
        "new_regression_count": summary["new_regression_count"],
        "deployment_stage_closeout_decision": summary["deployment_stage_closeout_decision"],
        "deployment_baseline_digest": summary["deployment_baseline_digest"],
        "runtime_default_behavior_change": summary["runtime_default_behavior_change"],
    }


def failure_records() -> list[dict[str, Any]]:
    return [
        {
            "failure_id": "F01",
            "test_node_id": "tests/test_configuration_contract.py::test_env_example_matches_current_contract",
            "test_file": "tests/test_configuration_contract.py",
            "test_name": "test_env_example_matches_current_contract",
            "failure_type": "AssertionError",
            "failure_message": "Extra item in .env.example names: OPK_RAG_TASK0170_DATABASE_URL",
            "first_failure_stage": "configuration_contract_static_inventory",
            "first_failing_assertion": "assert _env_example_names() == EXPECTED_ENV_NAMES",
            "expected_value": "EXPECTED_ENV_NAMES excludes OPK_RAG_TASK0170_DATABASE_URL",
            "actual_value": ".env.example includes OPK_RAG_TASK0170_DATABASE_URL",
            "expected_behavior": "The committed .env.example key set exactly matches the historical static EXPECTED_ENV_NAMES set.",
            "actual_behavior": ".env.example documents the cold-start isolated database URL authority introduced after the historical test expectation.",
            "expected_authority_source": "tests/test_configuration_contract.py static EXPECTED_ENV_NAMES",
            "actual_authority_source": ".env.example, CHANGELOG.md, TASK-0171/TASK-0172 cold-start database authority docs",
            "expected_authority_revision": "pre-TASK-0171 configuration contract expectation",
            "actual_authority_revision": "a41b77f9 fix(reproducibility): harden cold-start database bootstrap",
            "historical_expected_authority": "configuration contract test fixture",
            "current_runtime_authority": "cold-start database configuration authority",
            "authority_divergence_detected": True,
            "authoritative_side": "current_runtime",
            "drift_category": "configuration_drift",
            "runtime_behavior_affected": False,
            "deployment_behavior_affected": False,
            "benchmark_behavior_affected": False,
            "new_regression": False,
            "historical_drift": True,
            "drift_origin_task": "TASK-0171",
            "last_known_consistent_task": "TASK-0170",
            "first_known_divergent_task": "TASK-0171",
            "recommended_reconciliation_family": "update_configuration_authority",
            "safe_to_reconcile_without_runtime_change": True,
            "root_cause_id": "RC01",
        },
        {
            "failure_id": "F02",
            "test_node_id": "tests/test_task0098_project_rebaseline.py::test_task0098_contract_and_authority_documents_are_valid",
            "test_file": "tests/test_task0098_project_rebaseline.py",
            "test_name": "test_task0098_contract_and_authority_documents_are_valid",
            "failure_type": "AssertionError",
            "failure_message": "assert result['status'] == 'valid'; actual status is 'invalid'",
            "first_failure_stage": "task0098_governance_changed_path_validation",
            "first_failing_assertion": "assert result[\"status\"] == \"valid\"",
            "expected_value": "valid",
            "actual_value": "invalid",
            "expected_behavior": "TASK-0098 verifier remains valid while allowing only the TASK-0098-era governance changed-path set.",
            "actual_behavior": "Current repository contains later task governance files, so the historical TASK-0098 changed-path whitelist fails closed.",
            "expected_authority_source": "opk_rag.evaluation.task0098_project_rebaseline.ALLOWED_CHANGED_PREFIXES",
            "actual_authority_source": "current task governance workflow and git status containing TASK-0192 task/artifact paths",
            "expected_authority_revision": "TASK-0098 single-task rebaseline verifier",
            "actual_authority_revision": "current post-TASK-0098 governance task sequence through TASK-0192",
            "historical_expected_authority": "TASK-0098 verifier contract",
            "current_runtime_authority": "current repository governance state",
            "authority_divergence_detected": True,
            "authoritative_side": "current_runtime",
            "drift_category": "test_expectation_drift",
            "runtime_behavior_affected": False,
            "deployment_behavior_affected": False,
            "benchmark_behavior_affected": True,
            "new_regression": False,
            "historical_drift": True,
            "drift_origin_task": "unknown",
            "last_known_consistent_task": "TASK-0098",
            "first_known_divergent_task": "unknown",
            "recommended_reconciliation_family": "update_test_expectation",
            "safe_to_reconcile_without_runtime_change": True,
            "root_cause_id": "RC02",
        },
        {
            "failure_id": "F03",
            "test_node_id": "tests/test_task0101_structured_representation_adapters.py::test_registry_aliases_are_deterministic_and_unknown_formats_fail_closed",
            "test_file": "tests/test_task0101_structured_representation_adapters.py",
            "test_name": "test_registry_aliases_are_deterministic_and_unknown_formats_fail_closed",
            "failure_type": "AssertionError",
            "failure_message": "registry.supported_types() includes pdf; historical expectation was ('html', 'markdown', 'tex')",
            "first_failure_stage": "structured_adapter_registry_supported_types",
            "first_failing_assertion": "assert registry.supported_types() == (\"html\", \"markdown\", \"tex\")",
            "expected_value": "(\"html\", \"markdown\", \"tex\")",
            "actual_value": "(\"html\", \"markdown\", \"pdf\", \"tex\")",
            "expected_behavior": "The TASK-0101 structured adapter registry exposes only html, markdown, and tex.",
            "actual_behavior": "The current registry also exposes the TASK-0104 canonical PDF adapter.",
            "expected_authority_source": "tests/test_task0101_structured_representation_adapters.py historical assertion",
            "actual_authority_source": "opk_rag.document_adapters.registry and TASK-0104 canonical PDF adapter contract",
            "expected_authority_revision": "5f98e58f feat(document-adapters): add structured representation adapters v1",
            "actual_authority_revision": "3e9f43a6 feat(document-adapters): add canonical PDF adapter v1",
            "historical_expected_authority": "TASK-0101 adapter registry compatibility test",
            "current_runtime_authority": "current structured document adapter registry",
            "authority_divergence_detected": True,
            "authoritative_side": "current_runtime",
            "drift_category": "obsolete_compatibility_expectation",
            "runtime_behavior_affected": False,
            "deployment_behavior_affected": False,
            "benchmark_behavior_affected": True,
            "new_regression": False,
            "historical_drift": True,
            "drift_origin_task": "TASK-0104",
            "last_known_consistent_task": "TASK-0101",
            "first_known_divergent_task": "TASK-0104",
            "recommended_reconciliation_family": "update_test_expectation",
            "safe_to_reconcile_without_runtime_change": True,
            "root_cause_id": "RC03",
        },
    ]


def root_causes(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {failure["root_cause_id"]: failure for failure in failures}
    return [
        {
            "root_cause_id": "RC01",
            "failure_ids": ["F01"],
            "root_cause": "The environment contract test fixture was not reconciled after the cold-start isolated database variable became documented configuration authority.",
            "drift_category": by_id["RC01"]["drift_category"],
            "authoritative_side": "current_runtime",
            "recommended_reconciliation_family": "update_configuration_authority",
            "safe_to_reconcile_without_runtime_change": True,
        },
        {
            "root_cause_id": "RC02",
            "failure_ids": ["F02"],
            "root_cause": "The TASK-0098 verifier still treats its own historical changed-path whitelist as globally current for later governance tasks.",
            "drift_category": by_id["RC02"]["drift_category"],
            "authoritative_side": "current_runtime",
            "recommended_reconciliation_family": "update_test_expectation",
            "safe_to_reconcile_without_runtime_change": True,
        },
        {
            "root_cause_id": "RC03",
            "failure_ids": ["F03"],
            "root_cause": "The TASK-0101 supported-types assertion was not updated after TASK-0104 promoted the canonical PDF adapter into the registry.",
            "drift_category": by_id["RC03"]["drift_category"],
            "authoritative_side": "current_runtime",
            "recommended_reconciliation_family": "update_test_expectation",
            "safe_to_reconcile_without_runtime_change": True,
        },
    ]


def build_reconciliation_plan(failures: list[dict[str, Any]], causes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "bounded_governance_reconciliation_safe": True,
        "deployment_freeze_reentry_eligible": True,
        "recommended_next_task": "bounded_governance_configuration_reconciliation",
        "production_runtime_fix_required": False,
        "runtime_behavior_changes_allowed": False,
        "reconciliation_steps": [
            {
                "root_cause_id": cause["root_cause_id"],
                "family": cause["recommended_reconciliation_family"],
                "preserve_runtime_behavior": True,
                "failure_ids": cause["failure_ids"],
            }
            for cause in causes
        ],
        "diagnosed_failure_count": len(failures),
    }


def build_summary(failures: list[dict[str, Any]], causes: list[dict[str, Any]]) -> dict[str, Any]:
    invariance = production_runtime_invariance()
    runtime_regressions = [failure for failure in failures if failure["drift_category"] == "runtime_regression"]
    unknown = [failure for failure in failures if failure["drift_category"] == "unknown"]
    deployment_regressions = [failure for failure in failures if failure["deployment_behavior_affected"]]
    all_authority_resolved = all(failure["authoritative_side"] != "undetermined" for failure in failures)
    safe = not runtime_regressions and not deployment_regressions and not unknown and all_authority_resolved
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if safe else "partial",
        "source_authoritative_head": current_head(),
        **FULL_SUITE_INVENTORY,
        "diagnosed_failure_count": len(failures),
        "historical_drift_failure_count": sum(1 for failure in failures if failure["historical_drift"]),
        "runtime_regression_failure_count": len(runtime_regressions),
        "deployment_regression_failure_count": len(deployment_regressions),
        "unknown_failure_count": len(unknown),
        "unique_root_cause_count": len(causes),
        "shared_root_cause_detected": len(causes) < len(failures),
        "all_failures_have_first_loss_stage": all(bool(failure["first_failure_stage"]) for failure in failures),
        "all_failures_have_authority_resolution": all_authority_resolved,
        "production_runtime_mutation_count": 0,
        "runtime_default_behavior_change": invariance["runtime_default_behavior_change"],
        "new_regression_count": 0,
        "bounded_governance_reconciliation_safe": safe,
        "deployment_freeze_reentry_eligible": safe,
        "recommended_next_task": "bounded_governance_configuration_reconciliation" if safe else "runtime_defect_remediation",
        "production_runtime_invariance": invariance,
    }


def contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0192.full-suite-governance-configuration-drift-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "diagnose_before_repair": True,
        "production_runtime_mutation_allowed": False,
        "drift_categories": list(DRIFT_CATEGORIES),
        "reconciliation_families": list(RECONCILIATION_FAMILIES),
        "required_artifacts": ["summary.json", "failures.json", "root_causes.json", "reconciliation_plan.json"],
        "required_summary_fields": [
            "task_id",
            "task_status",
            "source_authoritative_head",
            "full_suite_pass",
            "full_suite_pass_count",
            "full_suite_skip_count",
            "full_suite_failure_count",
            "diagnosed_failure_count",
            "historical_drift_failure_count",
            "runtime_regression_failure_count",
            "deployment_regression_failure_count",
            "unknown_failure_count",
            "unique_root_cause_count",
            "shared_root_cause_detected",
            "all_failures_have_first_loss_stage",
            "all_failures_have_authority_resolution",
            "production_runtime_mutation_count",
            "new_regression_count",
            "bounded_governance_reconciliation_safe",
            "deployment_freeze_reentry_eligible",
            "recommended_next_task",
        ],
    }


def run_task0192() -> dict[str, Any]:
    failures = failure_records()
    causes = root_causes(failures)
    summary = build_summary(failures, causes)
    plan = build_reconciliation_plan(failures, causes)
    write_json(CONTRACT_PATH, contract())
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "failures.json", {"task_id": TASK_ID, "failures": failures})
    write_json(RESULT_DIR / "root_causes.json", {"task_id": TASK_ID, "root_causes": causes})
    write_json(RESULT_DIR / "reconciliation_plan.json", plan)
    return summary


def verify_task0192_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    missing = [path for path in [contract_path, *(result_dir / name for name in contract()["required_artifacts"])] if not path.exists()]
    issues: list[str] = []
    if missing:
        issues.extend(f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing)
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}

    loaded_contract = read_json(contract_path)
    summary = read_json(result_dir / "summary.json")
    failures = read_json(result_dir / "failures.json")["failures"]
    causes = read_json(result_dir / "root_causes.json")["root_causes"]
    plan = read_json(result_dir / "reconciliation_plan.json")

    for field in loaded_contract["required_summary_fields"]:
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("diagnosed_failure_count") != summary.get("full_suite_failure_count"):
        issues.append("diagnosed_failure_count must equal full_suite_failure_count")
    if summary.get("diagnosed_failure_count") != len(failures):
        issues.append("failure artifact count must equal diagnosed_failure_count")
    if summary.get("unique_root_cause_count") != len(causes):
        issues.append("root cause artifact count must equal unique_root_cause_count")
    if any(failure.get("drift_category") not in DRIFT_CATEGORIES for failure in failures):
        issues.append("failure drift_category outside controlled taxonomy")
    if any(failure.get("recommended_reconciliation_family") not in RECONCILIATION_FAMILIES for failure in failures):
        issues.append("failure reconciliation family outside controlled taxonomy")
    if summary.get("production_runtime_mutation_count") != 0:
        issues.append("production runtime mutation count must remain zero")
    if summary.get("runtime_regression_failure_count") != 0:
        issues.append("runtime regression count must remain zero for complete diagnosis")
    if summary.get("deployment_regression_failure_count") != 0:
        issues.append("deployment regression count must remain zero for complete diagnosis")
    if summary.get("unknown_failure_count") != 0:
        issues.append("unknown failure count must remain zero for complete diagnosis")
    if not plan.get("bounded_governance_reconciliation_safe"):
        issues.append("reconciliation plan must be safe when all failures resolve to historical drift")

    return {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
        "diagnosed_failure_count": summary.get("diagnosed_failure_count"),
        "bounded_governance_reconciliation_safe": summary.get("bounded_governance_reconciliation_safe"),
        "deployment_freeze_reentry_eligible": summary.get("deployment_freeze_reentry_eligible"),
    }
