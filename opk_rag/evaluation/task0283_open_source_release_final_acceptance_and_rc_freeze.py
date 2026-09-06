from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from opk_rag.evaluation import task0282_opk_rag_open_source_release_readiness_and_repository_finalization as task0282

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evaluation-data/results/task0283-open-source-release-final-acceptance-and-rc-freeze"
TASK = ROOT / "tasks/TASK-0283_open_source_release_final_acceptance_and_rc_freeze.md"
REPORT = ROOT / "docs/TASK0283_OPEN_SOURCE_RELEASE_FINAL_ACCEPTANCE_AND_RC_FREEZE_REPORT.md"
TASK0217 = ROOT / "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def frozen_runtime_contracts() -> dict[str, Any]:
    replay = read_json(TASK0217 / "guard_recovery_replay.json")
    ordinary = read_json(TASK0217 / "ordinary_query_control.json")
    structure = replay["structure"]
    graph = replay["graph"]
    fail_closed = replay["fail_closed"]
    structure_guard = structure.get("guard_trace") or {}
    graph_guard = graph.get("guard_trace") or {}
    graph_trace = graph.get("graph_trace") or {}
    fail_guard = fail_closed.get("guard_trace") or {}
    checks = {
        "ordinary_search_passed": ordinary.get("execution_status") == "passed",
        "structure_recovery_passed": (
            structure.get("execution_status") == "passed"
            and structure_guard.get("structure_lane_invoked") is True
            and structure_guard.get("recovery_attempt_count") == 1
        ),
        "one_hop_graph_recovery_passed": (
            graph.get("execution_status") == "passed"
            and graph_trace.get("graph_activated") is True
            and graph_trace.get("hop_depth") == 1
            and graph_guard.get("recovery_attempt_count") == 1
        ),
        "fail_closed_passed": (
            fail_closed.get("execution_status") == "passed"
            and fail_guard.get("fail_closed") is True
            and fail_guard.get("final_decision") == "fail_closed"
        ),
        "recovery_budget_preserved": all(
            (item.get("guard_trace") or {}).get("maximum_recovery_attempt_count") == 1
            for item in (structure, graph, fail_closed)
        ),
    }
    return {
        "schema_version": "opk-rag.task0283.frozen-runtime-contracts.v1",
        "authority_source": str((TASK0217 / "guard_recovery_replay.json").relative_to(ROOT)),
        "checks": checks,
        "valid": all(checks.values()),
    }


def run() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    snapshot = read_json(RESULT / "public_snapshot_manifest.json")
    clean = read_json(RESULT / "clean_snapshot_validation.json")
    runtime = read_json(RESULT / "runtime_smoke.json")
    contracts = frozen_runtime_contracts()
    write_json("frozen_runtime_contracts.json", contracts)

    source_gates = {
        "readme_quick_start": task0282._readme_links()["valid"],
        "open_source_files": task0282._open_source_files()["valid"],
        "public_claims": task0282._public_claims()["valid"],
        "ci_present": (ROOT / ".github/workflows/ci.yml").is_file(),
        "task_card_present": TASK.is_file(),
        "report_present": REPORT.is_file(),
    }
    snapshot_checks = snapshot.get("checks") or {}
    privacy = {
        "identity_finding_count": len(snapshot.get("identity_findings") or []),
        "secret_finding_count": len(snapshot.get("secret_findings") or []),
        "oversized_file_count": len(snapshot.get("oversized_files") or []),
        "forbidden_binary_count": len(snapshot.get("forbidden_binary_files") or []),
        "private_kb_absent": snapshot_checks.get("private_kb_absent") is True,
    }
    gates = {
        **source_gates,
        "snapshot_valid": snapshot.get("release_snapshot_valid") is True,
        "snapshot_matches_final_head": snapshot.get("development_git_head") == head,
        "snapshot_content_bound_to_runtime_smoke": (
            snapshot.get("snapshot_content_sha256")
            == (runtime.get("snapshot") or {}).get("content_sha256")
        ),
        "privacy_audit": (
            privacy["identity_finding_count"] == 0
            and privacy["secret_finding_count"] == 0
            and privacy["oversized_file_count"] == 0
            and privacy["forbidden_binary_count"] == 0
            and privacy["private_kb_absent"]
        ),
        "clean_snapshot_validation": clean.get("passed") is True,
        "isolated_public_runtime_smoke": runtime.get("passed") is True,
        "temporary_runtime_cleaned": runtime.get("cleanup", {}).get("passed") is True,
        "frozen_runtime_contracts": contracts["valid"],
    }
    blockers = sorted(name for name, passed in gates.items() if not passed)
    summary = {
        "schema_version": "opk-rag.task0283.summary.v1",
        "task_id": "TASK-0283",
        "task_status": "complete" if not blockers else "blocked",
        "candidate_decision": (
            "opk_rag_release_candidate_final_acceptance_passed"
            if not blockers
            else "hold_opk_rag_release_candidate"
        ),
        "development_git_head": head,
        "public_snapshot_file_count": snapshot.get("file_count"),
        "public_snapshot_total_bytes": snapshot.get("total_bytes"),
        "public_snapshot_content_sha256": snapshot.get("snapshot_content_sha256"),
        "source_worktree_clean_at_export": snapshot.get("development_worktree_clean"),
        "source_worktree_change_count_at_export": snapshot.get("development_worktree_change_count"),
        "release_commit_pending": (runtime.get("snapshot") or {}).get("release_commit_pending") is True,
        "known_release_blocker_count": len(blockers),
        "known_release_blockers": blockers,
        "privacy": privacy,
        "gates": gates,
        "github_release_created": False,
        "git_tag_created": False,
        "git_push_performed": False,
        "runtime_trace_authority_changed": False,
        "rag_backend_architecture_changed": False,
        "production_agent_authority_changed": False,
        "new_llm_invocation_added": False,
        "next_recommended_stage": (
            "github_release_candidate_and_portfolio_delivery"
            if not blockers
            else "task0283_release_acceptance_remediation"
        ),
    }
    write_json("summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    required = [
        "public_snapshot_manifest.json",
        "clean_snapshot_validation.json",
        "runtime_smoke.json",
        "frozen_runtime_contracts.json",
        "summary.json",
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    summary = read_json(RESULT / "summary.json") if not missing else {}
    checks = {
        "required_artifacts": not missing,
        "task_complete": summary.get("task_status") == "complete",
        "final_acceptance_passed": summary.get("candidate_decision")
        == "opk_rag_release_candidate_final_acceptance_passed",
        "zero_blockers": summary.get("known_release_blocker_count") == 0,
        "no_external_action": (
            summary.get("github_release_created") is False
            and summary.get("git_tag_created") is False
            and summary.get("git_push_performed") is False
        ),
        "authority_unchanged": (
            summary.get("runtime_trace_authority_changed") is False
            and summary.get("rag_backend_architecture_changed") is False
            and summary.get("production_agent_authority_changed") is False
            and summary.get("new_llm_invocation_added") is False
        ),
    }
    verification = {
        "schema_version": "opk-rag.task0283.verification.v1",
        "task_id": "TASK-0283",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
    write_json("verification.json", verification)
    return verification
