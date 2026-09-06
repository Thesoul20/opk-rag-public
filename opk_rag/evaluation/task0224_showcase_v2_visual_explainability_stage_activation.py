from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0224"
SCHEMA_VERSION = "opk-rag.task0224.showcase-v2-visual-explainability-stage-activation.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0224-showcase-v2-visual-explainability-stage-activation"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0224_showcase_v2_visual_explainability_stage_activation_contract.json"
TASK_PATH = ROOT / "tasks/TASK-0224_showcase_v2_visual_explainability_stage_activation.md"
ARCHITECTURE_PATH = ROOT / "docs/SHOWCASE_V2_VISUAL_EXPLAINABILITY_ARCHITECTURE.md"
PROJECT_STATE_PATH = ROOT / "PROJECT_STATE.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
CURRENT_AUTHORITY_PATH = ROOT / "docs/CURRENT_PROJECT_AUTHORITY.md"
REPORT_PATH = ROOT / "docs/TASK0224_SHOWCASE_V2_VISUAL_EXPLAINABILITY_STAGE_ACTIVATION_REPORT.md"

SHOWCASE_V1_AUTHORITY_PATHS = (
    ROOT / "evaluation-data/showcase/showcase_manifest_v1.json",
    ROOT / "evaluation-data/showcase/graph_retrieval_visualization_v1.json",
    ROOT / "evaluation-data/showcase/guarded_agent_decision_trace_v1.json",
    ROOT / "evaluation-data/showcase/end_to_end_rag_storyboard_v1.json",
    ROOT / "evaluation-data/showcase/showcase_recording_pack_v1.json",
    ROOT / "docs/TASK0223_SHOWCASE_RECORDING_READINESS_AND_DEMO_ASSET_FREEZE_REPORT.md",
)

EXPECTED_TASK0224_PREFIXES = (
    "tasks/TASK-0224_",
    "docs/SHOWCASE_V2_",
    "docs/TASK0224_",
    "opk_rag/evaluation/task0224_",
    "scripts/run_task0224_",
    "scripts/verify_task0224_",
    "tests/test_task0224_",
    "evaluation-data/contracts/task0224_",
    "evaluation-data/results/task0224-",
    "PROJECT_STATE.md",
    "CHANGELOG.md",
    "docs/CURRENT_PROJECT_AUTHORITY.md",
)

RUNTIME_PREFIX_GROUPS = {
    "core_runtime_source_changed": (
        "opk_rag/runtime_v2/",
        "opk_rag/search/",
        "opk_rag/reranking/",
        "opk_rag/embedding/",
        "opk_rag/answer/",
        "opk_rag/answerability/",
        "opk_rag/evidence/",
        "opk_rag/vector_backends/",
    ),
    "retrieval_runtime_source_changed": ("opk_rag/search/", "opk_rag/runtime_v2/retrieval"),
    "agent_runtime_source_changed": ("opk_rag/runtime_v2/",),
    "graph_runtime_source_changed": ("opk_rag/runtime_v2/graph",),
    "reranker_runtime_source_changed": ("opk_rag/reranking/",),
    "evidence_runtime_source_changed": ("opk_rag/evidence/",),
    "answer_runtime_source_changed": ("opk_rag/answer/", "opk_rag/answerability/"),
    "vector_backend_source_changed": ("opk_rag/vector_backends/",),
}

REQUIRED_ARCHITECTURE_MARKERS = (
    "OPK-RAG Core = runtime authority",
    "Showcase UI = visualization authority only",
    "## 7. Visualization Domains",
    "### 7.1 Retrieval Trace",
    "### 7.2 Guard Decision",
    "### 7.3 Knowledge / Document Graph",
    "### 7.4 Candidate → Rerank → Evidence",
    "### 7.5 Runtime / Production Status",
    "## 8. Future Runtime Trace Contract Boundary",
    "## 9. Real Runtime Data Policy",
    "## 10. Demo Scenario Architecture",
    "## 11. V1 Single-page UI Scope",
    "TASK-0225  RAG Runtime Trace Contract",
)

REQUIRED_STATE_MARKERS = (
    "Previous Stage: project_showcase_delivery",
    "Current Stage: showcase_v2_visual_explainability",
    "Showcase V1: COMPLETED",
    "Showcase V2 Visual Explainability Stage: ACTIVE",
    "Showcase UI Runtime Authority: OPK-RAG Core",
    "Showcase UI RAG Reimplementation Allowed: false",
    "Showcase UI V1: SINGLE PAGE",
    "Runtime Trace Contract Required: true",
    "Curated Demo Scenarios Required: true",
)

REQUIRED_CURRENT_AUTHORITY_MARKERS = (
    "current_stage: `showcase_v2_visual_explainability`",
    "showcase_v1_completed: `true`",
    "showcase_v2_active: `true`",
    "showcase_ui_runtime_authority: `opk_rag_core`",
    "showcase_ui_rag_reimplementation_allowed: `false`",
    "runtime_trace_contract_required: `true`",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def changed_paths() -> list[str]:
    tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    return sorted({row.strip() for row in [*tracked, *untracked] if row.strip()})


def unexpected_worktree_paths(paths: list[str] | None = None) -> list[str]:
    rows = paths if paths is not None else changed_paths()
    return [row for row in rows if not any(row == prefix or row.startswith(prefix) for prefix in EXPECTED_TASK0224_PREFIXES)]


def mutation_audit(paths: list[str] | None = None) -> dict[str, bool]:
    rows = paths if paths is not None else changed_paths()
    return {
        name: any(any(row.startswith(prefix) for prefix in prefixes) for row in rows)
        for name, prefixes in RUNTIME_PREFIX_GROUPS.items()
    }


def _all_markers_present(path: Path, markers: tuple[str, ...]) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, list(markers)
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    return not missing, missing


def architecture_audit() -> dict[str, Any]:
    valid, missing = _all_markers_present(ARCHITECTURE_PATH, REQUIRED_ARCHITECTURE_MARKERS)
    text = ARCHITECTURE_PATH.read_text(encoding="utf-8") if ARCHITECTURE_PATH.is_file() else ""
    checks = {
        "architecture_document_exists": ARCHITECTURE_PATH.is_file(),
        "architecture_markers_complete": valid,
        "visualization_domains_frozen": all(marker in text for marker in (
            "Retrieval Trace", "Guard Decision", "Knowledge / Document Graph", "Candidate → Rerank → Evidence", "Runtime / Production Status"
        )),
        "runtime_authority_boundary_frozen": "OPK-RAG Core = runtime authority" in text and "Showcase UI = visualization authority only" in text,
        "single_page_scope_frozen": "V1 Single-page UI Scope" in text,
        "real_runtime_policy_frozen": "Real Runtime Data Policy" in text and "must be real authoritative data" in text,
        "followup_sequence_defined": "TASK-0235  Final Visual Demo Video" in text,
    }
    return {
        "checks": checks,
        "architecture_valid": all(checks.values()),
        "missing_markers": missing,
    }


def state_audit() -> dict[str, Any]:
    project_valid, project_missing = _all_markers_present(PROJECT_STATE_PATH, REQUIRED_STATE_MARKERS)
    authority_valid, authority_missing = _all_markers_present(CURRENT_AUTHORITY_PATH, REQUIRED_CURRENT_AUTHORITY_MARKERS)
    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8") if CHANGELOG_PATH.is_file() else ""
    checks = {
        "project_state_valid": project_valid,
        "current_authority_valid": authority_valid,
        "changelog_entry_present": "TASK-0224 Showcase V2 Visual Explainability Stage Activation" in changelog_text,
        "showcase_v1_history_preserved": "TASK-0223 Showcase Recording Readiness and Demo Asset Freeze" in PROJECT_STATE_PATH.read_text(encoding="utf-8"),
    }
    return {
        "checks": checks,
        "state_valid": all(checks.values()),
        "missing_project_state_markers": project_missing,
        "missing_current_authority_markers": authority_missing,
    }


def showcase_v1_authority_audit() -> dict[str, Any]:
    rows = {str(path.relative_to(ROOT)): path.is_file() for path in SHOWCASE_V1_AUTHORITY_PATHS}
    return {
        "assets": rows,
        "showcase_v1_authority_preserved": all(rows.values()),
        "missing_authority_count": sum(1 for exists in rows.values() if not exists),
    }


def production_invariant_audit() -> dict[str, Any]:
    architecture = ARCHITECTURE_PATH.read_text(encoding="utf-8") if ARCHITECTURE_PATH.is_file() else ""
    project = PROJECT_STATE_PATH.read_text(encoding="utf-8") if PROJECT_STATE_PATH.is_file() else ""
    combined = architecture + "\n" + project
    checks = {
        "production_vector_backend_qdrant": "production_vector_backend=qdrant" in combined or "Production Vector Backend: Qdrant" in project,
        "default_initial_retrieval_policy_guarded_structure_aware": "default_initial_retrieval_policy=guarded_structure_aware" in combined,
        "graph_runtime_hop_depth_one": "graph_runtime_hop_depth=1" in combined,
        "search_reranker_precision_fp16_autocast": "production_search_reranker_precision=fp16_autocast" in combined,
        "ask_reranker_precision_fp32": "production_ask_reranker_precision=fp32" in combined,
        "planner_disabled": "planner_enabled=false" in combined,
        "unbounded_agent_loop_disabled": "unbounded_agent_loop_enabled=false" in combined,
        "maximum_recovery_attempt_count_one": "maximum_recovery_attempt_count=1" in combined,
    }
    return {"checks": checks, "production_invariants_valid": all(checks.values())}


def render_report(summary: Mapping[str, Any]) -> str:
    changed = "\n".join(f"- `{path}`" for path in summary.get("changed_paths", [])) or "- None"
    return f"""# TASK-0224 Showcase V2 Visual Explainability Stage Activation Report

## Result

```text
task_id={summary.get('task_id')}
task_status={summary.get('task_status')}
previous_stage={summary.get('previous_stage')}
current_stage={summary.get('current_stage')}
showcase_v1_completed={str(summary.get('showcase_v1_completed')).lower()}
showcase_v2_active={str(summary.get('showcase_v2_active')).lower()}
showcase_v2_architecture_defined={str(summary.get('showcase_v2_architecture_defined')).lower()}
showcase_v2_scope_frozen={str(summary.get('showcase_v2_scope_frozen')).lower()}
showcase_ui_runtime_authority={summary.get('showcase_ui_runtime_authority')}
showcase_ui_rag_reimplementation_allowed={str(summary.get('showcase_ui_rag_reimplementation_allowed')).lower()}
runtime_trace_contract_required={str(summary.get('runtime_trace_contract_required')).lower()}
production_runtime_behavior_changed={str(summary.get('production_runtime_behavior_changed')).lower()}
next_recommended_task={summary.get('next_recommended_task')}
```

## Architecture Decision

Showcase V2 is activated as an additive visual explainability and observability layer. OPK-RAG Core remains the only runtime authority. The future UI may render authoritative telemetry but may not implement independent Retrieval, Guard, Graph, Reranking, Evidence, or Answer logic.

The first UI implementation is frozen as a single-page showcase with five primary visualization domains: Retrieval Trace, Guard Decision, query-scoped Knowledge/Document Graph, Candidate→Rerank→Evidence, and Runtime/Production Status.

## Trace and Demo Scope

TASK-0224 freezes the classes of future trace data but deliberately does not freeze the final JSON/event wire schema. TASK-0225 is responsible for that contract. Curated scenarios must remain backed by real reproducible queries and should reuse Showcase V1 S01-S04 authority where appropriate.

## Production Preservation

- Production vector backend remains Qdrant.
- Initial retrieval remains `guarded_structure_aware`.
- Graph runtime hop depth remains `1`.
- Search reranker precision remains `fp16_autocast`; Ask remains `fp32`.
- Guard remains bounded with no unrestricted Planner or unbounded loop.
- No production runtime source files were modified by TASK-0224.
- No new runtime dependency was added.

## Changed Paths

{changed}

## Validation

Focused tests and independent verifier are the formal TASK-0224 gates. Git HEAD is required to remain the source authoritative HEAD recorded in the contract; no Git commit is created by this task.
"""


def build_summary() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    architecture = architecture_audit()
    state = state_audit()
    showcase_v1 = showcase_v1_authority_audit()
    invariants = production_invariant_audit()
    paths = changed_paths()
    mutations = mutation_audit(paths)
    unexpected = unexpected_worktree_paths(paths)
    head = git_head()
    source_head = str(contract["source_authoritative_head"])

    runtime_changed = any(mutations.values())
    complete = all((
        architecture["architecture_valid"],
        state["state_valid"],
        showcase_v1["showcase_v1_authority_preserved"],
        invariants["production_invariants_valid"],
        not runtime_changed,
        not unexpected,
        head == source_head,
    ))

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "source_authoritative_head": source_head,
        "current_git_head": head,
        "git_commit_created": head != source_head,
        "previous_stage": "project_showcase_delivery",
        "current_stage": "showcase_v2_visual_explainability",
        "showcase_v1_completed": showcase_v1["showcase_v1_authority_preserved"],
        "showcase_v2_active": state["state_valid"],
        "showcase_v2_architecture_defined": architecture["architecture_valid"],
        "showcase_v2_scope_frozen": architecture["architecture_valid"],
        "showcase_ui_primary_purpose": "visual_explainability_and_observability",
        "showcase_ui_runtime_authority": "opk_rag_core",
        "showcase_ui_rag_reimplementation_allowed": False,
        "showcase_ui_v1_single_page": True,
        "retrieval_trace_visualization_required": True,
        "guard_decision_visualization_required": True,
        "knowledge_graph_visualization_required": True,
        "candidate_rerank_evidence_visualization_required": True,
        "runtime_metrics_visualization_required": True,
        "runtime_trace_contract_required": True,
        "demo_scenario_support_required": True,
        "curated_demo_scenarios_required": True,
        "real_runtime_data_required": True,
        "fake_runtime_decisions_allowed": False,
        "fake_evidence_allowed": False,
        "fake_latency_allowed": False,
        "fake_graph_relationships_allowed": False,
        "showcase_v2_followup_sequence_defined": architecture["checks"]["followup_sequence_defined"],
        "production_invariants_valid": invariants["production_invariants_valid"],
        "production_retrieval_policy_changed": mutations["retrieval_runtime_source_changed"],
        "production_reranker_policy_changed": mutations["reranker_runtime_source_changed"],
        "production_graph_policy_changed": mutations["graph_runtime_source_changed"],
        "production_vector_backend_changed": mutations["vector_backend_source_changed"],
        "production_runtime_behavior_changed": runtime_changed,
        "new_runtime_dependency_required": False,
        "unexpected_worktree_modification_count": len(unexpected),
        "unexpected_worktree_paths": unexpected,
        "changed_path_count": len(paths),
        "changed_paths": paths,
        "mutation_audit": mutations,
        "architecture_audit": architecture,
        "state_audit": state,
        "showcase_v1_authority_audit": showcase_v1,
        "production_invariant_audit": invariants,
        "next_recommended_task": "TASK-0225",
    }


def run_task0224(*, write: bool = True) -> dict[str, Any]:
    summary = build_summary()
    if write:
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def verify_task0224_artifacts(*, update_summary: bool = True) -> dict[str, Any]:
    summary = build_summary()
    checks = {
        "task_card_exists": TASK_PATH.is_file(),
        "contract_exists": CONTRACT_PATH.is_file(),
        "architecture_exists": ARCHITECTURE_PATH.is_file(),
        "architecture_valid": summary["showcase_v2_architecture_defined"] is True,
        "state_valid": summary["showcase_v2_active"] is True,
        "showcase_v1_preserved": summary["showcase_v1_completed"] is True,
        "production_invariants_valid": summary["production_invariants_valid"] is True,
        "runtime_source_unchanged": summary["production_runtime_behavior_changed"] is False,
        "unexpected_worktree_paths_absent": summary["unexpected_worktree_modification_count"] == 0,
        "git_head_unchanged": summary["git_commit_created"] is False,
        "task_complete": summary["task_status"] == "complete",
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checks": checks,
        "verification_passed": all(checks.values()),
    }
    if update_summary:
        write_json(RESULT_DIR / "summary.json", summary)
        write_json(RESULT_DIR / "verification.json", result)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return result
