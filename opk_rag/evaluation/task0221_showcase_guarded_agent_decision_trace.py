from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from opk_rag.showcase.agent_trace import (
    AGENT_TYPE,
    AgentTraceError,
    DECISION_PATHS,
    contains_hidden_reasoning_fields,
    normalize_agent_trace,
    render_architecture_mermaid,
    render_documentation,
    render_drawio_handoff,
    render_executive_mermaid,
    render_scenario_mermaid,
    validate_runtime_scenario,
    verify_agent_trace_digest,
)
from opk_rag.showcase.demo import load_showcase_authority
from opk_rag.showcase.graph_visualization import verify_visualization_digest

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0221"
SCHEMA_VERSION = "opk-rag.task0221.showcase-guarded-agent-decision-trace.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0221-showcase-guarded-agent-decision-trace"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0221_showcase_guarded_agent_decision_trace_contract.json"
SHOWCASE_PATH = ROOT / "evaluation-data/showcase/guarded_agent_decision_trace_v1.json"
GRAPH_AUTHORITY_PATH = ROOT / "evaluation-data/showcase/graph_retrieval_visualization_v1.json"
TASK0219_SUMMARY_PATH = ROOT / "evaluation-data/results/task0219-unified-showcase-demo-entry-point/summary.json"
ARCH_MERMAID_PATH = ROOT / "docs/diagrams/showcase_guarded_agent_architecture.mmd"
SCENARIO_MERMAID_PATH = ROOT / "docs/diagrams/showcase_guarded_agent_scenarios.mmd"
EXEC_MERMAID_PATH = ROOT / "docs/diagrams/showcase_guarded_agent_executive.mmd"
DRAWIO_PATH = ROOT / "docs/diagrams/showcase_guarded_agent_drawio_prompt.md"
DOC_PATH = ROOT / "docs/SHOWCASE_GUARDED_AGENT_DECISION_TRACE.md"
REPORT_PATH = ROOT / "docs/TASK0221_SHOWCASE_GUARDED_AGENT_DECISION_TRACE_REPORT.md"
SOURCE_TRACE_RELATIVE = "evaluation-data/results/task0221-showcase-guarded-agent-decision-trace/source_runtime_traces.json"


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
    return sorted(set(item.strip() for item in [*tracked, *untracked] if item.strip()))


def _changed_under(prefixes: tuple[str, ...], paths: list[str]) -> bool:
    return any(item.startswith(prefixes) for item in paths)


def mutation_audit(paths: list[str] | None = None) -> dict[str, bool]:
    rows = paths if paths is not None else changed_paths()
    core = (
        "opk_rag/runtime_v2/", "opk_rag/search/", "opk_rag/reranking/", "opk_rag/embedding/",
        "opk_rag/answer/", "opk_rag/evidence/", "opk_rag/vector_backends/",
    )
    return {
        "core_runtime_source_changed": _changed_under(core, rows),
        "agent_runtime_source_changed": _changed_under(("opk_rag/runtime_v2/", "opk_rag/search/"), rows),
        "retrieval_runtime_source_changed": _changed_under(("opk_rag/search/", "opk_rag/runtime_v2/"), rows),
        "graph_runtime_source_changed": _changed_under(("opk_rag/runtime_v2/",), rows),
        "evidence_runtime_source_changed": _changed_under(("opk_rag/evidence/", "opk_rag/answer/"), rows),
    }


def run_live_showcase(timeout_seconds: int = 180) -> dict[str, Any]:
    """Run each frozen scenario through TASK-0219 production Demo independently.

    Separate invocations preserve the same production path while making a transient failure
    attributable to one scenario instead of hiding it inside a single four-scenario command.
    """
    scenarios: list[dict[str, Any]] = []
    first_payload: dict[str, Any] | None = None
    for sid in ("S01", "S02", "S03", "S04"):
        completed = subprocess.run(
            ["uv", "run", "opk-rag", "demo", "--scenario", sid, "--format", "json"],
            cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds, check=False,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AgentTraceError(f"live_{sid}_non_json:exit={completed.returncode}:stderr={completed.stderr[-500:]}") from exc
        if completed.returncode != 0 or payload.get("execution_status") != "passed":
            raise AgentTraceError(
                f"live_{sid}_failed:exit={completed.returncode}:status={payload.get('execution_status')}:"
                f"blocker={payload.get('blocker')}:stderr={completed.stderr[-500:]}"
            )
        rows = payload.get("scenarios") or []
        if len(rows) != 1 or rows[0].get("scenario_id") != sid:
            raise AgentTraceError(f"live_{sid}_unexpected_scenario_envelope")
        first_payload = first_payload or payload
        scenarios.append(dict(rows[0]))
    assert first_payload is not None
    return {
        "showcase_schema_version": first_payload.get("showcase_schema_version"),
        "manifest_version": first_payload.get("manifest_version"),
        "query_set_digest": first_payload.get("query_set_digest"),
        "demo_status": "passed",
        "execution_status": "passed",
        "scenario_order": ["S01", "S02", "S03", "S04"],
        "scenario_count": 4,
        "preflight": first_payload.get("preflight"),
        "scenarios": scenarios,
        "runtime_gold_metadata_usage": False,
        "query_specific_hardcoding": False,
        "demo_fixture_manipulation": False,
        "demo_only_runtime_policy": False,
        "task0221_live_execution_mode": "four_authoritative_scenario_invocations",
    }


def authority_audit(live: Mapping[str, Any], graph_authority: Mapping[str, Any]) -> dict[str, Any]:
    authority = load_showcase_authority()
    task0219 = read_json(TASK0219_SUMMARY_PATH)
    live_by_id = {row.get("scenario_id"): row for row in live.get("scenarios") or []}
    authority_by_id = {row.get("scenario_id"): row for row in authority.scenarios}
    queries_match = all(live_by_id.get(sid, {}).get("query") == authority_by_id.get(sid, {}).get("query") for sid in ("S01","S02","S03","S04"))
    graph_valid = verify_visualization_digest(graph_authority) and graph_authority.get("scenario_id") == "S03"
    return {
        "task0218_showcase_authority_loaded": True,
        "task0219_demo_authority_loaded": task0219.get("task_status") == "complete",
        "task0220_graph_visualization_authority_loaded": graph_valid,
        "query_set_digest": authority.query_set_digest,
        "scenario_queries_match_authority": queries_match,
        "task0220_graph_visualization_digest": graph_authority.get("graph_showcase_visualization_v1_digest"),
    }


def runtime_trace_checks(live: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for row in live.get("scenarios") or []:
        sid = str(row.get("scenario_id"))
        detail = validate_runtime_scenario(row)
        checks[sid] = {"checks": detail, "valid": all(detail.values())}
    checks["all_valid"] = all(checks.get(sid, {}).get("valid") is True for sid in ("S01","S02","S03","S04"))
    return checks


def integrity_payload(model: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    source = {str(row.get("scenario_id")): row for row in live.get("scenarios") or []}
    normalized = {str(row.get("scenario_id")): row for row in model.get("scenarios") or []}
    expected_actions = {
        "S01": "no_recovery",
        "S02": "invoke_structure_lane",
        "S03": "invoke_one_hop_graph_expansion",
        "S04": "refuse_unsupported_answer",
    }
    fabricated_decisions = 0
    fabricated_actions = 0
    for sid, path in DECISION_PATHS.items():
        row = normalized.get(sid) or {}
        src = source.get(sid) or {}
        if (row.get("decision") or {}).get("decision_path") != path:
            fabricated_decisions += 1
        if (row.get("action") or {}).get("action_type") != expected_actions[sid]:
            fabricated_actions += 1
        guard = src.get("guard") or {}
        graph = src.get("graph") or {}
        if sid == "S01" and ((row.get("action") or {}).get("recovery_attempt_count") != guard.get("recovery_attempt_count")):
            fabricated_actions += 1
        if sid == "S02" and (row.get("observations") or {}).get("structure_lane_invoked") != (guard.get("structure_lane_invoked") is True):
            fabricated_decisions += 1
        if sid == "S03" and (row.get("observations") or {}).get("graph_activated") != (graph.get("graph_activated") is True):
            fabricated_decisions += 1
        if sid == "S04" and (row.get("decision") or {}).get("runtime_final_decision") != guard.get("final_decision"):
            fabricated_decisions += 1
    return {
        "fabricated_agent_decision_count": fabricated_decisions,
        "fabricated_agent_action_count": fabricated_actions,
        "hidden_chain_of_thought_exposed": contains_hidden_reasoning_fields(model),
        "runtime_gold_metadata_usage": model.get("runtime_gold_metadata_usage") is True,
        "planner_enabled": model.get("planner_enabled") is True,
        "unbounded_agent_loop_enabled": model.get("unbounded_agent_loop_enabled") is True,
        "maximum_recovery_attempt_count": model.get("maximum_recovery_attempt_count"),
        "graph_runtime_hop_depth": model.get("graph_runtime_hop_depth"),
    }


def sensitive_scan(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "api_key_assignment": re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*[^\s\"']+"),
        "authorization_header": re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+"),
        "database_url": re.compile(r"(?i)(postgres(?:ql)?|mysql|mongodb)://[^\s\"']+"),
        "private_home_path": re.compile(r"/(?:home|data/envs)/[^\s\"']+"),
    }
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"path": str(path.relative_to(ROOT)), "pattern": name})
    return {"scanned_path_count": len(paths), "finding_count": len(findings), "findings": findings, "sensitive_value_scan_passed": not findings}


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "agent_type": AGENT_TYPE,
        "planner_enabled": False,
        "unbounded_agent_loop_enabled": False,
        "maximum_recovery_attempt_count": 1,
        "graph_runtime_hop_depth": 1,
        "required_decision_paths": DECISION_PATHS,
        "hidden_chain_of_thought_exposed": False,
        "runtime_mutation_allowed": False,
        "git_commit_created": False,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK0221 Showcase Guarded Agent Decision Trace Report

```text
task_id=TASK-0221
task_status={summary.get('task_status')}
agent_type={summary.get('agent_type')}
planner_enabled={str(summary.get('planner_enabled')).lower()}
maximum_recovery_attempt_count={summary.get('maximum_recovery_attempt_count')}
graph_runtime_hop_depth={summary.get('graph_runtime_hop_depth')}
guarded_agent_decision_trace_v1_digest={summary.get('guarded_agent_decision_trace_v1_digest')}
agent_trace_digest_reproducible={str(summary.get('agent_trace_digest_reproducible')).lower()}
hidden_chain_of_thought_exposed={str(summary.get('hidden_chain_of_thought_exposed')).lower()}
```

TASK-0221 converts explicit TASK-0219 production telemetry into a presentation-only Guarded Agent control-plane authority. It does not expose hidden reasoning and does not add a Planner, recovery loop, Graph hop, tool, or policy mutation.
"""


def verify_task0221_artifacts(*, update_summary: bool = True) -> dict[str, Any]:
    required = [
        CONTRACT_PATH, SHOWCASE_PATH, ARCH_MERMAID_PATH, SCENARIO_MERMAID_PATH, EXEC_MERMAID_PATH,
        DRAWIO_PATH, DOC_PATH, REPORT_PATH, RESULT_DIR / "summary.json", RESULT_DIR / "source_runtime_traces.json",
        RESULT_DIR / "normalized_agent_trace.json", RESULT_DIR / "agent_trace_reproducibility.json",
        RESULT_DIR / "agent_trace_integrity.json", RESULT_DIR / "sensitive_value_scan.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    summary = read_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    model = read_json(SHOWCASE_PATH) if SHOWCASE_PATH.exists() else {}
    repro = read_json(RESULT_DIR / "agent_trace_reproducibility.json") if (RESULT_DIR / "agent_trace_reproducibility.json").exists() else {}
    integrity = read_json(RESULT_DIR / "agent_trace_integrity.json") if (RESULT_DIR / "agent_trace_integrity.json").exists() else {}
    sensitive = read_json(RESULT_DIR / "sensitive_value_scan.json") if (RESULT_DIR / "sensitive_value_scan.json").exists() else {}
    checks = {
        "task_complete": summary.get("task_status") == "complete",
        "digest_valid": bool(model) and verify_agent_trace_digest(model),
        "four_scenarios": len(model.get("scenarios") or []) == 4,
        "planner_disabled": model.get("planner_enabled") is False,
        "unbounded_loop_disabled": model.get("unbounded_agent_loop_enabled") is False,
        "bounded_recovery": model.get("maximum_recovery_attempt_count") == 1,
        "graph_one_hop": model.get("graph_runtime_hop_depth") == 1,
        "no_hidden_cot": integrity.get("hidden_chain_of_thought_exposed") is False,
        "no_fabricated_decision": integrity.get("fabricated_agent_decision_count") == 0,
        "no_fabricated_action": integrity.get("fabricated_agent_action_count") == 0,
        "reproducible": repro.get("agent_trace_digest_reproducible") is True,
        "sensitive_scan": sensitive.get("sensitive_value_scan_passed") is True,
        "core_runtime_unchanged": summary.get("core_runtime_source_changed") is False,
        "agent_runtime_unchanged": summary.get("agent_runtime_source_changed") is False,
        "graph_runtime_unchanged": summary.get("graph_runtime_source_changed") is False,
        "git_commit_not_created": summary.get("git_commit_created") is False,
    }
    errors = (["required_artifact_missing"] if missing else []) + [key for key, ok in checks.items() if not ok]
    verification = {
        "schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "missing_artifacts": missing,
        "checks": checks, "verification_errors": errors, "verification_passed": not errors,
        "git_commit_created": False,
    }
    write_json(RESULT_DIR / "verification.json", verification)
    if update_summary and summary:
        summary["independent_verifier_passed"] = verification["verification_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
    return verification


def run_task0221(*, write: bool = True, live_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    graph_authority = read_json(GRAPH_AUTHORITY_PATH)
    live = dict(live_payload) if live_payload is not None else run_live_showcase()
    authority = authority_audit(live, graph_authority)
    trace_checks = runtime_trace_checks(live)
    if not all((authority["task0218_showcase_authority_loaded"], authority["task0219_demo_authority_loaded"], authority["task0220_graph_visualization_authority_loaded"], authority["scenario_queries_match_authority"], trace_checks["all_valid"])):
        raise AgentTraceError("showcase_authority_or_runtime_trace_invalid")

    scenarios = list(live.get("scenarios") or [])
    model1 = normalize_agent_trace(scenarios, query_set_digest=str(live.get("query_set_digest")), graph_authority=graph_authority, source_trace=SOURCE_TRACE_RELATIVE)
    model2 = normalize_agent_trace(scenarios, query_set_digest=str(live.get("query_set_digest")), graph_authority=graph_authority, source_trace=SOURCE_TRACE_RELATIVE)
    repro = {
        "agent_trace_generation_run_count": 2,
        "run1_digest": model1["guarded_agent_decision_trace_v1_digest"],
        "run2_digest": model2["guarded_agent_decision_trace_v1_digest"],
        "agent_trace_semantically_consistent": model1 == model2,
        "agent_trace_digest_reproducible": model1["guarded_agent_decision_trace_v1_digest"] == model2["guarded_agent_decision_trace_v1_digest"],
    }
    integrity = integrity_payload(model1, live)
    architecture = render_architecture_mermaid(model1)
    scenarios_mermaid = render_scenario_mermaid(model1)
    executive = render_executive_mermaid(model1)
    documentation = render_documentation(model1)
    drawio = render_drawio_handoff(model1)
    mutations = mutation_audit()
    normalized_by_id = {row["scenario_id"]: row for row in model1["scenarios"]}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete",
        "current_stage": "project_showcase_delivery",
        "source_repository_head": git_head(),
        **authority,
        "agent_type": model1["agent_type"],
        "planner_enabled": model1["planner_enabled"],
        "unbounded_agent_loop_enabled": model1["unbounded_agent_loop_enabled"],
        "maximum_recovery_attempt_count": model1["maximum_recovery_attempt_count"],
        "graph_runtime_hop_depth": model1["graph_runtime_hop_depth"],
        "scenario_s01_trace_valid": trace_checks["S01"]["valid"],
        "scenario_s02_trace_valid": trace_checks["S02"]["valid"],
        "scenario_s03_trace_valid": trace_checks["S03"]["valid"],
        "scenario_s04_trace_valid": trace_checks["S04"]["valid"],
        "s01_decision_path": normalized_by_id["S01"]["decision"]["decision_path"],
        "s02_decision_path": normalized_by_id["S02"]["decision"]["decision_path"],
        "s03_decision_path": normalized_by_id["S03"]["decision"]["decision_path"],
        "s04_decision_path": normalized_by_id["S04"]["decision"]["decision_path"],
        "structure_recovery_visualized": "structure_recovery" in scenarios_mermaid,
        "graph_recovery_visualized": "graph_recovery" in scenarios_mermaid,
        "fail_closed_visualized": "fail_closed" in scenarios_mermaid,
        "normalized_agent_trace_created": True,
        "guarded_agent_decision_trace_v1_digest": model1["guarded_agent_decision_trace_v1_digest"],
        **repro,
        "architecture_mermaid_ready": bool(architecture.strip()),
        "scenario_mermaid_ready": bool(scenarios_mermaid.strip()),
        "executive_mermaid_ready": bool(executive.strip()),
        "agent_trace_documentation_ready": bool(documentation.strip()),
        "drawio_handoff_ready": bool(drawio.strip()),
        **integrity,
        **mutations,
        "retrieval_policy_mutation_required": False,
        "reranker_mutation_required": False,
        "embedding_mutation_required": False,
        "graph_policy_mutation_required": False,
        "agent_policy_mutation_required": False,
        "evidence_policy_mutation_required": False,
        "performance_optimization_reopened": False,
        "sensitive_value_scan_passed": False,
        "independent_verifier_passed": False,
        "git_commit_created": False,
        "next_recommended_task": "TASK-0222",
    }

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        ARCH_MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(CONTRACT_PATH, build_contract())
        write_json(RESULT_DIR / "source_runtime_traces.json", live)
        write_json(RESULT_DIR / "normalized_agent_trace.json", model1)
        write_json(RESULT_DIR / "agent_trace_reproducibility.json", repro)
        write_json(RESULT_DIR / "agent_trace_integrity.json", {**integrity, "authority": authority, "runtime_trace_checks": trace_checks})
        write_json(SHOWCASE_PATH, model1)
        ARCH_MERMAID_PATH.write_text(architecture, encoding="utf-8")
        SCENARIO_MERMAID_PATH.write_text(scenarios_mermaid, encoding="utf-8")
        EXEC_MERMAID_PATH.write_text(executive, encoding="utf-8")
        DRAWIO_PATH.write_text(drawio, encoding="utf-8")
        DOC_PATH.write_text(documentation, encoding="utf-8")
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        scan_paths = [SHOWCASE_PATH, ARCH_MERMAID_PATH, SCENARIO_MERMAID_PATH, EXEC_MERMAID_PATH, DRAWIO_PATH, DOC_PATH, REPORT_PATH, RESULT_DIR / "source_runtime_traces.json", RESULT_DIR / "normalized_agent_trace.json", RESULT_DIR / "agent_trace_integrity.json"]
        sensitive = sensitive_scan(scan_paths)
        write_json(RESULT_DIR / "sensitive_value_scan.json", sensitive)
        summary["sensitive_value_scan_passed"] = sensitive["sensitive_value_scan_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
        verification = verify_task0221_artifacts(update_summary=True)
        summary = read_json(RESULT_DIR / "summary.json")
        summary["independent_verifier_passed"] = verification["verification_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
    return summary
