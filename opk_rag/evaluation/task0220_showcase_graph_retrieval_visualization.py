from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from opk_rag.showcase.demo import HISTORICAL_GRAPH_QUERY, load_showcase_authority
from opk_rag.showcase.graph_visualization import (
    AUTHORITATIVE_SCENARIO_ID,
    GraphVisualizationError,
    normalize_s03_trace,
    render_documentation,
    render_drawio_handoff,
    render_executive_mermaid,
    render_mermaid,
    validate_s03_trace,
    verify_visualization_digest,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0220"
SCHEMA_VERSION = "opk-rag.task0220.showcase-graph-retrieval-visualization.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0220-showcase-graph-retrieval-visualization"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0220_showcase_graph_retrieval_visualization_contract.json"
SHOWCASE_ARTIFACT_PATH = ROOT / "evaluation-data/showcase/graph_retrieval_visualization_v1.json"
MERMAID_PATH = ROOT / "docs/diagrams/showcase_graph_retrieval_s03.mmd"
EXECUTIVE_MERMAID_PATH = ROOT / "docs/diagrams/showcase_graph_retrieval_s03_executive.mmd"
DOC_PATH = ROOT / "docs/SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION.md"
DRAWIO_HANDOFF_PATH = ROOT / "docs/diagrams/showcase_graph_retrieval_s03_drawio_prompt.md"
REPORT_PATH = ROOT / "docs/TASK0220_SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION_REPORT.md"
TASK0219_SUMMARY = ROOT / "evaluation-data/results/task0219-unified-showcase-demo-entry-point/summary.json"
TASK0219_S03 = ROOT / "evaluation-data/results/task0219-unified-showcase-demo-entry-point/scenario_s03.json"
SOURCE_TRACE_RELATIVE = "evaluation-data/results/task0220-showcase-graph-retrieval-visualization/source_s03_runtime_trace.json"


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


def core_runtime_source_changed(paths: list[str] | None = None) -> bool:
    rows = paths if paths is not None else changed_paths()
    frozen_prefixes = (
        "opk_rag/runtime_v2/",
        "opk_rag/search/",
        "opk_rag/reranking/",
        "opk_rag/embedding/",
        "opk_rag/answer/",
        "opk_rag/vector_backends/",
        "opk_rag/evidence/",
    )
    return any(row.startswith(frozen_prefixes) for row in rows)


def graph_runtime_source_changed(paths: list[str] | None = None) -> bool:
    rows = paths if paths is not None else changed_paths()
    graph_runtime_files = (
        "opk_rag/runtime_v2/graph_activation.py",
        "opk_rag/runtime_v2/graph_retrieval.py",
        "opk_rag/runtime_v2/public_search_runtime.py",
    )
    return any(row in graph_runtime_files for row in rows)


def extract_s03(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = payload.get("scenarios") or []
    selected = [row for row in scenarios if isinstance(row, dict) and row.get("scenario_id") == AUTHORITATIVE_SCENARIO_ID]
    if len(selected) != 1:
        raise GraphVisualizationError(f"expected_one_s03_scenario:found={len(selected)}")
    return dict(selected[0])


def run_live_s03(timeout_seconds: int = 180) -> dict[str, Any]:
    completed = subprocess.run(
        ["uv", "run", "opk-rag", "demo", "--scenario", "S03", "--format", "json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise GraphVisualizationError(f"live_s03_cli_failed:exit={completed.returncode}:stderr={completed.stderr[-500:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GraphVisualizationError("live_s03_cli_non_json_output") from exc
    if payload.get("execution_status") != "passed":
        raise GraphVisualizationError(f"live_s03_showcase_not_passed:{payload.get('execution_status')}:{payload.get('blocker')}")
    return payload


def authority_audit() -> dict[str, Any]:
    authority = load_showcase_authority()
    task0219 = read_json(TASK0219_SUMMARY)
    historical_promoted = any(
        item.get("scenario_id") == "S03" and item.get("query") == HISTORICAL_GRAPH_QUERY
        for item in authority.scenarios
    )
    s03 = [item for item in authority.scenarios if item.get("scenario_id") == "S03"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0218_showcase_authority_loaded": True,
        "task0219_demo_authority_loaded": task0219.get("task_status") == "complete" and task0219.get("scenario_s03_valid") is True,
        "showcase_query_set_digest": authority.query_set_digest,
        "authoritative_scenario_id": "S03",
        "authoritative_s03_query": s03[0].get("query") if len(s03) == 1 else None,
        "authoritative_s03_query_valid": len(s03) == 1 and s03[0].get("query") != HISTORICAL_GRAPH_QUERY,
        "historical_graph_query_promoted": historical_promoted,
        "task0219_graph_sensitive_demo_valid": task0219.get("graph_sensitive_demo_valid") is True,
        "task0219_graph_runtime_hop_depth": task0219.get("graph_runtime_hop_depth"),
    }


def integrity_payload(model: Mapping[str, Any], source_s03: Mapping[str, Any]) -> dict[str, Any]:
    graph = source_s03.get("graph") or {}
    source_edges = {
        str(edge.get("edge_id"))
        for provenance in graph.get("candidate_provenance") or []
        for edge in provenance.get("graph_path") or []
    }
    model_edges = {str(edge.get("edge_id")) for edge in model.get("authoritative_relations") or []}
    source_expanded = {
        str(row.get("canonical_chunk_id") or row.get("candidate_id") or "")
        for row in graph.get("candidate_provenance") or []
    }
    model_expanded = {str(row.get("chunk_id") or "") for row in model.get("expanded_candidates") or []}
    layers = {
        "initial_candidate_layer_present": bool(model.get("initial_candidates")),
        "seed_layer_present": bool(model.get("seed_candidates")),
        "graph_activation_layer_present": (model.get("activation") or {}).get("graph_activated") is True,
        "authoritative_relation_layer_present": bool(model.get("authoritative_relations")),
        "expanded_candidate_layer_present": bool(model.get("expanded_candidates")),
        "reranking_layer_present": (model.get("retrieval") or {}).get("reranker_active") is True and bool(model.get("reranked_candidates")),
        "final_evidence_layer_present": bool(model.get("selected_evidence")),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        **layers,
        "all_required_layers_present": all(layers.values()),
        "authoritative_relation_ids_match_source_trace": model_edges == source_edges and bool(model_edges),
        "expanded_candidate_ids_match_source_trace": model_expanded == source_expanded and bool(model_expanded),
        "fabricated_graph_relation_count": len(model_edges - source_edges),
        "fabricated_candidate_count": len(model_expanded - source_expanded),
        "runtime_gold_metadata_usage": model.get("runtime_gold_metadata_usage") is True,
        "graph_candidate_provenance_available": bool(graph.get("candidate_provenance")),
        "graph_snapshot_authority_valid": (model.get("graph_authority") or {}).get("graph_snapshot_authority_valid") is True,
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
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scanned_path_count": len(paths),
        "finding_count": len(findings),
        "findings": findings,
        "sensitive_value_scan_passed": not findings,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "authoritative_scenario_id": "S03",
        "input_authorities": [
            "evaluation-data/showcase/showcase_manifest_v1.json",
            "TASK-0219 opk-rag demo --scenario S03 --format json",
        ],
        "required_runtime_invariants": {
            "graph_activated": True,
            "graph_hop_depth": 1,
            "expanded_candidate_count_min": 1,
            "runtime_gold_metadata_usage": False,
        },
        "mutation_policy": {
            "core_runtime_source_changed": False,
            "graph_runtime_source_changed": False,
            "performance_optimization_reopened": False,
        },
        "git_commit_created": False,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK0220 Showcase Graph Retrieval Visualization Report

```text
task_id=TASK-0220
task_status={summary.get('task_status')}
authoritative_scenario_id=S03
graph_activated={str(summary.get('graph_activated')).lower()}
graph_runtime_hop_depth={summary.get('graph_runtime_hop_depth')}
authoritative_relation_count={summary.get('authoritative_relation_count')}
expanded_candidate_count={summary.get('expanded_candidate_count')}
graph_showcase_visualization_v1_digest={summary.get('graph_showcase_visualization_v1_digest')}
visualization_digest_reproducible={str(summary.get('visualization_digest_reproducible')).lower()}
core_runtime_source_changed={str(summary.get('core_runtime_source_changed')).lower()}
```

TASK-0220 converts the live TASK-0219 S03 trace into a presentation-only visualization authority. The canonical model distinguishes initial candidates from the Graph-added candidate, records only the authoritative edge actually traversed by candidate provenance, and explicitly rejoins both lanes before the existing BGE reranker and evidence-selection path.

No Graph, retrieval, reranking, embedding, Agent, evidence, or performance policy is changed by this task.
"""


def verify_task0220_artifacts(*, update_summary: bool = True) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        SHOWCASE_ARTIFACT_PATH,
        MERMAID_PATH,
        EXECUTIVE_MERMAID_PATH,
        DOC_PATH,
        DRAWIO_HANDOFF_PATH,
        REPORT_PATH,
        RESULT_DIR / "summary.json",
        RESULT_DIR / "source_s03_runtime_trace.json",
        RESULT_DIR / "normalized_graph_visualization.json",
        RESULT_DIR / "visualization_reproducibility.json",
        RESULT_DIR / "visualization_integrity.json",
        RESULT_DIR / "sensitive_value_scan.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    errors: list[str] = []
    if missing:
        errors.append("required_artifact_missing")
    summary = read_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    model = read_json(SHOWCASE_ARTIFACT_PATH) if SHOWCASE_ARTIFACT_PATH.exists() else {}
    integrity = read_json(RESULT_DIR / "visualization_integrity.json") if (RESULT_DIR / "visualization_integrity.json").exists() else {}
    reproducibility = read_json(RESULT_DIR / "visualization_reproducibility.json") if (RESULT_DIR / "visualization_reproducibility.json").exists() else {}
    sensitive = read_json(RESULT_DIR / "sensitive_value_scan.json") if (RESULT_DIR / "sensitive_value_scan.json").exists() else {}
    checks = {
        "task_complete": summary.get("task_status") == "complete",
        "digest_valid": bool(model) and verify_visualization_digest(model),
        "all_layers_present": integrity.get("all_required_layers_present") is True,
        "no_fabricated_relations": integrity.get("fabricated_graph_relation_count") == 0,
        "no_fabricated_candidates": integrity.get("fabricated_candidate_count") == 0,
        "reproducible": reproducibility.get("visualization_digest_reproducible") is True,
        "sensitive_scan_passed": sensitive.get("sensitive_value_scan_passed") is True,
        "core_runtime_unchanged": summary.get("core_runtime_source_changed") is False,
        "graph_runtime_unchanged": summary.get("graph_runtime_source_changed") is False,
        "git_commit_not_created": summary.get("git_commit_created") is False,
    }
    errors.extend(key for key, ok in checks.items() if not ok)
    verification = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "missing_artifacts": missing,
        "checks": checks,
        "verification_errors": errors,
        "verification_passed": not errors,
        "git_commit_created": False,
    }
    write_json(RESULT_DIR / "verification.json", verification)
    if update_summary and summary:
        summary["independent_verifier_passed"] = verification["verification_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
    return verification


def run_task0220(*, write: bool = True) -> dict[str, Any]:
    authority = authority_audit()
    live_payload = run_live_s03()
    live_s03 = extract_s03(live_payload)
    source_checks = validate_s03_trace(live_s03)
    s03_live_replay_passed = all(source_checks.values())
    historical_s03 = read_json(TASK0219_S03)
    authority_alignment = {
        "query_matches_task0219": live_s03.get("query") == historical_s03.get("query") == authority.get("authoritative_s03_query"),
        "query_set_digest_matches_task0219": live_s03.get("query_set_digest") == historical_s03.get("query_set_digest") == authority.get("showcase_query_set_digest"),
        "graph_revision_matches_task0219": (live_s03.get("graph") or {}).get("graph_revision") == (historical_s03.get("graph") or {}).get("graph_revision"),
        "historical_graph_query_promoted": authority.get("historical_graph_query_promoted") is True,
    }
    if not s03_live_replay_passed:
        raise GraphVisualizationError("live_s03_replay_failed:" + ",".join(k for k, v in source_checks.items() if not v))
    if not all(value for key, value in authority_alignment.items() if key != "historical_graph_query_promoted") or authority_alignment["historical_graph_query_promoted"]:
        raise GraphVisualizationError("s03_authority_alignment_failed")

    model1 = normalize_s03_trace(live_s03, source_trace=SOURCE_TRACE_RELATIVE)
    model2 = normalize_s03_trace(live_s03, source_trace=SOURCE_TRACE_RELATIVE)
    reproduction = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "visualization_generation_run_count": 2,
        "run1_digest": model1["graph_showcase_visualization_v1_digest"],
        "run2_digest": model2["graph_showcase_visualization_v1_digest"],
        "visualization_semantically_consistent": model1 == model2,
        "visualization_digest_reproducible": model1["graph_showcase_visualization_v1_digest"] == model2["graph_showcase_visualization_v1_digest"],
    }
    integrity = integrity_payload(model1, live_s03)
    mermaid = render_mermaid(model1)
    executive_mermaid = render_executive_mermaid(model1)
    documentation = render_documentation(model1)
    drawio_handoff = render_drawio_handoff(model1)

    paths_before_state = changed_paths()
    core_changed = core_runtime_source_changed(paths_before_state)
    graph_changed = graph_runtime_source_changed(paths_before_state)
    graph = live_s03.get("graph") or {}
    evidence = live_s03.get("evidence") or {}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete",
        "current_stage": "project_showcase_delivery",
        "source_repository_head": git_head(),
        "task0218_showcase_authority_loaded": authority.get("task0218_showcase_authority_loaded") is True,
        "task0219_demo_authority_loaded": authority.get("task0219_demo_authority_loaded") is True,
        "authoritative_scenario_id": "S03",
        "authoritative_s03_query_valid": authority.get("authoritative_s03_query_valid") is True and authority_alignment["query_matches_task0219"],
        "s03_live_replay_passed": s03_live_replay_passed,
        "graph_activation_evaluated": graph.get("graph_activation_evaluated") is True,
        "graph_activated": graph.get("graph_activated") is True,
        "graph_runtime_hop_depth": graph.get("hop_depth"),
        "initial_candidate_count": len(model1.get("initial_candidates") or []),
        "seed_candidate_count": len(model1.get("seed_candidates") or []),
        "authoritative_relation_count": len(model1.get("authoritative_relations") or []),
        "expanded_candidate_count": len(model1.get("expanded_candidates") or []),
        "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0),
        "graph_candidate_provenance_available": bool(graph.get("candidate_provenance")),
        "normalized_graph_visualization_created": True,
        "graph_showcase_visualization_v1_digest": model1["graph_showcase_visualization_v1_digest"],
        "initial_candidate_layer_present": integrity["initial_candidate_layer_present"],
        "seed_layer_present": integrity["seed_layer_present"],
        "graph_activation_layer_present": integrity["graph_activation_layer_present"],
        "authoritative_relation_layer_present": integrity["authoritative_relation_layer_present"],
        "expanded_candidate_layer_present": integrity["expanded_candidate_layer_present"],
        "reranking_layer_present": integrity["reranking_layer_present"],
        "final_evidence_layer_present": integrity["final_evidence_layer_present"],
        "executive_visualization_ready": bool(executive_mermaid.strip()),
        "technical_visualization_ready": bool(mermaid.strip()),
        "mermaid_visualization_ready": bool(mermaid.strip()) and bool(executive_mermaid.strip()),
        "graph_visualization_documentation_ready": bool(documentation.strip()),
        "drawio_handoff_ready": bool(drawio_handoff.strip()),
        **reproduction,
        "historical_graph_query_promoted": False,
        "runtime_gold_metadata_usage": graph.get("runtime_gold_metadata_usage") is True,
        "fabricated_graph_relation_count": integrity["fabricated_graph_relation_count"],
        "fabricated_candidate_count": integrity["fabricated_candidate_count"],
        "core_runtime_source_changed": core_changed,
        "graph_runtime_source_changed": graph_changed,
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
        "next_recommended_task": "TASK-0221",
    }

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(CONTRACT_PATH, build_contract())
        write_json(RESULT_DIR / "source_s03_runtime_trace.json", live_payload)
        write_json(RESULT_DIR / "normalized_graph_visualization.json", model1)
        write_json(RESULT_DIR / "visualization_reproducibility.json", reproduction)
        write_json(RESULT_DIR / "visualization_integrity.json", {**integrity, "authority_alignment": authority_alignment, "source_trace_checks": source_checks})
        write_json(SHOWCASE_ARTIFACT_PATH, model1)
        MERMAID_PATH.write_text(mermaid, encoding="utf-8")
        EXECUTIVE_MERMAID_PATH.write_text(executive_mermaid, encoding="utf-8")
        DOC_PATH.write_text(documentation, encoding="utf-8")
        DRAWIO_HANDOFF_PATH.write_text(drawio_handoff, encoding="utf-8")
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        scan_paths = [
            SHOWCASE_ARTIFACT_PATH,
            MERMAID_PATH,
            EXECUTIVE_MERMAID_PATH,
            DOC_PATH,
            DRAWIO_HANDOFF_PATH,
            REPORT_PATH,
            RESULT_DIR / "source_s03_runtime_trace.json",
            RESULT_DIR / "normalized_graph_visualization.json",
            RESULT_DIR / "visualization_reproducibility.json",
            RESULT_DIR / "visualization_integrity.json",
        ]
        sensitive = sensitive_scan(scan_paths)
        write_json(RESULT_DIR / "sensitive_value_scan.json", sensitive)
        summary["sensitive_value_scan_passed"] = sensitive["sensitive_value_scan_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
        verification = verify_task0220_artifacts(update_summary=True)
        summary = read_json(RESULT_DIR / "summary.json")
        summary["independent_verifier_passed"] = verification["verification_passed"]
        write_json(RESULT_DIR / "summary.json", summary)
    return summary
