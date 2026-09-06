from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from opk_rag.showcase.end_to_end_storyboard import (
    build_claim_evidence_matrix,
    build_storyboard,
    contains_forbidden_reasoning,
    render_drawio_handoff,
    render_executive_mermaid,
    render_readme_fragment,
    render_scenario_overlay_mermaid,
    render_storyboard_document,
    render_talk_track,
    render_technical_mermaid,
    render_video_storyboard,
    storyboard_integrity,
    validate_authorities,
    verify_storyboard_digest,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0222"
SCHEMA_VERSION = "opk-rag.task0222.showcase-end-to-end-rag-architecture-storyboard.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0222-showcase-end-to-end-rag-architecture-storyboard"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0222_showcase_end_to_end_rag_architecture_storyboard_contract.json"
SHOWCASE_PATH = ROOT / "evaluation-data/showcase/end_to_end_rag_storyboard_v1.json"
MANIFEST_PATH = ROOT / "evaluation-data/showcase/showcase_manifest_v1.json"
GRAPH_PATH = ROOT / "evaluation-data/showcase/graph_retrieval_visualization_v1.json"
AGENT_PATH = ROOT / "evaluation-data/showcase/guarded_agent_decision_trace_v1.json"
TASK0219_PATH = ROOT / "evaluation-data/results/task0219-unified-showcase-demo-entry-point/summary.json"
TECH_MERMAID_PATH = ROOT / "docs/diagrams/showcase_end_to_end_rag_architecture.mmd"
EXEC_MERMAID_PATH = ROOT / "docs/diagrams/showcase_end_to_end_rag_executive.mmd"
OVERLAY_MERMAID_PATH = ROOT / "docs/diagrams/showcase_end_to_end_scenario_overlay.mmd"
DRAWIO_PATH = ROOT / "docs/diagrams/showcase_end_to_end_rag_drawio_prompt.md"
DOC_PATH = ROOT / "docs/SHOWCASE_END_TO_END_RAG_STORYBOARD.md"
TALK_PATH = ROOT / "docs/SHOWCASE_END_TO_END_TALK_TRACK.md"
VIDEO_PATH = ROOT / "docs/SHOWCASE_VIDEO_STORYBOARD.md"
README_FRAGMENT_PATH = ROOT / "docs/fragments/SHOWCASE_ARCHITECTURE_README.md"
REPORT_PATH = ROOT / "docs/TASK0222_SHOWCASE_END_TO_END_RAG_ARCHITECTURE_STORYBOARD_REPORT.md"


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
    return sorted(set(row.strip() for row in [*tracked, *untracked] if row.strip()))


def mutation_audit(paths: list[str] | None = None) -> dict[str, bool]:
    rows = paths if paths is not None else changed_paths()
    prefixes = {
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
    }
    return {name: any(any(row.startswith(prefix) for prefix in ps) for row in rows) for name, ps in prefixes.items()}


def load_authorities() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = [MANIFEST_PATH, GRAPH_PATH, AGENT_PATH, TASK0219_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing_authority:" + ",".join(missing))
    return read_json(MANIFEST_PATH), read_json(GRAPH_PATH), read_json(AGENT_PATH), read_json(TASK0219_PATH)


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "task_type": "showcase_storyboard",
        "authorities": [
            "evaluation-data/showcase/showcase_manifest_v1.json",
            "evaluation-data/results/task0219-unified-showcase-demo-entry-point/summary.json",
            "evaluation-data/showcase/graph_retrieval_visualization_v1.json",
            "evaluation-data/showcase/guarded_agent_decision_trace_v1.json",
        ],
        "required_invariants": {
            "maximum_recovery_attempt_count": 1,
            "graph_runtime_hop_depth": 1,
            "planner_enabled": False,
            "unbounded_agent_loop_enabled": False,
            "candidate_and_evidence_distinct": True,
            "graph_candidate_rejoins_reranking_path": True,
        },
        "runtime_mutation_allowed": False,
        "git_commit_created": False,
    }


def claim_matrix_valid(matrix: Mapping[str, Any]) -> bool:
    claims = matrix.get("claims") or []
    return (
        matrix.get("claim_evidence_matrix_valid") is True
        and int(matrix.get("unsupported_showcase_claim_count") or 0) == 0
        and len(claims) >= 8
        and all(row.get("claim") and row.get("artifact") and row.get("authority") for row in claims)
    )


def sensitive_scan(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "api_key_assignment": re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*[^\s\"']+"),
        "authorization_header": re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+"),
        "database_url": re.compile(r"(?i)(postgres(?:ql)?|mysql|mongodb)://[^\s\"']+"),
        "private_host_path": re.compile(r"/(?:home|data/envs)/[^\s\"']+"),
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


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK0222 Showcase End-to-End RAG Architecture Storyboard Report

```text
task_id=TASK-0222
task_status={summary.get('task_status')}
end_to_end_rag_storyboard_v1_digest={summary.get('end_to_end_rag_storyboard_v1_digest')}
data_plane_ready={str(summary.get('data_plane_ready')).lower()}
control_plane_ready={str(summary.get('control_plane_ready')).lower()}
validation_plane_ready={str(summary.get('validation_plane_ready')).lower()}
candidate_and_evidence_visually_distinct={str(summary.get('candidate_and_evidence_visually_distinct')).lower()}
graph_candidate_rejoins_reranking_path={str(summary.get('graph_candidate_rejoins_reranking_path')).lower()}
unsupported_showcase_claim_count={summary.get('unsupported_showcase_claim_count')}
```

TASK-0222 composes the frozen TASK-0218/0219/0220/0221 showcase authorities into one deterministic presentation model. It introduces no new RAG capability. The storyboard makes Candidate/Evidence separation, Guarded Agent control, bounded Structure/Graph recovery, shared BGE reranking, and Answerability fail-closed behavior visible in one narrative.
"""


def required_artifact_paths() -> list[Path]:
    return [
        CONTRACT_PATH,
        SHOWCASE_PATH,
        TECH_MERMAID_PATH,
        EXEC_MERMAID_PATH,
        OVERLAY_MERMAID_PATH,
        DRAWIO_PATH,
        DOC_PATH,
        TALK_PATH,
        VIDEO_PATH,
        README_FRAGMENT_PATH,
        REPORT_PATH,
        RESULT_DIR / "summary.json",
        RESULT_DIR / "authority_alignment.json",
        RESULT_DIR / "normalized_storyboard.json",
        RESULT_DIR / "storyboard_reproducibility.json",
        RESULT_DIR / "claim_evidence_matrix.json",
        RESULT_DIR / "storyboard_integrity.json",
        RESULT_DIR / "sensitive_value_scan.json",
    ]


def verify_task0222_artifacts(*, update_summary: bool = True) -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in required_artifact_paths() if not path.exists()]
    summary = read_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    model = read_json(SHOWCASE_PATH) if SHOWCASE_PATH.exists() else {}
    integrity = read_json(RESULT_DIR / "storyboard_integrity.json") if (RESULT_DIR / "storyboard_integrity.json").exists() else {}
    reproduction = read_json(RESULT_DIR / "storyboard_reproducibility.json") if (RESULT_DIR / "storyboard_reproducibility.json").exists() else {}
    claims = read_json(RESULT_DIR / "claim_evidence_matrix.json") if (RESULT_DIR / "claim_evidence_matrix.json").exists() else {}
    sensitive = read_json(RESULT_DIR / "sensitive_value_scan.json") if (RESULT_DIR / "sensitive_value_scan.json").exists() else {}
    checks = {
        "task_complete": summary.get("task_status") == "complete",
        "digest_valid": bool(model) and verify_storyboard_digest(model),
        "planes_ready": all(summary.get(key) is True for key in ("data_plane_ready", "control_plane_ready", "validation_plane_ready")),
        "candidate_evidence_distinct": integrity.get("candidate_and_evidence_distinct") is True,
        "graph_rejoins_reranking": integrity.get("graph_rejoins_pool") is True and integrity.get("pool_precedes_reranker") is True,
        "bounded_recovery": integrity.get("recovery_budget_one") is True,
        "graph_one_hop": integrity.get("graph_one_hop") is True,
        "four_overlays_valid": all(integrity.get(key) is True for key in ("s01_overlay", "s02_overlay", "s03_overlay", "s04_overlay")),
        "claim_matrix_valid": claim_matrix_valid(claims),
        "no_hidden_cot": integrity.get("no_hidden_cot") is True and not contains_forbidden_reasoning(model),
        "reproducible": reproduction.get("storyboard_digest_reproducible") is True and reproduction.get("storyboard_semantically_consistent") is True,
        "sensitive_scan": sensitive.get("sensitive_value_scan_passed") is True,
        "runtime_unchanged": all(summary.get(key) is False for key in (
            "core_runtime_source_changed",
            "retrieval_runtime_source_changed",
            "agent_runtime_source_changed",
            "graph_runtime_source_changed",
            "reranker_runtime_source_changed",
            "evidence_runtime_source_changed",
            "answer_runtime_source_changed",
        )),
        "git_commit_not_created": summary.get("git_commit_created") is False,
    }
    errors = (["required_artifact_missing"] if missing else []) + [key for key, value in checks.items() if not value]
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


def run_task0222(*, write: bool = True) -> dict[str, Any]:
    manifest, graph, agent, task0219 = load_authorities()
    authority_checks = validate_authorities(manifest, graph, agent, task0219)
    if not all(authority_checks.values()):
        failed = [key for key, value in authority_checks.items() if not value]
        raise RuntimeError("authority_alignment_failed:" + ",".join(failed))

    model1 = build_storyboard(manifest, graph, agent, task0219)
    model2 = build_storyboard(manifest, graph, agent, task0219)
    integrity = storyboard_integrity(model1)
    claims = build_claim_evidence_matrix(
        query_digest=str(manifest["showcase_demo_query_set_v1_digest"]),
        graph_digest=str(graph["graph_showcase_visualization_v1_digest"]),
        agent_digest=str(agent["guarded_agent_decision_trace_v1_digest"]),
    )
    reproduction = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "storyboard_generation_run_count": 2,
        "run1_digest": model1["end_to_end_rag_storyboard_v1_digest"],
        "run2_digest": model2["end_to_end_rag_storyboard_v1_digest"],
        "storyboard_semantically_consistent": model1 == model2,
        "storyboard_digest_reproducible": model1["end_to_end_rag_storyboard_v1_digest"] == model2["end_to_end_rag_storyboard_v1_digest"],
    }

    technical = render_technical_mermaid(model1)
    executive = render_executive_mermaid(model1)
    overlay = render_scenario_overlay_mermaid(model1)
    drawio = render_drawio_handoff(model1)
    documentation = render_storyboard_document(model1)
    talk = render_talk_track(model1)
    video = render_video_storyboard(model1)
    readme_fragment = render_readme_fragment(model1)

    mutation = mutation_audit()
    planes = (model1.get("architecture") or {}).get("planes") or {}
    stages = {row.get("stage_id") for row in (model1.get("architecture") or {}).get("stages") or []}
    overlays = model1.get("scenario_overlays") or {}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete",
        "current_stage": "project_showcase_delivery",
        "source_repository_head": git_head(),
        "task0218_showcase_authority_loaded": True,
        "task0219_demo_authority_loaded": True,
        "task0220_graph_visualization_authority_loaded": True,
        "task0221_agent_trace_authority_loaded": True,
        "showcase_query_set_digest": manifest.get("showcase_demo_query_set_v1_digest"),
        "task0220_graph_visualization_digest": graph.get("graph_showcase_visualization_v1_digest"),
        "task0221_agent_trace_digest": agent.get("guarded_agent_decision_trace_v1_digest"),
        "data_plane_ready": bool(planes.get("data_plane")),
        "control_plane_ready": bool(planes.get("control_plane")),
        "validation_plane_ready": bool(planes.get("validation_plane")),
        "retriever_stage_present": "initial_retrieval" in stages,
        "candidate_pool_stage_present": "candidate_pool" in stages and "unified_candidate_pool" in stages,
        "guarded_agent_stage_present": "guard_decision" in stages,
        "structure_recovery_stage_present": "structure_recovery" in stages,
        "graph_recovery_stage_present": "graph_recovery" in stages,
        "reranker_stage_present": "reranking" in stages,
        "evidence_stage_present": "evidence_composition" in stages,
        "answerability_stage_present": "answerability" in stages,
        "answer_stage_present": "answer" in stages,
        "fail_closed_stage_present": "fail_closed" in stages,
        "candidate_and_evidence_visually_distinct": integrity["candidate_and_evidence_distinct"],
        "graph_candidate_rejoins_reranking_path": integrity["graph_rejoins_pool"] and integrity["pool_precedes_reranker"],
        "structure_recovery_bounded": (overlays.get("S02") or {}).get("recovery_attempt_count") == 1 and (overlays.get("S02") or {}).get("maximum_recovery_attempt_count") == 1,
        "graph_recovery_bounded": (overlays.get("S03") or {}).get("recovery_attempt_count") == 1 and (overlays.get("S03") or {}).get("maximum_recovery_attempt_count") == 1,
        "maximum_recovery_attempt_count": 1,
        "graph_runtime_hop_depth": 1,
        "planner_enabled": False,
        "unbounded_agent_loop_enabled": False,
        "scenario_s01_overlay_valid": integrity["s01_overlay"],
        "scenario_s02_overlay_valid": integrity["s02_overlay"],
        "scenario_s03_overlay_valid": integrity["s03_overlay"],
        "scenario_s04_overlay_valid": integrity["s04_overlay"],
        "normalized_storyboard_created": True,
        "end_to_end_rag_storyboard_v1_digest": model1["end_to_end_rag_storyboard_v1_digest"],
        **reproduction,
        "technical_architecture_mermaid_ready": bool(technical.strip()),
        "executive_architecture_mermaid_ready": bool(executive.strip()),
        "scenario_overlay_mermaid_ready": bool(overlay.strip()),
        "drawio_handoff_ready": bool(drawio.strip()),
        "storyboard_documentation_ready": bool(documentation.strip()),
        "talk_track_ready": all(marker in talk for marker in ("60-second", "3-minute", "8-minute")),
        "video_storyboard_ready": "00:00" in video and "04:20" in video,
        "readme_architecture_fragment_ready": "```mermaid" in readme_fragment,
        "claim_evidence_matrix_valid": claim_matrix_valid(claims),
        "unsupported_showcase_claim_count": int(claims.get("unsupported_showcase_claim_count") or 0),
        "hidden_chain_of_thought_exposed": contains_forbidden_reasoning(model1),
        **mutation,
        "performance_optimization_reopened": False,
        "sensitive_value_scan_passed": False,
        "independent_verifier_passed": False,
        "git_commit_created": False,
        "next_recommended_task": "TASK-0223",
    }

    required_true = [
        "data_plane_ready", "control_plane_ready", "validation_plane_ready",
        "retriever_stage_present", "candidate_pool_stage_present", "guarded_agent_stage_present",
        "structure_recovery_stage_present", "graph_recovery_stage_present", "reranker_stage_present",
        "evidence_stage_present", "answerability_stage_present", "answer_stage_present", "fail_closed_stage_present",
        "candidate_and_evidence_visually_distinct", "graph_candidate_rejoins_reranking_path",
        "structure_recovery_bounded", "graph_recovery_bounded",
        "scenario_s01_overlay_valid", "scenario_s02_overlay_valid", "scenario_s03_overlay_valid", "scenario_s04_overlay_valid",
        "storyboard_semantically_consistent", "storyboard_digest_reproducible",
        "technical_architecture_mermaid_ready", "executive_architecture_mermaid_ready", "scenario_overlay_mermaid_ready",
        "drawio_handoff_ready", "storyboard_documentation_ready", "talk_track_ready", "video_storyboard_ready",
        "readme_architecture_fragment_ready", "claim_evidence_matrix_valid",
    ]
    if not all(summary[key] is True for key in required_true) or summary["unsupported_showcase_claim_count"] != 0 or summary["hidden_chain_of_thought_exposed"] is not False or any(mutation.values()):
        summary["task_status"] = "partial"

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        TECH_MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
        README_FRAGMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(CONTRACT_PATH, build_contract())
        write_json(RESULT_DIR / "authority_alignment.json", {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "checks": authority_checks, "all_authorities_aligned": all(authority_checks.values())})
        write_json(RESULT_DIR / "normalized_storyboard.json", model1)
        write_json(RESULT_DIR / "storyboard_reproducibility.json", reproduction)
        write_json(RESULT_DIR / "claim_evidence_matrix.json", claims)
        write_json(RESULT_DIR / "storyboard_integrity.json", integrity)
        write_json(SHOWCASE_PATH, model1)
        TECH_MERMAID_PATH.write_text(technical, encoding="utf-8")
        EXEC_MERMAID_PATH.write_text(executive, encoding="utf-8")
        OVERLAY_MERMAID_PATH.write_text(overlay, encoding="utf-8")
        DRAWIO_PATH.write_text(drawio, encoding="utf-8")
        DOC_PATH.write_text(documentation, encoding="utf-8")
        TALK_PATH.write_text(talk, encoding="utf-8")
        VIDEO_PATH.write_text(video, encoding="utf-8")
        README_FRAGMENT_PATH.write_text(readme_fragment, encoding="utf-8")
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        scan_paths = [
            SHOWCASE_PATH, TECH_MERMAID_PATH, EXEC_MERMAID_PATH, OVERLAY_MERMAID_PATH, DRAWIO_PATH,
            DOC_PATH, TALK_PATH, VIDEO_PATH, README_FRAGMENT_PATH, REPORT_PATH,
            RESULT_DIR / "normalized_storyboard.json", RESULT_DIR / "claim_evidence_matrix.json",
        ]
        sensitive = sensitive_scan(scan_paths)
        write_json(RESULT_DIR / "sensitive_value_scan.json", sensitive)
        summary["sensitive_value_scan_passed"] = sensitive["sensitive_value_scan_passed"]
        if summary["sensitive_value_scan_passed"] is not True:
            summary["task_status"] = "partial"
        write_json(RESULT_DIR / "summary.json", summary)
        verification = verify_task0222_artifacts(update_summary=True)
        summary = read_json(RESULT_DIR / "summary.json")
        summary["independent_verifier_passed"] = verification["verification_passed"]
        if verification["verification_passed"] is not True:
            summary["task_status"] = "partial"
        write_json(RESULT_DIR / "summary.json", summary)
    return summary
