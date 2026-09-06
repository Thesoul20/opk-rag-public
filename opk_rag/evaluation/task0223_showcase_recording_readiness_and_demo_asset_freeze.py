from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from opk_rag.runtime.dotenv import load_project_env
from opk_rag.showcase.demo import list_showcase_scenarios, load_showcase_authority, preflight as demo_preflight
from opk_rag.showcase.recording_pack import (
    CANONICAL_SCENARIO_ORDER,
    EXECUTIVE_VISUAL_ASSETS,
    EXPECTED_SCENARIOS,
    FALLBACK_TRACES,
    PRESENTATION_DOCS,
    TECHNICAL_VISUAL_ASSETS,
    build_claim_policy,
    build_fallback_matrix,
    build_public_asset_inventory,
    build_recording_pack,
    file_sha256,
    recording_dry_run,
    render_recording_guide,
    render_safety_checklist,
    scan_unsupported_affirmative_claims,
    validate_authorities,
    validate_live_scenario,
    verify_claim_policy_digest,
    verify_recording_pack_digest,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0223"
SCHEMA_VERSION = "opk-rag.task0223.showcase-recording-readiness-and-demo-asset-freeze.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0223-showcase-recording-readiness-and-demo-asset-freeze"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0223_showcase_recording_readiness_and_demo_asset_freeze_contract.json"
RECORDING_PACK_PATH = ROOT / "evaluation-data/showcase/showcase_recording_pack_v1.json"
CLAIM_POLICY_PATH = ROOT / "evaluation-data/showcase/showcase_claim_policy_v1.json"
MANIFEST_PATH = ROOT / "evaluation-data/showcase/showcase_manifest_v1.json"
TASK0219_SUMMARY_PATH = ROOT / "evaluation-data/results/task0219-unified-showcase-demo-entry-point/summary.json"
GRAPH_PATH = ROOT / "evaluation-data/showcase/graph_retrieval_visualization_v1.json"
AGENT_PATH = ROOT / "evaluation-data/showcase/guarded_agent_decision_trace_v1.json"
STORYBOARD_PATH = ROOT / "evaluation-data/showcase/end_to_end_rag_storyboard_v1.json"
CLAIM_MATRIX_PATH = ROOT / "evaluation-data/results/task0222-showcase-end-to-end-rag-architecture-storyboard/claim_evidence_matrix.json"
RECORDING_GUIDE_PATH = ROOT / "docs/SHOWCASE_RECORDING_GUIDE.md"
SAFETY_CHECKLIST_PATH = ROOT / "docs/SHOWCASE_RECORDING_SAFETY_CHECKLIST.md"
REPORT_PATH = ROOT / "docs/TASK0223_SHOWCASE_RECORDING_READINESS_AND_DEMO_ASSET_FREEZE_REPORT.md"
TALK_TRACK_PATH = ROOT / "docs/SHOWCASE_END_TO_END_TALK_TRACK.md"
VIDEO_STORYBOARD_PATH = ROOT / "docs/SHOWCASE_VIDEO_STORYBOARD.md"
README_FRAGMENT_PATH = ROOT / "docs/fragments/SHOWCASE_ARCHITECTURE_README.md"

UPSTREAM_AUTHORITY_PATHS = {
    "task0218": MANIFEST_PATH,
    "task0220": GRAPH_PATH,
    "task0221": AGENT_PATH,
    "task0222": STORYBOARD_PATH,
}

HASHED_UPSTREAM_ASSETS = tuple(
    dict.fromkeys(
        [
            *TECHNICAL_VISUAL_ASSETS,
            *EXECUTIVE_VISUAL_ASSETS,
            "docs/SHOWCASE_END_TO_END_RAG_STORYBOARD.md",
            "docs/SHOWCASE_END_TO_END_TALK_TRACK.md",
            "docs/SHOWCASE_VIDEO_STORYBOARD.md",
            "docs/SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION.md",
            "docs/SHOWCASE_GUARDED_AGENT_DECISION_TRACE.md",
            "docs/fragments/SHOWCASE_ARCHITECTURE_README.md",
        ]
    )
)

EXPECTED_TASK0223_PREFIXES = (
    "tasks/TASK-0223_",
    "opk_rag/showcase/recording_pack.py",
    "opk_rag/evaluation/task0223_",
    "scripts/run_task0223_",
    "scripts/verify_task0223_",
    "scripts/preflight_showcase_recording.py",
    "scripts/export_showcase_recording_pack.py",
    "tests/test_task0223_",
    "evaluation-data/contracts/task0223_",
    "evaluation-data/results/task0223-",
    "evaluation-data/showcase/showcase_recording_pack_v1.json",
    "evaluation-data/showcase/showcase_claim_policy_v1.json",
    "docs/SHOWCASE_RECORDING_GUIDE.md",
    "docs/SHOWCASE_RECORDING_SAFETY_CHECKLIST.md",
    "docs/TASK0223_",
    "docs/PROJECT_DEMO_RUNBOOK.md",
    "PROJECT_STATE.md",
    "CHANGELOG.md",
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
    return sorted(set(row.strip() for row in [*tracked, *untracked] if row.strip()))


def unexpected_worktree_paths(paths: list[str] | None = None) -> list[str]:
    rows = paths if paths is not None else changed_paths()
    return [row for row in rows if not any(row == prefix or row.startswith(prefix) for prefix in EXPECTED_TASK0223_PREFIXES)]


def mutation_audit(paths: list[str] | None = None) -> dict[str, bool]:
    rows = paths if paths is not None else changed_paths()
    groups = {
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
    return {name: any(any(row.startswith(prefix) for prefix in prefixes) for row in rows) for name, prefixes in groups.items()}


def upstream_authority_diff_audit() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for task, path in UPSTREAM_AUTHORITY_PATHS.items():
        completed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        )
        result[f"{task}_authority_diff"] = completed.returncode != 0
    return result


def load_authorities() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = [MANIFEST_PATH, TASK0219_SUMMARY_PATH, GRAPH_PATH, AGENT_PATH, STORYBOARD_PATH, CLAIM_MATRIX_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing_recording_authority:" + ",".join(missing))
    return tuple(read_json(path) for path in required)  # type: ignore[return-value]


def upstream_asset_hashes() -> dict[str, str]:
    missing = [path for path in HASHED_UPSTREAM_ASSETS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError("missing_hashed_showcase_asset:" + ",".join(missing))
    return {path: file_sha256(ROOT / path) for path in HASHED_UPSTREAM_ASSETS}


def run_lightweight_preflight() -> dict[str, Any]:
    load_project_env(ROOT)
    manifest, task0219, graph, agent, storyboard, claim_matrix = load_authorities()
    authority_checks = validate_authorities(manifest, task0219, graph, agent, storyboard, claim_matrix)
    authority = load_showcase_authority()
    raw_demo_preflight = demo_preflight(authority)
    scenario_listing = list_showcase_scenarios()
    listed_ids = [row.get("scenario_id") for row in scenario_listing.get("scenarios") or []]
    cli = subprocess.run(
        ["uv", "run", "opk-rag", "demo", "--list", "--format", "json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    cli_json_valid = False
    if cli.returncode == 0:
        try:
            payload = json.loads(cli.stdout)
            cli_json_valid = [row.get("scenario_id") for row in payload.get("scenarios") or []] == list(EXPECTED_SCENARIOS)
        except json.JSONDecodeError:
            cli_json_valid = False
    unexpected = unexpected_worktree_paths()
    checks = {
        "repository_root_valid": raw_demo_preflight.get("repository_root_valid") is True,
        "all_upstream_authorities_valid": all(authority_checks.values()),
        "scenario_registry_valid": listed_ids == list(EXPECTED_SCENARIOS),
        "demo_cli_available": cli.returncode == 0 and cli_json_valid,
        "production_vector_backend_qdrant": raw_demo_preflight.get("production_vector_backend_valid") is True,
        "showcase_knowledge_base_resolvable": raw_demo_preflight.get("showcase_knowledge_base_resolvable") is True,
        "qdrant_reachable": raw_demo_preflight.get("qdrant_reachable") is True,
        "qdrant_collection_exists": raw_demo_preflight.get("qdrant_collection_exists") is True,
        "runtime_configuration_available": raw_demo_preflight.get("required_runtime_configuration_available") is True,
        "unexpected_worktree_modifications_absent": not unexpected,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checks": checks,
        "preflight_valid": all(checks.values()),
        "unexpected_worktree_modification_count": len(unexpected),
        "unexpected_worktree_paths": unexpected,
        "scenario_ids": listed_ids,
        "knowledge_base_resolvable": raw_demo_preflight.get("showcase_knowledge_base_resolvable") is True,
        "qdrant_reachable": raw_demo_preflight.get("qdrant_reachable") is True,
        "qdrant_collection_exists": raw_demo_preflight.get("qdrant_collection_exists") is True,
        "production_embedding_model": raw_demo_preflight.get("production_embedding_model"),
        "production_reranker_model": raw_demo_preflight.get("production_reranker_model"),
        "graph_runtime_hop_depth": raw_demo_preflight.get("graph_runtime_hop_depth"),
        "database_url_available": raw_demo_preflight.get("database_url_available") is True,
        "actual_public_recording_requires_clean_committed_worktree": True,
    }


def run_live_scenario(sid: str, *, timeout_seconds: int = 180, max_attempts: int = 2) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        completed = subprocess.run(
            ["uv", "run", "opk-rag", "demo", "--scenario", sid, "--format", "json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        payload: dict[str, Any] | None = None
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = None
        if completed.returncode == 0 and payload is not None:
            scenarios = payload.get("scenarios") or []
            if len(scenarios) == 1 and scenarios[0].get("scenario_id") == sid:
                return {
                    "scenario_id": sid,
                    "attempt_count": attempt,
                    "command_returncode": completed.returncode,
                    "trace": scenarios[0],
                }
        blocker = payload.get("blocker") if isinstance(payload, dict) else None
        last_error = f"returncode={completed.returncode};blocker={blocker};stderr_present={bool(completed.stderr.strip())}"
    raise RuntimeError(f"live_scenario_failed:{sid}:{last_error}")


def run_all_live_scenarios() -> dict[str, Any]:
    rows = [run_live_scenario(sid) for sid in EXPECTED_SCENARIOS]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "canonical_scenario_order": list(EXPECTED_SCENARIOS),
        "scenarios": rows,
    }


def validate_live_readiness(live: Mapping[str, Any], graph_authority: Mapping[str, Any]) -> dict[str, Any]:
    rows = {str(row.get("scenario_id")): row for row in live.get("scenarios") or []}
    per_scenario: dict[str, Any] = {}
    for sid in EXPECTED_SCENARIOS:
        wrapper = rows.get(sid) or {}
        trace = wrapper.get("trace") or {}
        checks = validate_live_scenario(sid, trace, graph_authority)
        per_scenario[sid] = {
            "attempt_count": wrapper.get("attempt_count"),
            "checks": checks,
            "live_ready": bool(checks) and all(checks.values()),
            "decision": (trace.get("guard") or {}).get("final_decision"),
            "recovery_attempt_count": (trace.get("guard") or {}).get("recovery_attempt_count"),
            "graph_activated": (trace.get("graph") or {}).get("graph_activated"),
            "graph_hop_depth": (trace.get("graph") or {}).get("hop_depth"),
            "answer_status": (trace.get("answer") or {}).get("status"),
            "refusal_reason_code": (trace.get("answer") or {}).get("refusal_reason_code") or (trace.get("guard") or {}).get("refusal_reason_code"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenarios": per_scenario,
        "all_scenarios_live_ready": all(per_scenario[sid]["live_ready"] for sid in EXPECTED_SCENARIOS),
    }


def video_storyboard_freeze_check(text: str) -> dict[str, Any]:
    markers = ["00:00", "00:20", "00:50", "01:20", "01:50", "02:20", "03:00", "03:30", "04:00"]
    positions = [text.find(marker) for marker in markers]
    valid = all(pos >= 0 for pos in positions) and positions == sorted(positions)
    return {"markers": markers, "positions": positions, "video_storyboard_frozen": valid}


def claim_alignment() -> dict[str, Any]:
    talk = TALK_TRACK_PATH.read_text(encoding="utf-8")
    readme = README_FRAGMENT_PATH.read_text(encoding="utf-8")
    video = VIDEO_STORYBOARD_PATH.read_text(encoding="utf-8")
    talk_hits = scan_unsupported_affirmative_claims(talk)
    readme_hits = scan_unsupported_affirmative_claims(readme)
    video_hits = scan_unsupported_affirmative_claims(video)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "talk_track_sections_present": all(marker in talk for marker in ("60-second", "3-minute", "8-minute")),
        "talk_track_claim_alignment_valid": not talk_hits,
        "readme_claim_alignment_valid": not readme_hits,
        "video_storyboard_claim_alignment_valid": not video_hits,
        "unsupported_spoken_claim_count": len(talk_hits),
        "unsupported_readme_claim_count": len(readme_hits),
        "unsupported_recording_claim_count": len(talk_hits) + len(readme_hits) + len(video_hits),
        "talk_track_findings": talk_hits,
        "readme_findings": readme_hits,
        "video_findings": video_hits,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "task_type": "showcase_recording_freeze",
        "canonical_scenario_order": list(EXPECTED_SCENARIOS),
        "required_live_scenarios": list(EXPECTED_SCENARIOS),
        "required_governance": {
            "maximum_recovery_attempt_count": 1,
            "graph_runtime_hop_depth": 1,
            "planner_enabled": False,
            "unbounded_agent_loop_enabled": False,
        },
        "runtime_mutation_allowed": False,
        "performance_optimization_reopened": False,
        "git_commit_created": False,
    }


def sensitive_scan(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "api_key_assignment": re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*[^\s\"']+"),
        "authorization_header": re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+"),
        "database_url": re.compile(r"(?i)(postgres(?:ql)?|mysql|mongodb)://[^\s\"']+"),
        "private_host_path": re.compile(r"/(?:home|data/envs)/[^\s\"']+"),
    }
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
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
    return f"""# TASK0223 Showcase Recording Readiness and Demo Asset Freeze Report

```text
task_id=TASK-0223
task_status={summary.get('task_status')}
showcase_recording_freeze_decision={summary.get('showcase_recording_freeze_decision')}
showcase_recording_pack_v1_digest={summary.get('showcase_recording_pack_v1_digest')}
canonical_scenario_order={summary.get('canonical_scenario_order')}
recording_s01_live_ready={str(summary.get('recording_s01_live_ready')).lower()}
recording_s02_live_ready={str(summary.get('recording_s02_live_ready')).lower()}
recording_s03_live_ready={str(summary.get('recording_s03_live_ready')).lower()}
recording_s04_live_ready={str(summary.get('recording_s04_live_ready')).lower()}
recording_dry_run_passed={str(summary.get('recording_dry_run_passed')).lower()}
public_safe_asset_count={summary.get('public_safe_asset_count')}
unsupported_recording_claim_count={summary.get('unsupported_recording_claim_count')}
```

TASK-0223 freezes the operational showcase surface rather than changing RAG behavior. The final recording pack fixes scenario order, commands, diagrams, fallbacks, public-safe asset classifications, claim boundaries, and failure handling. Live S01-S04 readiness remains an explicit gate, while any fallback must be labelled as a frozen authoritative artifact.
"""


def required_artifact_paths() -> list[Path]:
    return [
        CONTRACT_PATH,
        RECORDING_PACK_PATH,
        CLAIM_POLICY_PATH,
        RECORDING_GUIDE_PATH,
        SAFETY_CHECKLIST_PATH,
        REPORT_PATH,
        RESULT_DIR / "summary.json",
        RESULT_DIR / "authority_alignment.json",
        RESULT_DIR / "recording_readiness.json",
        RESULT_DIR / "live_scenarios.json",
        RESULT_DIR / "recording_pack_reproducibility.json",
        RESULT_DIR / "fallback_matrix.json",
        RESULT_DIR / "public_asset_inventory.json",
        RESULT_DIR / "claim_alignment.json",
        RESULT_DIR / "recording_dry_run.json",
        RESULT_DIR / "sensitive_value_scan.json",
    ]


def verify_task0223_artifacts(*, update_summary: bool = True) -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in required_artifact_paths() if not path.exists()]
    summary = read_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    pack = read_json(RECORDING_PACK_PATH) if RECORDING_PACK_PATH.exists() else {}
    policy = read_json(CLAIM_POLICY_PATH) if CLAIM_POLICY_PATH.exists() else {}
    readiness = read_json(RESULT_DIR / "recording_readiness.json") if (RESULT_DIR / "recording_readiness.json").exists() else {}
    reproducibility = read_json(RESULT_DIR / "recording_pack_reproducibility.json") if (RESULT_DIR / "recording_pack_reproducibility.json").exists() else {}
    fallback = read_json(RESULT_DIR / "fallback_matrix.json") if (RESULT_DIR / "fallback_matrix.json").exists() else {}
    inventory = read_json(RESULT_DIR / "public_asset_inventory.json") if (RESULT_DIR / "public_asset_inventory.json").exists() else {}
    claims = read_json(RESULT_DIR / "claim_alignment.json") if (RESULT_DIR / "claim_alignment.json").exists() else {}
    dry_run = read_json(RESULT_DIR / "recording_dry_run.json") if (RESULT_DIR / "recording_dry_run.json").exists() else {}
    sensitive = read_json(RESULT_DIR / "sensitive_value_scan.json") if (RESULT_DIR / "sensitive_value_scan.json").exists() else {}
    upstream_hashes_valid = bool(pack.get("upstream_asset_hashes")) and all(
        (ROOT / path).is_file() and file_sha256(ROOT / path) == digest
        for path, digest in (pack.get("upstream_asset_hashes") or {}).items()
    )
    checks = {
        "task_complete": summary.get("task_status") == "complete",
        "freeze_decision": summary.get("showcase_recording_freeze_decision") == "freeze",
        "pack_digest_valid": bool(pack) and verify_recording_pack_digest(pack),
        "claim_policy_digest_valid": bool(policy) and verify_claim_policy_digest(policy),
        "upstream_asset_hashes_valid": upstream_hashes_valid,
        "all_live_scenarios_ready": readiness.get("all_scenarios_live_ready") is True,
        "all_fallbacks_ready": set((fallback.get("scenarios") or {}).keys()) == set(EXPECTED_SCENARIOS),
        "inventory_valid": inventory.get("public_asset_inventory_valid") is True and inventory.get("unclassified_asset_count") == 0,
        "claim_alignment_valid": claims.get("talk_track_claim_alignment_valid") is True and claims.get("readme_claim_alignment_valid") is True and claims.get("unsupported_recording_claim_count") == 0,
        "dry_run_passed": dry_run.get("recording_dry_run_passed") is True,
        "reproducible": reproducibility.get("recording_pack_semantically_consistent") is True and reproducibility.get("recording_pack_digest_reproducible") is True,
        "runtime_unchanged": all(summary.get(key) is False for key in (
            "core_runtime_source_changed",
            "retrieval_runtime_source_changed",
            "agent_runtime_source_changed",
            "graph_runtime_source_changed",
            "reranker_runtime_source_changed",
            "evidence_runtime_source_changed",
            "answer_runtime_source_changed",
        )),
        "upstream_authorities_unchanged": all(summary.get(key) is False for key in (
            "task0218_authority_diff",
            "task0220_authority_diff",
            "task0221_authority_diff",
            "task0222_authority_diff",
        )),
        "sensitive_scan": sensitive.get("sensitive_value_scan_passed") is True,
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
        if not verification["verification_passed"]:
            summary["task_status"] = "partial" if summary.get("showcase_recording_freeze_decision") != "blocked" else "blocked"
        write_json(RESULT_DIR / "summary.json", summary)
    return verification


def run_task0223(*, write: bool = True, live_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    source_head = git_head()
    manifest, task0219, graph, agent, storyboard, claim_matrix = load_authorities()
    authority_checks = validate_authorities(manifest, task0219, graph, agent, storyboard, claim_matrix)
    if not all(authority_checks.values()):
        failed = [key for key, value in authority_checks.items() if not value]
        raise RuntimeError("recording_authority_alignment_failed:" + ",".join(failed))

    preflight = run_lightweight_preflight()
    if preflight.get("preflight_valid") is not True:
        failed = [key for key, value in (preflight.get("checks") or {}).items() if value is not True]
        raise RuntimeError("recording_preflight_failed:" + ",".join(failed))

    live = dict(live_payload) if live_payload is not None else run_all_live_scenarios()
    readiness = validate_live_readiness(live, graph)

    policy = build_claim_policy(claim_matrix, storyboard_digest=str(storyboard.get("end_to_end_rag_storyboard_v1_digest")))
    hashes = upstream_asset_hashes()
    pack1 = build_recording_pack(
        manifest=manifest,
        graph=graph,
        agent=agent,
        storyboard=storyboard,
        claim_policy=policy,
        upstream_asset_hashes=hashes,
    )
    pack2 = build_recording_pack(
        manifest=manifest,
        graph=graph,
        agent=agent,
        storyboard=storyboard,
        claim_policy=policy,
        upstream_asset_hashes=hashes,
    )
    reproducibility = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "recording_pack_generation_run_count": 2,
        "run1_digest": pack1["showcase_recording_pack_v1_digest"],
        "run2_digest": pack2["showcase_recording_pack_v1_digest"],
        "recording_pack_semantically_consistent": pack1 == pack2,
        "recording_pack_digest_reproducible": pack1["showcase_recording_pack_v1_digest"] == pack2["showcase_recording_pack_v1_digest"],
    }

    guide = render_recording_guide(pack1)
    safety = render_safety_checklist()
    fallback = build_fallback_matrix()
    claims = claim_alignment()
    video_freeze = video_storyboard_freeze_check(VIDEO_STORYBOARD_PATH.read_text(encoding="utf-8"))
    mutation = mutation_audit()
    authority_diff = upstream_authority_diff_audit()

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(CONTRACT_PATH, build_contract())
        write_json(CLAIM_POLICY_PATH, policy)
        write_json(RECORDING_PACK_PATH, pack1)
        RECORDING_GUIDE_PATH.write_text(guide, encoding="utf-8")
        SAFETY_CHECKLIST_PATH.write_text(safety, encoding="utf-8")

    inventory = build_public_asset_inventory(ROOT)
    dry_run = recording_dry_run(
        pack1,
        ROOT,
        recording_guide_ready=RECORDING_GUIDE_PATH.exists() if write else bool(guide),
        safety_checklist_ready=SAFETY_CHECKLIST_PATH.exists() if write else bool(safety),
    )

    fallback_ready = {
        sid: (
            (ROOT / (fallback.get("scenarios") or {}).get(sid, {}).get("fallback_trace", "")).is_file()
            and all((ROOT / path).is_file() for path in (fallback.get("scenarios") or {}).get(sid, {}).get("fallback_visualizations", []))
        )
        for sid in EXPECTED_SCENARIOS
    }
    live_ready = {sid: bool((readiness.get("scenarios") or {}).get(sid, {}).get("live_ready")) for sid in EXPECTED_SCENARIOS}

    public_scan_paths = [
        RECORDING_PACK_PATH,
        CLAIM_POLICY_PATH,
        RECORDING_GUIDE_PATH,
        SAFETY_CHECKLIST_PATH,
        TALK_TRACK_PATH,
        VIDEO_STORYBOARD_PATH,
        README_FRAGMENT_PATH,
        *[ROOT / path for path in TECHNICAL_VISUAL_ASSETS],
        *[ROOT / path for path in EXECUTIVE_VISUAL_ASSETS],
        *[ROOT / path for path in FALLBACK_TRACES.values()],
    ]
    if write:
        write_json(RESULT_DIR / "authority_alignment.json", {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "checks": authority_checks, "all_authorities_aligned": all(authority_checks.values())})
        write_json(RESULT_DIR / "preflight.json", preflight)
        write_json(RESULT_DIR / "live_scenarios.json", live)
        write_json(RESULT_DIR / "recording_readiness.json", readiness)
        write_json(RESULT_DIR / "recording_pack_reproducibility.json", reproducibility)
        write_json(RESULT_DIR / "fallback_matrix.json", fallback)
        write_json(RESULT_DIR / "public_asset_inventory.json", inventory)
        write_json(RESULT_DIR / "claim_alignment.json", {**claims, **video_freeze})
        write_json(RESULT_DIR / "recording_dry_run.json", dry_run)

    sensitive = sensitive_scan(public_scan_paths)
    if write:
        write_json(RESULT_DIR / "sensitive_value_scan.json", sensitive)

    required_boolean_gates = [
        preflight.get("preflight_valid") is True,
        readiness.get("all_scenarios_live_ready") is True,
        all(fallback_ready.values()),
        inventory.get("public_asset_inventory_valid") is True,
        inventory.get("unclassified_asset_count") == 0,
        claims.get("talk_track_claim_alignment_valid") is True,
        claims.get("readme_claim_alignment_valid") is True,
        claims.get("unsupported_recording_claim_count") == 0,
        video_freeze.get("video_storyboard_frozen") is True,
        dry_run.get("recording_dry_run_passed") is True,
        reproducibility.get("recording_pack_semantically_consistent") is True,
        reproducibility.get("recording_pack_digest_reproducible") is True,
        sensitive.get("sensitive_value_scan_passed") is True,
        not any(mutation.values()),
        not any(authority_diff.values()),
    ]
    freeze = all(required_boolean_gates)
    freeze_decision = "freeze" if freeze else ("blocked" if not readiness.get("all_scenarios_live_ready") or not all(authority_checks.values()) else "partial")
    task_status = "complete" if freeze else freeze_decision

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": task_status,
        "current_stage": "project_showcase_delivery",
        "source_repository_head": source_head,
        "task0218_showcase_authority_loaded": True,
        "task0219_demo_authority_loaded": True,
        "task0220_graph_visualization_authority_loaded": True,
        "task0221_agent_trace_authority_loaded": True,
        "task0222_storyboard_authority_loaded": True,
        "showcase_recording_freeze_decision": freeze_decision,
        "showcase_recording_sequence_v1_frozen": freeze,
        "showcase_recording_pack_v1_created": True,
        "showcase_recording_pack_v1_digest": pack1["showcase_recording_pack_v1_digest"],
        "showcase_claim_policy_v1_created": True,
        "showcase_claim_policy_v1_digest": policy["showcase_claim_policy_v1_digest"],
        "canonical_scenario_order": CANONICAL_SCENARIO_ORDER,
        **{f"recording_{sid.lower()}_live_ready": live_ready[sid] for sid in EXPECTED_SCENARIOS},
        **{f"{sid.lower()}_fallback_ready": fallback_ready[sid] for sid in EXPECTED_SCENARIOS},
        "technical_architecture_asset_ready": (ROOT / "docs/diagrams/showcase_end_to_end_rag_architecture.mmd").is_file(),
        "graph_visualization_asset_ready": (ROOT / "docs/diagrams/showcase_graph_retrieval_s03.mmd").is_file(),
        "agent_visualization_asset_ready": (ROOT / "docs/diagrams/showcase_guarded_agent_scenarios.mmd").is_file(),
        "executive_architecture_asset_ready": (ROOT / "docs/diagrams/showcase_end_to_end_rag_executive.mmd").is_file(),
        "recording_guide_ready": RECORDING_GUIDE_PATH.is_file() if write else bool(guide),
        "recording_safety_checklist_ready": SAFETY_CHECKLIST_PATH.is_file() if write else bool(safety),
        "video_storyboard_frozen": video_freeze["video_storyboard_frozen"],
        "talk_track_frozen": TALK_TRACK_PATH.is_file() and claims["talk_track_sections_present"],
        "readme_architecture_fragment_frozen": README_FRAGMENT_PATH.is_file(),
        "public_asset_inventory_valid": inventory["public_asset_inventory_valid"],
        "public_safe_asset_count": inventory["public_safe_asset_count"],
        "unclassified_asset_count": inventory["unclassified_asset_count"],
        "talk_track_claim_alignment_valid": claims["talk_track_claim_alignment_valid"],
        "readme_claim_alignment_valid": claims["readme_claim_alignment_valid"],
        "unsupported_spoken_claim_count": claims["unsupported_spoken_claim_count"],
        "unsupported_readme_claim_count": claims["unsupported_readme_claim_count"],
        "unsupported_recording_claim_count": claims["unsupported_recording_claim_count"],
        "maximum_recovery_attempt_count": 1,
        "graph_runtime_hop_depth": 1,
        "planner_enabled": False,
        "unbounded_agent_loop_enabled": False,
        "recording_dry_run_count": dry_run["recording_dry_run_count"],
        "recording_dry_run_passed": dry_run["recording_dry_run_passed"],
        **reproducibility,
        "demo_specific_retrieval_policy": False,
        "demo_specific_graph_policy": False,
        "demo_specific_agent_policy": False,
        "demo_specific_answerability_policy": False,
        "demo_specific_threshold_override": False,
        "demo_specific_candidate_injection": False,
        **authority_diff,
        **mutation,
        "performance_optimization_reopened": False,
        "estimated_showcase_duration_seconds": 260,
        "unexpected_worktree_modification_count": preflight.get("unexpected_worktree_modification_count"),
        "sensitive_value_scan_passed": sensitive["sensitive_value_scan_passed"],
        "independent_verifier_passed": False,
        "git_commit_created": False,
        "next_recommended_task": "TASK-0224",
    }

    if write:
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        verification = verify_task0223_artifacts(update_summary=True)
        summary = read_json(RESULT_DIR / "summary.json")
        summary["independent_verifier_passed"] = verification["verification_passed"]
        if not verification["verification_passed"]:
            summary["task_status"] = "partial" if summary.get("showcase_recording_freeze_decision") != "blocked" else "blocked"
        write_json(RESULT_DIR / "summary.json", summary)
    return summary
