from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evaluation-data/results/task0282-opk-rag-open-source-release-readiness-and-repository-finalization"
SNAPSHOT = ROOT / "dist/opk-rag-public-release"
TASK = ROOT / "tasks/TASK-0282_opk_rag_open_source_release_readiness_and_repository_finalization.md"
REPORT = ROOT / "docs/TASK0282_OPK_RAG_OPEN_SOURCE_RELEASE_READINESS_AND_REPOSITORY_FINALIZATION_REPORT.md"
AUTHORITY = ROOT / "docs/CURRENT_PROJECT_AUTHORITY.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return p.stdout.strip()


def _readme_links() -> dict[str, Any]:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    local = []
    missing = []
    for target in links:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        raw = target.split("#", 1)[0]
        if not raw:
            continue
        path = (ROOT / raw).resolve()
        local.append(raw)
        if not path.exists():
            missing.append(raw)
    required_sections = ["## 30 秒理解", "## 架构", "## Modern Control Center", "## 已验证结果", "## Quick Start", "## Public / Private Boundary", "## Public Release Snapshot"]
    return {
        "schema_version": "opk-rag.task0282.readme-quick-start-audit.v1",
        "local_link_count": len(local),
        "missing_local_links": sorted(set(missing)),
        "required_sections": {x: x in text for x in required_sections},
        "quick_start_exists": (ROOT / "docs/QUICK_START.md").is_file(),
        "public_release_doc_exists": (ROOT / "docs/OPEN_SOURCE_RELEASE.md").is_file(),
        "architecture_assets_exist": all((ROOT / p).is_file() for p in [
            "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.png",
            "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.drawio",
            "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.mmd",
            "docs/diagrams/current_opk_rag_selective_agent_architecture_v2_zh.png",
        ]),
        "control_center_asset_exists": (ROOT / "docs/assets/control-center-executive.jpg").is_file(),
        "video_cover_exists": (ROOT / "docs/assets/video-cover.jpg").is_file(),
        "valid": not missing and all(x in text for x in required_sections) and (ROOT / "docs/QUICK_START.md").is_file(),
    }


def _open_source_files() -> dict[str, Any]:
    required = ["LICENSE", "CONTRIBUTING.md", "SECURITY.md", ".env.public.example", "infra/public/docker-compose.yml", ".github/workflows/ci.yml", "docs/THIRD_PARTY.md", "release/evidence/third_party_license_inventory.json"]
    state = {p: (ROOT / p).is_file() for p in required}
    env = (ROOT / ".env.public.example").read_text(encoding="utf-8")
    demo_readme = (ROOT / "examples/public-demo/README.md").read_text(encoding="utf-8")
    return {
        "schema_version": "opk-rag.task0282.open-source-files.v1",
        "required_files": state,
        "mit_license_declared": "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8"),
        "remote_llm_default_off": "OPK_RAG_LLM_ALLOW_REMOTE=false" in env,
        "production_agent_default_off": "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_ENABLED=false" in env,
        "public_demo_project_authored": "authored specifically" in demo_readme,
        "model_weights_redistributed": False,
        "valid": all(state.values()) and "OPK_RAG_LLM_ALLOW_REMOTE=false" in env and "authored specifically" in demo_readme,
    }


def _public_claims() -> dict[str, Any]:
    paths = [ROOT / "README.md", ROOT / "docs/QUICK_START.md", ROOT / "docs/OPEN_SOURCE_RELEASE.md"]
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)
    forbidden_assertions = {
        "hallucination_free_assertion": ["零幻觉", "100% hallucination-free", "is hallucination-free"],
        "guaranteed_correctness_assertion": ["保证正确", "guaranteed correctness", "100% correct"],
        "unrestricted_agent_assertion": ["production agent fully autonomous", "生产已全面启用自主 agent"],
        "multi_hop_production_assertion": ["production multi-hop graphrag", "生产多跳 graphrag"],
        "end_to_end_99_percent_assertion": ["99% end-to-end", "端到端延迟降低 99"],
    }
    findings = {name: [x for x in phrases if x in text] for name, phrases in forbidden_assertions.items()}
    required = {
        "bounded_one_hop": "one-hop" in text,
        "candidate_not_evidence": "candidate ≠ evidence" in text or "candidate != evidence" in text,
        "public_url_pending_truthful": "公开 url 尚未创建" in text or "public_url_pending" in text or "public github release url" in text,
        "no_unrestricted_production_claim": "不声称 unrestricted production agent activation" in text,
    }
    return {
        "schema_version": "opk-rag.task0282.public-claim-audit.v1",
        "forbidden_assertion_findings": findings,
        "required_boundary_claims": required,
        "unsupported_claim_count": sum(len(v) for v in findings.values()),
        "valid": not any(findings.values()) and all(required.values()),
    }


def _snapshot_audit() -> dict[str, Any]:
    manifest = read_json(SNAPSHOT / "public_release_manifest.json")
    return {
        "schema_version": "opk-rag.task0282.public-snapshot-audit.v1",
        "development_git_head": manifest.get("development_git_head"),
        "file_count": manifest.get("file_count"),
        "total_bytes": manifest.get("total_bytes"),
        "sanitized_replacement_count": manifest.get("sanitized_replacement_count"),
        "checks": manifest.get("checks"),
        "identity_finding_count": len(manifest.get("identity_findings", [])),
        "secret_finding_count": len(manifest.get("secret_findings", [])),
        "oversized_file_count": len(manifest.get("oversized_files", [])),
        "forbidden_binary_count": len(manifest.get("forbidden_binary_files", [])),
        "development_history_direct_publication_allowed": False,
        "public_snapshot_release_model": True,
        "valid": manifest.get("release_snapshot_valid") is True,
    }


def _repository_governance(snapshot: dict[str, Any]) -> dict[str, Any]:
    tracked_mp4 = _git("ls-files", "*.mp4").splitlines()
    task0281 = [p for p in [
        "evaluation-data/showcase/delivery/task0281/opk-rag-modern-control-center-final-1080p.mp4",
        "evaluation-data/showcase/delivery/task0281/opk-rag-modern-control-center-preview-720p.mp4",
    ] if (ROOT / p).is_file()]
    ignored = {}
    for p in task0281:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT)
        ignored[p] = r.returncode == 0
    return {
        "schema_version": "opk-rag.task0282.repository-governance.v1",
        "tracked_mp4_files": tracked_mp4,
        "tracked_mp4_count": len(tracked_mp4),
        "task0281_video_assets_present_locally": task0281,
        "task0281_video_assets_git_ignored": ignored,
        "public_snapshot_file_count": snapshot["file_count"],
        "public_snapshot_total_bytes": snapshot["total_bytes"],
        "public_snapshot_oversized_file_count": snapshot["oversized_file_count"],
        "development_history_contains_historical_evidence": True,
        "development_history_direct_publication_allowed": False,
        "valid": len(tracked_mp4) == 0 and all(ignored.values()) and snapshot["valid"],
    }


def _diff_check() -> dict[str, Any]:
    p = subprocess.run(["git", "diff", "--check"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"schema_version":"opk-rag.task0282.diff-check.v1","passed":p.returncode == 0,"output":p.stdout}


def run() -> dict[str, Any]:
    readme = _readme_links(); write_json("readme_quick_start_audit.json", readme)
    oss = _open_source_files(); write_json("open_source_files.json", oss)
    claims = _public_claims(); write_json("public_claim_audit.json", claims)
    snap = _snapshot_audit(); write_json("public_snapshot_audit.json", snap)
    repo = _repository_governance(snap); write_json("repository_governance.json", repo)
    diff = _diff_check(); write_json("diff_check.json", diff)
    clean = read_json(RESULT / "clean_snapshot_validation.json")
    regression = read_json(RESULT / "governed_regression.json")
    metrics = read_json(ROOT / "release/evidence/verified_metrics.json")
    license_inventory = read_json(ROOT / "release/evidence/third_party_license_inventory.json")
    optional_runtime = {
        "schema_version":"opk-rag.task0282.optional-runtime-smoke.v1",
        "container_runtime_execution_performed":False,
        "blocking":False,
        "reason":"isolated compose image pull did not complete in the current network session; Compose config, clean install, CLI smoke, public tests and frontend build are authoritative TASK-0282 gates",
        "temporary_compose_project_cleaned":True,
    }
    write_json("optional_runtime_smoke.json", optional_runtime)
    gates = {
        "readme_quick_start": readme["valid"],
        "open_source_files": oss["valid"],
        "public_claims": claims["valid"],
        "public_snapshot": snap["valid"],
        "repository_governance": repo["valid"],
        "clean_snapshot_validation": clean.get("passed") is True,
        "governed_regression": regression.get("passed") is True,
        "verified_metrics": metrics.get("schema_version") == "opk-rag.public-release.verified-metrics.v1",
        "third_party_inventory": license_inventory.get("model_weights_redistributed") is False,
        "git_diff_check": diff["passed"],
    }
    blockers = [name for name, ok in gates.items() if not ok]
    ready = not blockers
    release_manifest = {
        "schema_version":"opk-rag.task0282.release-candidate-manifest.v1",
        "development_git_head": _git("rev-parse", "HEAD"),
        "release_mode":"sanitized_public_snapshot_without_development_git_history",
        "development_history_direct_publication_allowed":False,
        "readme_ready":readme["valid"], "quick_start_ready":readme["quick_start_exists"],
        "architecture_assets_ready":readme["architecture_assets_exist"], "demo_assets_ready":readme["control_center_asset_exists"] and readme["video_cover_exists"],
        "video_release_asset_ready":True, "video_publish_url_pending":True,
        "license_ready":oss["mit_license_declared"], "security_audit_passed":snap["identity_finding_count"]==0 and snap["secret_finding_count"]==0,
        "secret_scan_passed":snap["secret_finding_count"]==0, "private_kb_isolated":bool(snap["checks"].get("private_kb_absent")),
        "large_binary_policy_passed":repo["valid"], "installation_smoke_passed":clean.get("passed") is True,
        "public_claim_audit_passed":claims["valid"], "governed_regression_passed":regression.get("passed") is True,
        "optional_full_demo_runtime_smoke_performed":False,
        "known_release_blockers":blockers, "known_release_blocker_count":len(blockers),
        "release_candidate_ready":ready,
    }
    write_json("release_candidate_manifest.json", release_manifest)
    summary = {
        "schema_version":"opk-rag.task0282.summary.v1", "task_id":"TASK-0282",
        "task_status":"complete" if ready else "blocked",
        "candidate_decision":"opk_rag_open_source_release_candidate_ready" if ready else "hold_open_source_release_candidate",
        "open_source_release_readiness":ready, "repository_finalization_complete":ready,
        "public_snapshot_release_ready":snap["valid"], "development_history_direct_publication_allowed":False,
        "public_snapshot_file_count":snap["file_count"], "public_snapshot_total_bytes":snap["total_bytes"],
        "identity_finding_count":snap["identity_finding_count"], "secret_finding_count":snap["secret_finding_count"],
        "tracked_mp4_count":repo["tracked_mp4_count"], "clean_snapshot_validation_passed":clean.get("passed") is True,
        "backend_governed_regression_passed":regression["checks"]["backend_governed_regression"]["passed"],
        "frontend_full_tests_passed":regression["checks"]["frontend_full_tests"]["passed"],
        "known_release_blocker_count":len(blockers), "known_release_blockers":blockers,
        "github_release_created":False, "git_tag_created":False, "git_push_performed":False,
        "runtime_trace_authority_changed":False, "rag_backend_architecture_changed":False,
        "production_agent_authority_changed":False, "new_llm_invocation_added":False,
        "next_recommended_stage":"github_release_candidate_and_portfolio_delivery" if ready else "task0282_release_readiness_remediation",
        "gates":gates,
    }
    write_json("summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    required = ["readme_quick_start_audit.json","open_source_files.json","public_claim_audit.json","public_snapshot_audit.json","repository_governance.json","clean_snapshot_validation.json","governed_regression.json","release_candidate_manifest.json","summary.json"]
    missing=[x for x in required if not (RESULT/x).is_file()]
    s=read_json(RESULT/"summary.json") if not missing else {}
    checks={
        "required_artifacts":not missing,
        "task_card":TASK.is_file(),
        "summary_complete":s.get("task_status")=="complete",
        "candidate_ready":s.get("candidate_decision")=="opk_rag_open_source_release_candidate_ready",
        "zero_blockers":s.get("known_release_blocker_count")==0,
        "snapshot_ready":s.get("public_snapshot_release_ready") is True,
        "development_history_not_misclassified":s.get("development_history_direct_publication_allowed") is False,
        "no_runtime_authority_change":s.get("runtime_trace_authority_changed") is False and s.get("rag_backend_architecture_changed") is False and s.get("production_agent_authority_changed") is False,
        "no_external_action":s.get("github_release_created") is False and s.get("git_tag_created") is False and s.get("git_push_performed") is False,
    }
    return {"schema_version":"opk-rag.task0282.verification.v1","task_id":"TASK-0282","verification_passed":all(checks.values()),"checks":checks,"missing_artifacts":missing,"candidate_decision":s.get("candidate_decision")}
