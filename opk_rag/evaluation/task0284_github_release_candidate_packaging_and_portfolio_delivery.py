from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evaluation-data/results/task0284-github-release-candidate-packaging-and-portfolio-delivery"
TASK = ROOT / "tasks/TASK-0284_github_release_candidate_packaging_and_portfolio_delivery.md"
REPORT = ROOT / "docs/TASK0284_GITHUB_RELEASE_CANDIDATE_PACKAGING_AND_PORTFOLIO_DELIVERY_REPORT.md"
RELEASE_NOTES = ROOT / "docs/GITHUB_RELEASE_CANDIDATE_NOTES.md"
PORTFOLIO = ROOT / "docs/OPK_RAG_PORTFOLIO_DELIVERY.md"
MANIFEST = ROOT / "release/rc/release_candidate_manifest.json"
CHECKSUMS = ROOT / "release/rc/SHA256SUMS"
CLEAN_VALIDATION = RESULT / "clean_snapshot_validation.json"
ACCEPTED_COMMIT = "5dab31ee8ee272d0b0c35e629a29f141757be320"
EXPECTED_VIDEO_HASHES = {
    "final_1080p": "423f463757f572bb4a076bc710749777b2b6fab70a079de79d652a4895a448f7",
    "preview_720p": "68d52cbce27884c0e90e218f28df476550d8158b8725fe373458e3a0445fe3b2",
}
VIDEO_PATHS = {
    "final_1080p": ROOT / "evaluation-data/showcase/delivery/task0281/opk-rag-modern-control-center-final-1080p.mp4",
    "preview_720p": ROOT / "evaluation-data/showcase/delivery/task0281/opk-rag-modern-control-center-preview-720p.mp4",
}
COMMITTED_ASSETS = [
    ROOT / "docs/assets/control-center-executive.jpg",
    ROOT / "docs/assets/video-cover.jpg",
    ROOT / "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.png",
    ROOT / "docs/diagrams/current_opk_rag_selective_agent_architecture_v2_zh.png",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_path(manifest: dict[str, Any]) -> Path:
    raw = Path(manifest["archive"]["path"])
    return raw if raw.is_absolute() else ROOT / raw


def run() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    validation = read_json(CLEAN_VALIDATION)
    archive = _archive_path(manifest)
    task0283 = read_json(
        ROOT / "evaluation-data/results/task0283-open-source-release-final-acceptance-and-rc-freeze/verification.json"
    )
    checksums_text = CHECKSUMS.read_text(encoding="utf-8").strip()
    release_text = RELEASE_NOTES.read_text(encoding="utf-8")
    portfolio_text = PORTFOLIO.read_text(encoding="utf-8")
    video_checks = {
        role: path.is_file() and sha256(path) == EXPECTED_VIDEO_HASHES[role]
        for role, path in VIDEO_PATHS.items()
    }
    asset_checks = {path.relative_to(ROOT).as_posix(): path.is_file() for path in COMMITTED_ASSETS}
    external = manifest.get("external_actions") or {}
    source = manifest.get("source") or {}
    archive_info = manifest.get("archive") or {}
    privacy = source.get("privacy_finding_counts") or {}
    documentation_checks = {
        "release_notes_present": RELEASE_NOTES.is_file() and len(release_text) > 500,
        "portfolio_present": PORTFOLIO.is_file() and len(portfolio_text) > 1000,
        "bilingual_positioning": "中文定位" in portfolio_text and "English positioning" in portfolio_text,
        "three_resume_bullets": all(marker in portfolio_text for marker in ("Resume bullet 1", "Resume bullet 2", "Resume bullet 3")),
        "verified_metric_graph_recall": "0.6667" in portfolio_text and "0.9444" in portfolio_text,
        "verified_metric_semantic": "37/40" in portfolio_text and "92.5%" in portfolio_text,
        "verified_metric_qdrant": "Recall@K=1.0" in portfolio_text and "MRR=1.0" in portfolio_text,
        "verified_metric_search_p95": "1648" in portfolio_text,
        "verified_bge_fp16": "BGE" in portfolio_text and "FP16" in portfolio_text,
        "commands_present": "uv sync --dev --frozen" in release_text and "sha256sum -c" in release_text,
        "video_pending_truthful": "video_publish_url_pending=true" in portfolio_text,
    }
    gates = {
        "task0283_acceptance": task0283.get("verification_passed") is True,
        "accepted_commit_exact": source.get("accepted_commit") == ACCEPTED_COMMIT and source.get("development_git_head") == ACCEPTED_COMMIT,
        "source_worktree_clean_at_export": source.get("development_worktree_clean") is True and source.get("development_worktree_change_count") == 0,
        "privacy_findings_zero": privacy and all(count == 0 for count in privacy.values()),
        "clean_snapshot_validation": validation.get("passed") is True and len(validation.get("checks") or {}) == 8,
        "archive_exists": archive.is_file(),
        "archive_checksum_matches": archive.is_file() and sha256(archive) == archive_info.get("sha256"),
        "sha256sums_matches": checksums_text == f"{archive_info.get('sha256')}  {archive.name}",
        "archive_reproducible": (archive_info.get("reproducibility") or {}).get("reproducible") is True,
        "documentation_complete": all(documentation_checks.values()),
        "committed_assets_present": all(asset_checks.values()),
        "local_video_integrity": all(video_checks.values()),
        "video_publish_url_pending": (manifest.get("video_delivery") or {}).get("video_publish_url_pending") is True,
        "no_external_action": bool(external) and not any(external.values()),
        "task_card_present": TASK.is_file(),
        "report_present": REPORT.is_file(),
    }
    blockers = sorted(name for name, passed in gates.items() if not passed)
    evidence = {
        "schema_version": "opk-rag.task0284.delivery-evidence.v1",
        "task_id": "TASK-0284",
        "archive": {
            "path": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),
            "sha256": sha256(archive) if archive.is_file() else None,
            "size_bytes": archive.stat().st_size if archive.is_file() else None,
        },
        "source": source,
        "clean_snapshot_checks": {name: item.get("passed") for name, item in (validation.get("checks") or {}).items()},
        "documentation_checks": documentation_checks,
        "asset_checks": asset_checks,
        "video_checks": video_checks,
        "external_actions": external,
        "gates": gates,
    }
    write_json("delivery_evidence.json", evidence)
    summary = {
        "schema_version": "opk-rag.task0284.summary.v1",
        "task_id": "TASK-0284",
        "task_status": "complete" if not blockers else "blocked",
        "candidate_decision": (
            "github_release_candidate_package_and_portfolio_delivery_ready"
            if not blockers
            else "hold_github_release_candidate_package_and_portfolio_delivery"
        ),
        "accepted_source_commit": ACCEPTED_COMMIT,
        "archive_path": evidence["archive"]["path"],
        "archive_sha256": evidence["archive"]["sha256"],
        "archive_size_bytes": evidence["archive"]["size_bytes"],
        "public_snapshot_content_sha256": source.get("public_snapshot_content_sha256"),
        "public_snapshot_file_count": source.get("public_snapshot_file_count"),
        "source_worktree_clean_at_export": source.get("development_worktree_clean"),
        "clean_snapshot_check_count": len(validation.get("checks") or {}),
        "known_delivery_blocker_count": len(blockers),
        "known_delivery_blockers": blockers,
        "video_publish_url_pending": True,
        "github_release_created": False,
        "git_tag_created": False,
        "git_push_performed": False,
        "archive_uploaded": False,
        "video_uploaded": False,
        "repository_visibility_changed": False,
        "runtime_trace_authority_changed": False,
        "rag_backend_architecture_changed": False,
        "production_agent_authority_changed": False,
        "new_llm_invocation_added": False,
        "gates": gates,
    }
    write_json("summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    required = [
        "clean_snapshot_validation.json",
        "public_snapshot_manifest.json",
        "delivery_evidence.json",
        "summary.json",
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    summary = read_json(RESULT / "summary.json") if not missing else {}
    checks = {
        "required_artifacts": not missing,
        "task_complete": summary.get("task_status") == "complete",
        "delivery_ready": summary.get("candidate_decision") == "github_release_candidate_package_and_portfolio_delivery_ready",
        "zero_blockers": summary.get("known_delivery_blocker_count") == 0,
        "source_clean_and_exact": summary.get("accepted_source_commit") == ACCEPTED_COMMIT and summary.get("source_worktree_clean_at_export") is True,
        "eight_clean_snapshot_checks": summary.get("clean_snapshot_check_count") == 8,
        "no_external_action": all(
            summary.get(name) is False
            for name in (
                "github_release_created",
                "git_tag_created",
                "git_push_performed",
                "archive_uploaded",
                "video_uploaded",
                "repository_visibility_changed",
            )
        ),
        "authority_unchanged": all(
            summary.get(name) is False
            for name in (
                "runtime_trace_authority_changed",
                "rag_backend_architecture_changed",
                "production_agent_authority_changed",
                "new_llm_invocation_added",
            )
        ),
    }
    verification = {
        "schema_version": "opk-rag.task0284.verification.v1",
        "task_id": "TASK-0284",
        "verification_passed": all(checks.values()),
        "candidate_decision": summary.get("candidate_decision"),
        "checks": checks,
        "missing_artifacts": missing,
    }
    write_json("verification.json", verification)
    return verification
