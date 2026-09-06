from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evaluation-data/results/task0281-modern-control-center-video-editing-packaging-and-final-delivery"
DELIVERY = ROOT / "evaluation-data/showcase/delivery/task0281"
PREV = ROOT / "evaluation-data/results/task0280-formal-modern-control-center-video-recording"
TASK = ROOT / "tasks/TASK-0281_modern_control_center_video_editing_packaging_and_final_delivery.md"
REPORT = ROOT / "docs/TASK0281_MODERN_CONTROL_CENTER_VIDEO_EDITING_PACKAGING_AND_FINAL_DELIVERY_REPORT.md"
AUTHORITY = ROOT / "docs/CURRENT_PROJECT_AUTHORITY.md"
MANIFEST = DELIVERY / "delivery_manifest.json"
EXPECTED_SOURCE_SHA = "1c379fdb3c54e54ea6d5e375d6b89a0ba00807a01b849b3d4677972534b8505a"
EXPECTED_SCENES = ["intro", "s01", "s02", "s03", "s03_graph", "s03_lineage", "s04", "close"]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def entry() -> dict[str, Any]:
    summary = read(PREV / "summary.json")
    verification = read(PREV / "verification.json")
    gates = {
        "task0280_complete": summary.get("task_status") == "complete",
        "task0280_verified": verification.get("verification_passed") is True,
        "formal_video_recorded": summary.get("candidate_decision") == "formal_modern_control_center_video_recorded",
        "next_decision_matches": summary.get("next_decision") == "advance_to_modern_video_editing_packaging_and_final_delivery",
        "runtime_trace_unchanged": summary.get("runtime_trace_authority_changed") is False,
        "rag_backend_unchanged": summary.get("rag_backend_architecture_changed") is False,
        "production_agent_unchanged": summary.get("production_agent_authority_changed") is False,
    }
    return {"schema_version": "opk-rag.task0281.entry-authority.v1", "gates": gates, "entry_gate_passed": all(gates.values())}


def _tracked(relative_path: str) -> bool:
    p = subprocess.run(["git", "ls-files", "--error-unmatch", relative_path], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return p.returncode == 0


def validate_manifest() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read(MANIFEST)
    source = manifest.get("source", {})
    variants = {x.get("role"): x for x in manifest.get("variants", [])}
    final = variants.get("final_1080p", {})
    preview = variants.get("preview_720p", {})
    q = manifest.get("quality", {})
    editing = manifest.get("editing", {})
    runtime = manifest.get("runtime_integrity", {})
    repo = manifest.get("repository_governance", {})

    video_checks = {
        "source_sha_matches_task0280": source.get("sha256") == EXPECTED_SOURCE_SHA,
        "trim_is_exactly_half_second": editing.get("leading_black_trim_seconds") == 0.5,
        "final_h264": final.get("codec") == "h264",
        "final_1080p": (final.get("width"), final.get("height")) == (1920, 1080),
        "final_30fps": final.get("fps") == "30/1",
        "final_yuv420p": final.get("pix_fmt") == "yuv420p",
        "final_no_audio": final.get("audio_stream_count") == 0,
        "final_faststart": final.get("faststart") is True,
        "final_decode_clean": final.get("decode_error_bytes") == 0,
        "final_no_black_interval": final.get("black_interval_count") == 0,
        "preview_h264": preview.get("codec") == "h264",
        "preview_720p": (preview.get("width"), preview.get("height")) == (1280, 720),
        "preview_30fps": preview.get("fps") == "30/1",
        "preview_no_audio": preview.get("audio_stream_count") == 0,
        "preview_faststart": preview.get("faststart") is True,
        "preview_decode_clean": preview.get("decode_error_bytes") == 0,
        "preview_no_black_interval": preview.get("black_interval_count") == 0,
        "final_ssim_at_least_099": float(q.get("final_1080p_ssim_vs_aligned_source") or 0) >= 0.99,
        "manifest_delivery_ready": manifest.get("delivery_ready") is True,
    }
    video = {
        "schema_version": "opk-rag.task0281.video-validation.v1",
        "checks": video_checks,
        "final_1080p": final,
        "preview_720p": preview,
        "quality": q,
        "video_validation_passed": all(video_checks.values()),
    }

    frames = manifest.get("representative_frames", [])
    frame_checks = {
        "eight_frames": len(frames) == 8,
        "scene_order_covered": [x.get("scene") for x in frames] == EXPECTED_SCENES,
        "all_1920x1080": all((x.get("width"), x.get("height")) == (1920, 1080) for x in frames),
        "all_nonblank": all(x.get("nonblank") is True for x in frames),
        "all_unique": len({x.get("sha256") for x in frames}) == 8,
        "cover_present": (DELIVERY / "opk-rag-modern-control-center-cover.jpg").is_file(),
    }
    frame = {"schema_version": "opk-rag.task0281.frame-validation.v1", "checks": frame_checks, "frames": frames, "frame_validation_passed": all(frame_checks.values())}

    final_rel = str(final.get("path") or "")
    preview_rel = str(preview.get("path") or "")
    repo_checks = {
        "binary_policy_git_ignored_release_asset": repo.get("video_binary_policy") == "git_ignored_release_asset",
        "final_git_ignored": repo.get("final_video_git_ignored") is True,
        "preview_git_ignored": repo.get("preview_video_git_ignored") is True,
        "final_not_tracked": bool(final_rel) and not _tracked(final_rel),
        "preview_not_tracked": bool(preview_rel) and not _tracked(preview_rel),
        "publish_url_pending": repo.get("video_publish_url_pending") is True,
        "raw_desktop_inventory_not_persisted": repo.get("raw_desktop_inventory_persisted") is False,
        "private_runtime_material_not_persisted": repo.get("private_runtime_material_persisted") is False,
    }
    repository = {"schema_version": "opk-rag.task0281.repository-governance.v1", "checks": repo_checks, "repository_governance_passed": all(repo_checks.values())}

    content_checks = {
        "scene_order_unchanged": editing.get("scene_order") == EXPECTED_SCENES and editing.get("scene_order_changed") is False,
        "semantic_execution_retry_zero": editing.get("semantic_execution_retry_count") == 0,
        "preferred_answer_retry_zero": editing.get("preferred_answer_retry_count") == 0,
        "runtime_not_reexecuted": editing.get("runtime_reexecution_performed") is False,
        "ui_decision_authority_false": runtime.get("ui_decision_authority") is False,
        "runtime_trace_unchanged": runtime.get("runtime_trace_authority_changed") is False,
        "rag_backend_unchanged": runtime.get("rag_backend_architecture_changed") is False,
        "production_agent_unchanged": runtime.get("production_agent_authority_changed") is False,
        "new_llm_invocation_false": runtime.get("new_llm_invocation_added") is False,
        "graph_max_hop_one": runtime.get("graph_max_hop") == 1,
        "recovery_max_attempts_one": runtime.get("recovery_max_attempts") == 1,
    }
    content = {"schema_version": "opk-rag.task0281.content-integrity.v1", "checks": content_checks, "content_integrity_passed": all(content_checks.values())}
    return video, frame, repository, content


def run() -> dict[str, Any]:
    e = entry()
    write("entry_authority.json", e)
    video, frame, repository, content = validate_manifest()
    write("video_validation.json", video)
    write("frame_validation.json", frame)
    write("repository_governance.json", repository)
    write("content_integrity.json", content)
    gates = {
        "entry": e["entry_gate_passed"],
        "video": video["video_validation_passed"],
        "frames": frame["frame_validation_passed"],
        "repository": repository["repository_governance_passed"],
        "content": content["content_integrity_passed"],
    }
    complete = all(gates.values())
    m = read(MANIFEST)
    variants = {x["role"]: x for x in m["variants"]}
    summary = {
        "schema_version": "opk-rag.task0281.summary.v1",
        "task_id": "TASK-0281",
        "task_status": "complete" if complete else "partial",
        "candidate_decision": "modern_control_center_video_packaged_and_delivery_ready" if complete else "hold_for_video_delivery_validation",
        "next_recommended_stage": "project_open_source_release_readiness" if complete else None,
        "final_video_delivery_ready": complete,
        "final_1080p_path": variants["final_1080p"]["path"],
        "final_1080p_sha256": variants["final_1080p"]["sha256"],
        "final_1080p_size_bytes": variants["final_1080p"]["size_bytes"],
        "preview_720p_path": variants["preview_720p"]["path"],
        "preview_720p_sha256": variants["preview_720p"]["sha256"],
        "preview_720p_size_bytes": variants["preview_720p"]["size_bytes"],
        "duration_seconds": variants["final_1080p"]["duration_seconds"],
        "final_1080p_ssim_vs_aligned_source": m["quality"]["final_1080p_ssim_vs_aligned_source"],
        "leading_black_trim_seconds": m["editing"]["leading_black_trim_seconds"],
        "video_publish_url_pending": True,
        "audio_stream_count": 0,
        "screen_only_delivery": True,
        "known_delivery_blocker_count": 0 if complete else sum(not x for x in gates.values()),
        "runtime_trace_authority_changed": False,
        "rag_backend_architecture_changed": False,
        "production_agent_authority_changed": False,
        "new_llm_invocation_added": False,
        "gates": gates,
    }
    write("summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    needed = ["entry_authority.json", "video_validation.json", "frame_validation.json", "repository_governance.json", "content_integrity.json", "summary.json"]
    missing = [x for x in needed if not (RESULT / x).is_file()]
    summary = read(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    checks = {
        "required_artifacts": not missing,
        "task_exists": TASK.is_file(),
        "report_exists": REPORT.is_file(),
        "authority_exists": AUTHORITY.is_file(),
        "delivery_readme_exists": (DELIVERY / "README.md").is_file(),
        "manifest_exists": MANIFEST.is_file(),
        "task_complete": summary.get("task_status") == "complete",
        "final_decision": summary.get("candidate_decision") == "modern_control_center_video_packaged_and_delivery_ready",
        "next_stage": summary.get("next_recommended_stage") == "project_open_source_release_readiness",
        "zero_blockers": summary.get("known_delivery_blocker_count") == 0,
        "runtime_trace_unchanged": summary.get("runtime_trace_authority_changed") is False,
        "rag_backend_unchanged": summary.get("rag_backend_architecture_changed") is False,
        "production_agent_unchanged": summary.get("production_agent_authority_changed") is False,
    }
    return {
        "schema_version": "opk-rag.task0281.verification.v1",
        "task_id": "TASK-0281",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
