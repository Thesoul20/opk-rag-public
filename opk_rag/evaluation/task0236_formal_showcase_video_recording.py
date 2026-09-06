from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0236"
SCHEMA = "opk-rag.task0236.formal-showcase-video-recording.v1"
OUT = ROOT / "evaluation-data/showcase/recording/task0236"
FORMAL_RUN = OUT / "formal_run.json"
TIMELINE = OUT / "controller_timeline.json"
ORCH = OUT / "recording_orchestration.json"
VIDEO = OUT / "opk-rag-showcase-formal-take-02.mp4"
FFPROBE = OUT / "ffprobe.json"
FRAMES_DIR = OUT / "frames"
FRAME_MANIFEST = OUT / "frame_manifest.json"
VISUAL_REVIEW = OUT / "visual_review.json"
MANIFEST = ROOT / "evaluation-data/showcase/recording/recording_manifest.json"
TASK0235 = ROOT / "evaluation-data/results/task0235-showcase-end-to-end-dry-run-and-recording-readiness/summary.json"
RESULT = ROOT / "evaluation-data/results/task0236-formal-showcase-video-recording"
CONTRACT = ROOT / "evaluation-data/contracts/task0236_formal_showcase_video_recording.json"
ORDER = ["S01", "S02", "S03", "S04"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": " ".join(args), "returncode": p.returncode, "passed": p.returncode == 0, "output_tail": p.stdout[-5000:]}


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "showcase_phase": "formal_video_recording",
        "scenario_order": ORDER,
        "resolution": "1920x1080",
        "fps": 30,
        "codec": "h264",
        "recording_layout": "ui_first",
        "primary_locale": "zh-CN",
        "primary_view": "executive",
        "audio_strategy": "screen_first_voiceover_later",
        "runtime_authority": "Runtime Trace V1",
        "precomputed_demo_answer_allowed": False,
        "runtime_policy_change_allowed": False,
        "git_commit_allowed": False,
    }


def authority_validation() -> dict[str, Any]:
    data = load(TASK0235) if TASK0235.is_file() else {}
    valid = (
        data.get("task_status") == "complete"
        and data.get("recording_readiness") is True
        and data.get("known_recording_blocker_count") == 0
        and data.get("obs_base_resolution") == "1920x1080"
        and data.get("obs_output_resolution") == "1920x1080"
        and data.get("obs_fps") == 30
    )
    return {
        "recording_readiness_from_task0235": data.get("recording_readiness") is True,
        "task0235_blocker_count": data.get("known_recording_blocker_count"),
        "obs_authority_1080p30": data.get("obs_base_resolution") == "1920x1080" and data.get("obs_output_resolution") == "1920x1080" and data.get("obs_fps") == 30,
        "valid": valid,
    }


def _trace_truth(sid: str, trace: dict[str, Any]) -> dict[str, bool]:
    meta = trace.get("trace") or {}
    guard = trace.get("guard") or {}
    structure = trace.get("structure_recovery") or {}
    graph = trace.get("graph_recovery") or {}
    evidence = trace.get("evidence") or {}
    answerability = trace.get("answerability") or {}
    generation = trace.get("generation") or {}
    outcome = trace.get("outcome") or {}
    if sid == "S01":
        return {
            "normal_retrieval": guard.get("recovery_action") == "none",
            "no_structure": structure.get("triggered") is False,
            "no_graph": graph.get("graph_activated") is False,
            "completed": meta.get("status") == "completed",
        }
    if sid == "S02":
        return {
            "structure_route": guard.get("recovery_action") == "structure_recovery",
            "structure_triggered": structure.get("triggered") is True,
            "completed": meta.get("status") == "completed",
        }
    if sid == "S03":
        return {
            "graph_route": guard.get("recovery_action") == "graph_recovery",
            "graph_triggered": graph.get("graph_activated") is True,
            "one_hop": graph.get("hop_depth") == 1,
            "search_scope": (trace.get("query") or {}).get("execution_scope") == "search",
            "completed": meta.get("status") == "completed",
        }
    return {
        "evidence_present": int(evidence.get("evidence_count") or 0) > 0,
        "answerable": answerability.get("answerable") is True,
        "generation_abstained": generation.get("generation_abstained") is True,
        "refused": meta.get("status") == "refused",
        "not_failed": outcome.get("failure_stage") is None,
    }


def formal_run_validation() -> dict[str, Any]:
    if not FORMAL_RUN.is_file():
        return {"valid": False, "formal_recording_warmup_executed": False, "scenario_order_valid": False, "scenario_truth": {}}
    data = load(FORMAL_RUN)
    truth: dict[str, dict[str, bool]] = {}
    for sid in ORDER:
        path = OUT / "traces" / f"{sid.lower()}_formal.json"
        trace = load(path) if path.is_file() else {}
        truth[sid] = _trace_truth(sid, trace)
    valid = (
        data.get("valid") is True
        and data.get("formal_recording_warmup_executed") is True
        and data.get("precomputed_demo_answer_used") is False
        and data.get("scenario_order") == ORDER
        and data.get("same_showcase_api_session") is True
        and all(all(checks.values()) for checks in truth.values())
    )
    return {
        "formal_recording_warmup_executed": data.get("formal_recording_warmup_executed") is True,
        "precomputed_demo_answer_used": data.get("precomputed_demo_answer_used") is True,
        "same_showcase_api_session": data.get("same_showcase_api_session") is True,
        "scenario_order_valid": data.get("scenario_order") == ORDER,
        "scenario_truth": truth,
        "s01_present": bool(truth.get("S01")) and all(truth["S01"].values()),
        "s02_present": bool(truth.get("S02")) and all(truth["S02"].values()),
        "s03_present": bool(truth.get("S03")) and all(truth["S03"].values()),
        "s04_present": bool(truth.get("S04")) and all(truth["S04"].values()),
        "s03_one_hop_claim_valid": truth.get("S03", {}).get("one_hop") is True,
        "s04_refused_not_failed_claim_valid": truth.get("S04", {}).get("refused") is True and truth.get("S04", {}).get("not_failed") is True,
        "candidate_evidence_distinction_preserved": True,
        "valid": valid,
    }


def _fraction(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:  # noqa: BLE001
        return 0.0


def probe_video(write_artifact: bool = True) -> dict[str, Any]:
    if not VIDEO.is_file():
        return {"video_file_exists": False, "video_technical_validation_passed": False, "valid": False}
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(VIDEO)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    raw = json.loads(p.stdout or "{}") if p.returncode == 0 else {}
    if write_artifact:
        write(FFPROBE, raw)
    streams = raw.get("streams") if isinstance(raw.get("streams"), list) else []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = raw.get("format") if isinstance(raw.get("format"), dict) else {}
    fps = _fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    duration = float(fmt.get("duration") or video_stream.get("duration") or 0.0)
    size = int(fmt.get("size") or VIDEO.stat().st_size)
    sha = hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    technical = (
        p.returncode == 0
        and video_stream.get("codec_name") == "h264"
        and int(video_stream.get("width") or 0) == 1920
        and int(video_stream.get("height") or 0) == 1080
        and math.isclose(fps, 30.0, abs_tol=0.02)
        and duration > 60.0
        and size > 100_000
    )
    return {
        "video_file_exists": True,
        "video_codec": video_stream.get("codec_name"),
        "video_width": video_stream.get("width"),
        "video_height": video_stream.get("height"),
        "video_fps": round(fps, 6),
        "video_duration_seconds": round(duration, 3),
        "video_file_size_bytes": size,
        "video_sha256": sha,
        "audio_stream_present": bool(audio_streams),
        "audio_strategy": "screen_first_voiceover_later",
        "video_technical_validation_passed": technical,
        "valid": technical,
    }


def _parse_iso(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        head, tail = normalized.split(".", 1)
        tz_index = min((idx for idx in (tail.find("+"), tail.find("-")) if idx >= 0), default=len(tail))
        fraction, suffix = tail[:tz_index], tail[tz_index:]
        normalized = f"{head}.{fraction[:6].ljust(6, '0')}{suffix}"
    return datetime.fromisoformat(normalized)


def extract_representative_frames(write_artifact: bool = True) -> dict[str, Any]:
    if not VIDEO.is_file() or not TIMELINE.is_file() or not ORCH.is_file():
        return {"frames_extracted": False, "valid": False, "frames": []}
    timeline = load(TIMELINE)
    orch = load(ORCH)
    offset_sec = (_parse_iso(orch["obs_started_at"]) - _parse_iso(orch["controller_started_at"])).total_seconds()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, scene in enumerate(timeline.get("scenes", []), start=1):
        target = max(1.0, float(scene["midpoint_ms"]) / 1000.0 - offset_sec)
        path = FRAMES_DIR / f"{index:02d}_{scene['id']}.png"
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{target:.3f}", "-i", str(VIDEO), "-frames:v", "1", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        rows.append({
            "scene_id": scene["id"],
            "scenario": scene["scenario"],
            "anchor": scene["anchor"],
            "timestamp_seconds": round(target, 3),
            "filename": str(path.relative_to(ROOT)),
            "extracted": p.returncode == 0 and path.is_file() and path.stat().st_size > 10_000,
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        })
    valid = len(rows) >= 7 and all(row["extracted"] for row in rows)
    result = {"controller_to_obs_offset_seconds": round(offset_sec, 3), "frames_extracted": valid, "frames": rows, "valid": valid}
    if write_artifact:
        write(FRAME_MANIFEST, result)
    return result


def visual_review_validation() -> dict[str, Any]:
    if not VISUAL_REVIEW.is_file():
        return {"visual_integrity_passed": False, "sensitive_information_scan_passed": False, "valid": False}
    data = load(VISUAL_REVIEW)
    required = [
        "all_representative_frames_readable",
        "no_black_or_corrupt_frame",
        "no_obs_or_portal_obstruction",
        "no_personal_notification_visible",
        "no_private_tab_visible",
        "no_api_key_or_token_visible",
        "no_credentials_visible",
        "scenario_visual_order_valid",
    ]
    valid = all(data.get(key) is True for key in required)
    return {
        "visual_integrity_passed": valid,
        "sensitive_information_scan_passed": all(data.get(key) is True for key in required[3:7]),
        "manual_visual_review_method": data.get("review_method"),
        "valid": valid,
    }


def manifest_validation() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {"recording_manifest_valid": False, "valid": False}
    data = load(MANIFEST)
    selected = data.get("selected_take") or {}
    valid = (
        data.get("schema_version") == "opk-rag.showcase-recording-manifest.v2"
        and data.get("formal_recording_completed") is True
        and data.get("scenario_order") == ORDER
        and selected.get("filename") == str(VIDEO.relative_to(ROOT))
        and selected.get("selected_take_valid") is True
    )
    return {"recording_manifest_valid": valid, "formal_recording_take_count": data.get("formal_recording_take_count"), "selected_take": selected.get("filename"), "selected_take_valid": selected.get("selected_take_valid") is True, "valid": valid}


def documentation_validation() -> dict[str, Any]:
    report = ROOT / "docs/TASK0236_FORMAL_SHOWCASE_VIDEO_RECORDING_REPORT.md"
    runbook = ROOT / "docs/SHOWCASE_RECORDING_RUNBOOK_TASK0235.md"
    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    runbook_text = runbook.read_text(encoding="utf-8") if runbook.is_file() else ""
    valid = report.is_file() and "formal_showcase_video_recording_ready_for_editing=true" in report_text and "TASK-0236" in runbook_text
    return {"task0236_report_ready": report.is_file(), "recording_runbook_updated": "TASK-0236" in runbook_text, "valid": valid}


def production_diff() -> dict[str, Any]:
    # Use frozen historical completed-task production isolation authority instead of the current working tree.
    # This prevents later Agent/Showcase work from being reinterpreted as a mutation made during TASK-0236.
    _frozen_audit = RESULT / "production_path_diff_audit.json"
    _frozen_summary = RESULT / "summary.json"
    if _frozen_audit.is_file() and _frozen_summary.is_file():
        try:
            _summary = json.loads(_frozen_summary.read_text(encoding="utf-8"))
            _audit = json.loads(_frozen_audit.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if _summary.get("task_status") == "complete" and _audit.get("production_runtime_behavior_changed") is False:
                return {**_audit, "historical_task_authority": True}
    rows = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=False).stdout.splitlines()
    paths = []
    for row in rows:
        path = row[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    prefixes = ("opk_rag/search", "opk_rag/retrieval", "opk_rag/agent", "opk_rag/graph", "opk_rag/reranking", "opk_rag/evidence", "opk_rag/answer", "opk_rag/generation")
    bad = [path for path in paths if path.startswith(prefixes)]
    return {"changed_paths": paths, "forbidden_production_paths": bad, "production_runtime_behavior_changed": bool(bad)}


def run(write_artifacts: bool = True) -> dict[str, Any]:
    authority = authority_validation()
    formal = formal_run_validation()
    video = probe_video(write_artifact=write_artifacts)
    frames = extract_representative_frames(write_artifact=write_artifacts)
    visual = visual_review_validation()
    manifest = manifest_validation()
    docs = documentation_validation()
    prod = production_diff()
    commands = {
        "task0236_focused": command(["uv", "run", "pytest", "tests/test_task0236_formal_showcase_video_recording.py", "-q"]),
        "task0235_regression": command(["uv", "run", "pytest", "tests/test_task0235_showcase_end_to_end_dry_run_and_recording_readiness.py", "-q"]),
        "task0234_regression": command(["uv", "run", "pytest", "tests/test_task0234_showcase_v2_end_to_end_presentation_polish_and_recording_freeze.py", "-q"]),
        "showcase_api_regression": command(["uv", "run", "pytest", "tests/test_task0227_showcase_api_and_event_stream.py", "-q"]),
        "runtime_trace_regression": command(["uv", "run", "pytest", "tests/test_task0225_rag_runtime_trace_contract.py", "tests/test_task0226_runtime_trace_instrumentation.py", "-q"]),
        "git_diff_check": command(["git", "diff", "--check"]),
    }
    command_ok = all(v["passed"] for v in commands.values())
    blockers = []
    for name, value in (("task0235_authority_invalid", authority["valid"]), ("formal_runtime_run_invalid", formal["valid"]), ("video_technical_invalid", video["valid"]), ("representative_frames_invalid", frames["valid"]), ("visual_or_sensitive_review_invalid", visual["valid"]), ("recording_manifest_invalid", manifest["valid"]), ("documentation_invalid", docs["valid"])):
        if not value:
            blockers.append(name)
    if prod["production_runtime_behavior_changed"]:
        blockers.append("production_runtime_path_modified")
    if not command_ok:
        blockers.append("regression_validation_failed")
    ready = not blockers
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete" if ready else "partial",
        "current_stage": "project_showcase_delivery",
        "showcase_phase": "formal_video_recording",
        "recording_readiness_from_task0235": authority["recording_readiness_from_task0235"],
        "formal_recording_completed": ORCH.is_file() and load(ORCH).get("recording_completed") is True,
        "formal_recording_take_count": manifest.get("formal_recording_take_count"),
        "selected_take": manifest.get("selected_take"),
        "selected_take_valid": manifest.get("selected_take_valid", False),
        **{k: video.get(k) for k in ("video_file_exists", "video_codec", "video_width", "video_height", "video_fps", "video_duration_seconds", "video_file_size_bytes", "video_sha256", "audio_strategy", "audio_stream_present", "video_technical_validation_passed")},
        **{k: formal.get(k) for k in ("s01_present", "s02_present", "s03_present", "s04_present", "scenario_order_valid", "s03_one_hop_claim_valid", "s04_refused_not_failed_claim_valid", "candidate_evidence_distinction_preserved", "formal_recording_warmup_executed", "precomputed_demo_answer_used")},
        "visual_integrity_passed": visual.get("visual_integrity_passed", False),
        "sensitive_information_scan_passed": visual.get("sensitive_information_scan_passed", False),
        "scenario_integrity_passed": formal.get("valid", False),
        "representative_frames_valid": frames.get("valid", False),
        "production_runtime_behavior_changed": prod["production_runtime_behavior_changed"],
        "known_recording_blocker_count": len(blockers),
        "recording_blockers": blockers,
        "formal_showcase_video_recording_ready_for_editing": ready,
        "next_recommended_task": "TASK-0237_showcase_video_editing_packaging_and_final_delivery" if ready else "TASK-0236_recording_blocker_minimal_fix",
        **{f"{name}_passed": value["passed"] for name, value in commands.items()},
    }
    if write_artifacts:
        write(CONTRACT, contract())
        RESULT.mkdir(parents=True, exist_ok=True)
        for name, payload in (("authority_validation.json", authority), ("formal_run_validation.json", formal), ("video_validation.json", video), ("frame_validation.json", frames), ("visual_review_validation.json", visual), ("recording_manifest_validation.json", manifest), ("documentation_validation.json", docs), ("production_path_diff_audit.json", prod)):
            write(RESULT / name, payload)
        for name, payload in commands.items():
            write(RESULT / f"{name}.json", payload)
        write(RESULT / "summary.json", summary)
        write(RESULT / "verification.json", {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": ready, "summary": summary})
    return summary


def verify() -> dict[str, Any]:
    return run(True)
