from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0237"
SCHEMA = "opk-rag.task0237.showcase-video-editing-packaging-and-final-delivery.v1"
SOURCE = ROOT / "evaluation-data/showcase/recording/task0236/opk-rag-showcase-formal-take-02.mp4"
SOURCE_SHA = "4665e5b735d968c715d2c446218e35355736309662a5ec0dc8128871aaca1bec"
EDIT = ROOT / "evaluation-data/showcase/editing/task0237"
FINAL_DIR = ROOT / "evaluation-data/showcase/final-video"
VIDEO = FINAL_DIR / "opk-rag-showcase-final-v1.mp4"
SRT = FINAL_DIR / "opk-rag-showcase-final-v1.zh-CN.srt"
COVER = FINAL_DIR / "opk-rag-showcase-final-v1-cover.png"
FINAL_MANIFEST = FINAL_DIR / "final_video_manifest.json"
RESULT = ROOT / "evaluation-data/results/task0237-showcase-video-editing-packaging-and-final-delivery"
CONTRACT = ROOT / "evaluation-data/contracts/task0237_showcase_video_editing_packaging_and_final_delivery.json"
TASK0236 = ROOT / "evaluation-data/results/task0236-formal-showcase-video-recording/summary.json"
EXPECTED_ORDER = ["S01", "S02", "S03", "S04"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(args: list[str], *, timeout: int = 300) -> dict[str, Any]:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return {"command": " ".join(args), "returncode": p.returncode, "passed": p.returncode == 0, "output_tail": p.stdout[-6000:]}


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "showcase_phase": "video_editing_and_final_delivery",
        "source_take": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA,
        "scenario_order": EXPECTED_ORDER,
        "target": {"container": "mp4", "video_codec": "h264", "width": 1920, "height": 1080, "fps": 30, "pixel_format": "yuv420p", "audio_codec": "aac", "sample_rate": 48000},
        "subtitle_language": "zh-CN",
        "runtime_authority": "opk-rag.runtime-trace.v1",
        "background_music_required": False,
        "runtime_policy_change_allowed": False,
        "git_commit_allowed": False,
    }


def source_authority_validation() -> dict[str, Any]:
    task = load(TASK0236) if TASK0236.is_file() else {}
    actual = sha256(SOURCE) if SOURCE.is_file() else None
    valid = (
        SOURCE.is_file()
        and actual == SOURCE_SHA
        and task.get("task_status") == "complete"
        and task.get("selected_take_valid") is True
        and task.get("formal_showcase_video_recording_ready_for_editing") is True
        and task.get("video_sha256") == SOURCE_SHA
        and task.get("production_runtime_behavior_changed") is False
    )
    return {"source_take_exists": SOURCE.is_file(), "source_take_sha256": actual, "task0236_ready_for_editing": task.get("formal_showcase_video_recording_ready_for_editing") is True, "valid": valid}


def timeline_validation() -> dict[str, Any]:
    source_path = EDIT / "source_timeline.json"
    edit_path = EDIT / "edit_timeline.json"
    if not source_path.is_file() or not edit_path.is_file():
        return {"source_timeline_valid": False, "edit_timeline_valid": False, "valid": False}
    source = load(source_path)
    edit = load(edit_path)
    duration = float(edit.get("final_duration_seconds") or 0.0)
    ids = [seg.get("id") for seg in edit.get("segments", [])]
    required_ids = ["intro", "s01_overview", "s01_normal", "s02_structure", "s03_graph", "s03_candidate", "s04_refusal", "closing"]
    source_valid = source.get("source_sha256") == SOURCE_SHA and any(seg.get("keep") is False and seg.get("id") == "capture_init" for seg in source.get("segments", []))
    edit_valid = (
        edit.get("scenario_order") == EXPECTED_ORDER
        and 150 <= duration <= 210
        and ids == required_ids
        and edit.get("runtime_authority_preserved") is True
        and edit.get("production_runtime_behavior_changed") is False
        and edit.get("background_music") is None
    )
    return {"source_timeline_valid": source_valid, "edit_timeline_valid": edit_valid, "final_duration_target_seconds": duration, "scenario_order_valid": edit.get("scenario_order") == EXPECTED_ORDER, "valid": source_valid and edit_valid}


def _fraction(value: str | None) -> float:
    try:
        return float(Fraction(value or "0"))
    except Exception:  # noqa: BLE001
        return 0.0


def video_validation(write_artifact: bool = True) -> dict[str, Any]:
    if not VIDEO.is_file():
        return {"final_video_file_exists": False, "final_video_technical_validation_passed": False, "valid": False}
    p = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(VIDEO)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    raw = json.loads(p.stdout or "{}") if p.returncode == 0 else {}
    if write_artifact:
        write(EDIT / "final_ffprobe.json", raw)
    streams = raw.get("streams") if isinstance(raw.get("streams"), list) else []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = raw.get("format") if isinstance(raw.get("format"), dict) else {}
    fps = _fraction(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    size = int(fmt.get("size") or VIDEO.stat().st_size)
    digest = sha256(VIDEO)
    technical = (
        p.returncode == 0
        and video.get("codec_name") == "h264"
        and int(video.get("width") or 0) == 1920
        and int(video.get("height") or 0) == 1080
        and video.get("pix_fmt") == "yuv420p"
        and math.isclose(fps, 30.0, abs_tol=0.02)
        and audio.get("codec_name") == "aac"
        and int(audio.get("sample_rate") or 0) == 48000
        and 150 <= duration <= 210
        and size > 100_000
    )
    return {
        "final_video_file_exists": True,
        "final_video_codec": video.get("codec_name"),
        "final_video_width": video.get("width"),
        "final_video_height": video.get("height"),
        "final_video_pixel_format": video.get("pix_fmt"),
        "final_video_fps": round(fps, 6),
        "final_video_duration_seconds": round(duration, 3),
        "final_video_file_size_bytes": size,
        "final_video_sha256": digest,
        "audio_stream_present": bool(audio),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate") or 0) if audio else None,
        "final_video_technical_validation_passed": technical,
        "valid": technical,
    }


SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def _srt_ms(value: str) -> int:
    m = SRT_TIME.fullmatch(value.strip())
    if not m:
        raise ValueError(value)
    h, minute, sec, ms = map(int, m.groups())
    return (((h * 60) + minute) * 60 + sec) * 1000 + ms


def subtitle_validation() -> dict[str, Any]:
    if not SRT.is_file():
        return {"subtitle_present": False, "subtitle_validation_passed": False, "valid": False}
    blocks = [b.strip() for b in SRT.read_text(encoding="utf-8-sig").strip().split("\n\n") if b.strip()]
    ranges: list[tuple[int, int]] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        ranges.append((_srt_ms(start), _srt_ms(end)))
    ordered = bool(ranges) and all(a < b for a, b in ranges) and all(ranges[i][0] >= ranges[i - 1][0] for i in range(1, len(ranges)))
    duration_limit = (video_validation(False).get("final_video_duration_seconds") or 210) * 1000 + 500
    within = all(end <= duration_limit for _, end in ranges)
    valid = len(ranges) >= 8 and ordered and within
    return {"subtitle_present": True, "subtitle_language": "zh-CN", "subtitle_cue_count": len(ranges), "subtitle_timing_ordered": ordered, "subtitle_within_video": within, "subtitle_validation_passed": valid, "valid": valid}


def voice_assets_validation() -> dict[str, Any]:
    path = EDIT / "voice_asset_manifest.json"
    if not path.is_file():
        return {"voice_assets_valid": False, "valid": False}
    data = load(path)
    rows = data.get("segments") or []
    valid = data.get("voice") == "zh-CN-YunxiNeural" and data.get("all_segments_fit_windows") is True and len(rows) == 8 and all((ROOT / row.get("media", "")).is_file() for row in rows)
    return {"voice_provider": data.get("provider"), "voice": data.get("voice"), "voice_segment_count": len(rows), "all_segments_fit_windows": data.get("all_segments_fit_windows"), "voice_assets_valid": valid, "valid": valid}


def audio_validation() -> dict[str, Any]:
    if not VIDEO.is_file():
        return {"final_video_audio_validation_passed": False, "valid": False}
    decode = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(VIDEO), "-map", "0:a:0", "-f", "null", "-"], timeout=180)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(VIDEO), "-map", "0:a:0", "-af", "loudnorm=I=-16:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=180)
    text = p.stdout
    matches = re.findall(r'"input_i"\s*:\s*"?(-?[\d.]+)"?.*?"input_tp"\s*:\s*"?(-?[\d.]+)"?', text, re.S)
    integrated = float(matches[-1][0]) if matches else None
    peak = float(matches[-1][1]) if matches else None
    loudness_ok = integrated is not None and -18.5 <= integrated <= -13.5 and peak is not None and peak <= -0.5
    voice_timing_valid = voice_assets_validation().get("valid", False)
    subtitle_timing_valid = subtitle_validation().get("valid", False)
    clipping = peak is None or peak >= 0.0
    valid = decode["passed"] and p.returncode == 0 and loudness_ok and voice_timing_valid and subtitle_timing_valid and not clipping
    result = {
        "review_method": "AAC full decode + loudnorm measurement + TTS segment window fit + subtitle timing alignment",
        "audio_decode_passed": decode["passed"],
        "integrated_loudness_lufs": integrated,
        "true_peak_dbtp": peak,
        "loudness_target_acceptable": loudness_ok,
        "voice_sync_valid": voice_timing_valid and subtitle_timing_valid,
        "audio_clipping": clipping,
        "synthetic_voice_source": "zh-CN-YunxiNeural",
        "background_music_present": False,
        "final_video_audio_validation_passed": valid,
        "valid": valid,
    }
    write(EDIT / "audio_review.json", result)
    return result


def decode_smoke_validation() -> dict[str, Any]:
    result = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(VIDEO), "-map", "0:v:0", "-f", "null", "-"], timeout=240)
    return {"full_video_decode_smoke_passed": result["passed"], "returncode": result["returncode"], "valid": result["passed"]}


def extract_final_frames() -> dict[str, Any]:
    if not VIDEO.is_file():
        return {"frames_valid": False, "valid": False}
    frame_dir = FINAL_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    points = [
        ("intro", 3.0), ("s01_overview", 16.0), ("s01_normal", 38.0), ("s02_structure", 60.0),
        ("s03_graph", 90.0), ("s03_candidate", 122.0), ("s04_refusal", 150.0), ("closing", 175.0),
    ]
    rows = []
    for idx, (name, ts) in enumerate(points, start=1):
        path = frame_dir / f"{idx:02d}_{name}.png"
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{ts:.3f}", "-i", str(VIDEO), "-frames:v", "1", str(path)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        rows.append({"id": name, "timestamp_seconds": ts, "filename": str(path.relative_to(ROOT)), "extracted": p.returncode == 0 and path.is_file() and path.stat().st_size > 10_000, "size_bytes": path.stat().st_size if path.is_file() else 0})
    valid = len(rows) == 8 and all(row["extracted"] for row in rows)
    payload = {"schema_version": "opk-rag.task0237.final-frame-manifest.v1", "frames": rows, "frames_valid": valid, "valid": valid}
    write(EDIT / "final_frame_manifest.json", payload)
    return payload


def review_validation() -> dict[str, Any]:
    visual_path = EDIT / "visual_review.json"
    content_path = EDIT / "content_accuracy_review.json"
    if not visual_path.is_file() or not content_path.is_file():
        return {"final_video_visual_validation_passed": False, "content_accuracy_validation_passed": False, "sensitive_information_scan_passed": False, "valid": False}
    visual = load(visual_path)
    content = load(content_path)
    visual_keys = ["all_representative_frames_readable", "no_black_or_corrupt_final_scene_frame", "subtitles_readable", "callouts_do_not_block_key_ui", "no_portal_or_obs_overlay", "no_private_content_visible", "no_secret_or_credential_visible", "scenario_visual_order_valid"]
    visual_ok = all(visual.get(k) is True for k in visual_keys)
    content_keys = ["guarded_agent_not_free_planner", "maximum_recovery_attempt_bounded", "graph_not_default_for_every_query", "candidate_not_equal_evidence", "recovered_candidate_reranked", "s04_refused_not_failed"]
    content_ok = all(content.get(k) is True for k in content_keys) and content.get("graph_hop_depth_claim") == 1 and content.get("production_vector_backend") == "qdrant"
    sensitive_ok = visual.get("no_private_content_visible") is True and visual.get("no_secret_or_credential_visible") is True
    return {"final_video_visual_validation_passed": visual_ok, "content_accuracy_validation_passed": content_ok, "sensitive_information_scan_passed": sensitive_ok, "valid": visual_ok and content_ok and sensitive_ok}


def documentation_validation() -> dict[str, Any]:
    report = ROOT / "docs/TASK0237_SHOWCASE_VIDEO_EDITING_PACKAGING_AND_FINAL_DELIVERY_REPORT.md"
    narration = ROOT / "docs/SHOWCASE_FINAL_VIDEO_NARRATION_TASK0237.md"
    readme = ROOT / "README.md"
    project_state = ROOT / "PROJECT_STATE.md"
    changelog = ROOT / "CHANGELOG.md"
    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    narration_text = narration.read_text(encoding="utf-8") if narration.is_file() else ""
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    state_text = project_state.read_text(encoding="utf-8") if project_state.is_file() else ""
    changelog_text = changelog.read_text(encoding="utf-8") if changelog.is_file() else ""
    valid = (
        "final_showcase_video_publishable=true" in report_text
        and "zh-CN-YunxiNeural" in narration_text
        and "Final Showcase Video" in readme_text
        and "Final Showcase Video Delivery: COMPLETED (TASK-0237)" in state_text
        and "TASK-0237 Showcase Video Editing" in changelog_text
    )
    return {
        "task0237_report_ready": report.is_file(),
        "narration_document_ready": narration.is_file(),
        "readme_final_video_link_ready": "Final Showcase Video" in readme_text,
        "project_state_updated": "Final Showcase Video Delivery: COMPLETED (TASK-0237)" in state_text,
        "changelog_updated": "TASK-0237 Showcase Video Editing" in changelog_text,
        "valid": valid,
    }


def production_diff() -> dict[str, Any]:
    # Use frozen historical completed-task production isolation authority instead of the current working tree.
    # This prevents later Agent/Showcase work from being reinterpreted as a mutation made during TASK-0237.
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
    prefixes = ("opk_rag/search", "opk_rag/retrieval", "opk_rag/agent", "opk_rag/graph", "opk_rag/reranking", "opk_rag/evidence", "opk_rag/answer", "opk_rag/generation", "opk_rag/embedding")
    bad = [path for path in paths if path.startswith(prefixes)]
    return {"changed_paths": paths, "forbidden_production_paths": bad, "production_runtime_behavior_changed": bool(bad)}


def manifest_validation() -> dict[str, Any]:
    if not FINAL_MANIFEST.is_file():
        return {"final_video_manifest_valid": False, "valid": False}
    data = load(FINAL_MANIFEST)
    video = video_validation(False)
    valid = (
        data.get("schema_version") == "opk-rag.showcase-final-video-manifest.v1"
        and data.get("source_take_sha256") == SOURCE_SHA
        and data.get("scenario_order") == EXPECTED_ORDER
        and data.get("filename") == str(VIDEO.relative_to(ROOT))
        and data.get("sha256") == video.get("final_video_sha256")
        and data.get("publishable") is True
        and data.get("runtime_authority_preserved") is True
        and data.get("production_runtime_behavior_changed") is False
    )
    return {"final_video_manifest_valid": valid, "publishable": data.get("publishable"), "video_publish_url_pending": data.get("video_publish_url_pending"), "valid": valid}


def write_final_manifest(video: dict[str, Any], subtitle: dict[str, Any], reviews: dict[str, Any], audio: dict[str, Any]) -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=False).stdout.strip()
    payload = {
        "schema_version": "opk-rag.showcase-final-video-manifest.v1",
        "video_version": "opk-rag-showcase-final-v1",
        "source_take": str(SOURCE.relative_to(ROOT)),
        "source_take_sha256": SOURCE_SHA,
        "git_head_before_task0237_commit": head,
        "filename": str(VIDEO.relative_to(ROOT)),
        "codec": video.get("final_video_codec"),
        "width": video.get("final_video_width"),
        "height": video.get("final_video_height"),
        "fps": video.get("final_video_fps"),
        "pixel_format": video.get("final_video_pixel_format"),
        "duration_seconds": video.get("final_video_duration_seconds"),
        "file_size_bytes": video.get("final_video_file_size_bytes"),
        "sha256": video.get("final_video_sha256"),
        "audio_codec": video.get("audio_codec"),
        "audio_stream_present": video.get("audio_stream_present"),
        "audio_sample_rate": video.get("audio_sample_rate"),
        "integrated_loudness_lufs": audio.get("integrated_loudness_lufs"),
        "subtitle_present": subtitle.get("subtitle_present"),
        "subtitle_language": "zh-CN",
        "subtitle_sidecar": str(SRT.relative_to(ROOT)),
        "cover": str(COVER.relative_to(ROOT)),
        "scenario_order": EXPECTED_ORDER,
        "source_runtime_trace_authority": True,
        "runtime_authority_preserved": True,
        "production_runtime_behavior_changed": False,
        "visual_validation_passed": reviews.get("final_video_visual_validation_passed"),
        "content_accuracy_validation_passed": reviews.get("content_accuracy_validation_passed"),
        "sensitive_information_scan_passed": reviews.get("sensitive_information_scan_passed"),
        "publishable": all([video.get("valid"), subtitle.get("valid"), reviews.get("valid"), audio.get("valid")]),
        "video_publish_url_pending": True,
    }
    write(FINAL_MANIFEST, payload)
    return payload


def run_validation(write_artifacts: bool = True) -> dict[str, Any]:
    source = source_authority_validation()
    timeline = timeline_validation()
    voice = voice_assets_validation()
    video = video_validation(write_artifact=write_artifacts)
    subtitle = subtitle_validation()
    audio = audio_validation() if VIDEO.is_file() else {"valid": False, "final_video_audio_validation_passed": False}
    decode = decode_smoke_validation() if VIDEO.is_file() else {"valid": False, "full_video_decode_smoke_passed": False}
    frames = extract_final_frames() if VIDEO.is_file() else {"valid": False, "frames_valid": False}
    reviews = review_validation()
    docs = documentation_validation()
    prod = production_diff()

    if video.get("valid") and subtitle.get("valid") and reviews.get("valid") and audio.get("valid"):
        write_final_manifest(video, subtitle, reviews, audio)
    manifest = manifest_validation()

    commands = {
        "task0237_focused": command(["uv", "run", "pytest", "tests/test_task0237_showcase_video_editing_packaging_and_final_delivery.py", "-q"]),
        "task0236_regression": command(["uv", "run", "pytest", "tests/test_task0236_formal_showcase_video_recording.py", "-q"]),
        "task0235_regression": command(["uv", "run", "pytest", "tests/test_task0235_showcase_end_to_end_dry_run_and_recording_readiness.py", "-q"]),
        "task0234_regression": command(["uv", "run", "pytest", "tests/test_task0234_showcase_v2_end_to_end_presentation_polish_and_recording_freeze.py", "-q"]),
        "git_diff_check": command(["git", "diff", "--check"]),
    }
    command_ok = all(v["passed"] for v in commands.values())

    blockers = []
    checks = [
        ("source_take_invalid", source.get("valid")),
        ("edit_timeline_invalid", timeline.get("valid")),
        ("voice_assets_invalid", voice.get("valid")),
        ("final_video_technical_invalid", video.get("valid")),
        ("audio_invalid", audio.get("valid")),
        ("subtitle_invalid", subtitle.get("valid")),
        ("decode_smoke_failed", decode.get("valid")),
        ("representative_frames_invalid", frames.get("valid")),
        ("visual_or_content_review_invalid", reviews.get("valid")),
        ("documentation_invalid", docs.get("valid")),
        ("final_manifest_invalid", manifest.get("valid")),
    ]
    for name, passed in checks:
        if not passed:
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
        "showcase_phase": "video_editing_and_final_delivery",
        "source_take": str(SOURCE.relative_to(ROOT)),
        "source_take_sha256": source.get("source_take_sha256"),
        "source_take_valid": source.get("valid", False),
        "edit_timeline_valid": timeline.get("edit_timeline_valid", False),
        "final_video_file": str(VIDEO.relative_to(ROOT)),
        **{k: video.get(k) for k in ("final_video_file_exists", "final_video_codec", "final_video_width", "final_video_height", "final_video_pixel_format", "final_video_fps", "final_video_duration_seconds", "final_video_file_size_bytes", "final_video_sha256", "audio_stream_present", "audio_codec", "audio_sample_rate", "final_video_technical_validation_passed")},
        "final_video_visual_validation_passed": reviews.get("final_video_visual_validation_passed", False),
        "final_video_audio_validation_passed": audio.get("final_video_audio_validation_passed", False),
        "audio_integrity_passed": audio.get("valid", False),
        "integrated_loudness_lufs": audio.get("integrated_loudness_lufs"),
        "subtitle_present": subtitle.get("subtitle_present", False),
        "subtitle_language": subtitle.get("subtitle_language"),
        "subtitle_validation_passed": subtitle.get("subtitle_validation_passed", False),
        "scenario_order_valid": timeline.get("scenario_order_valid", False),
        "runtime_authority_preserved": timeline.get("valid", False) and reviews.get("content_accuracy_validation_passed", False),
        "visual_integrity_passed": reviews.get("final_video_visual_validation_passed", False),
        "content_accuracy_validation_passed": reviews.get("content_accuracy_validation_passed", False),
        "sensitive_information_scan_passed": reviews.get("sensitive_information_scan_passed", False),
        "full_video_decode_smoke_passed": decode.get("full_video_decode_smoke_passed", False),
        "final_video_manifest_valid": manifest.get("final_video_manifest_valid", False),
        "documentation_ready": docs.get("valid", False),
        "final_delivery_package_ready": manifest.get("valid", False) and docs.get("valid", False) and COVER.is_file() and SRT.is_file(),
        "video_publish_url_pending": True,
        "production_runtime_behavior_changed": prod["production_runtime_behavior_changed"],
        "known_final_delivery_blocker_count": len(blockers),
        "final_delivery_blockers": blockers,
        "final_showcase_video_ready": ready,
        "final_showcase_video_publishable": ready,
        "project_showcase_video_delivery": "complete" if ready else "incomplete",
        "next_recommended_stage": "project_open_source_release_governance" if ready else "task0237_minimal_delivery_fix",
        **{f"{name}_passed": value["passed"] for name, value in commands.items()},
    }
    if write_artifacts:
        write(CONTRACT, contract())
        RESULT.mkdir(parents=True, exist_ok=True)
        for name, payload in (("source_authority_validation.json", source), ("timeline_validation.json", timeline), ("voice_assets_validation.json", voice), ("video_validation.json", video), ("audio_validation.json", audio), ("subtitle_validation.json", subtitle), ("decode_smoke_validation.json", decode), ("frame_validation.json", frames), ("review_validation.json", reviews), ("documentation_validation.json", docs), ("production_path_diff_audit.json", prod), ("final_manifest_validation.json", manifest)):
            write(RESULT / name, payload)
        for name, payload in commands.items():
            write(RESULT / f"{name}.json", payload)
        write(RESULT / "summary.json", summary)
        write(RESULT / "verification.json", {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": ready, "summary": summary})
    return summary


def verify() -> dict[str, Any]:
    return run_validation(True)
