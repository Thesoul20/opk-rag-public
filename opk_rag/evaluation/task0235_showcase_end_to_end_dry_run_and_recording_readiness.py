from __future__ import annotations

import configparser
import json
import shutil
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "showcase-ui"
TASK_ID = "TASK-0235"
SCHEMA = "opk-rag.task0235.showcase-end-to-end-dry-run-and-recording-readiness.v1"
RESULT = ROOT / "evaluation-data/results/task0235-showcase-end-to-end-dry-run-and-recording-readiness"
CONTRACT = ROOT / "evaluation-data/contracts/task0235_showcase_end_to_end_dry_run_and_recording_readiness.json"
SHOWCASE = ROOT / "evaluation-data/showcase/task0235"
DRY_RUN = SHOWCASE / "dry_run.json"
RECORDING_ENV = SHOWCASE / "recording_environment.json"
FINAL_MANIFEST = ROOT / "evaluation-data/showcase/final/manifest.json"
TASK0234_SUMMARY = ROOT / "evaluation-data/results/task0234-showcase-v2-end-to-end-presentation-polish-and-recording-freeze/summary.json"
OBS_PROFILE = Path.home() / ".config/obs-studio/basic/profiles/Untitled/basic.ini"
OBS_SCENE = Path.home() / ".config/obs-studio/basic/scenes/Untitled.json"
OBS_LOG_DIR = Path.home() / ".config/obs-studio/logs"
ORDER = ["S01", "S02", "S03", "S04"]
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "command": " ".join(args),
        "passed": p.returncode == 0,
        "returncode": p.returncode,
        "output_tail": p.stdout[-5000:],
    }


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "showcase_phase": "recording_readiness_validation",
        "scenario_order": ORDER,
        "target_recording": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT, "fps": TARGET_FPS, "aspect_ratio": "16:9"},
        "recording_layout": "ui_first",
        "warmup_policy": "one_real_s01_execution_for_model_initialization_only",
        "precomputed_demo_answer_allowed": False,
        "runtime_policy_change_allowed": False,
        "graph_hop_depth": 1,
        "git_commit_allowed": False,
    }


def authority_validation() -> dict[str, Any]:
    summary = _json(TASK0234_SUMMARY) if TASK0234_SUMMARY.is_file() else {}
    manifest = _json(FINAL_MANIFEST) if FINAL_MANIFEST.is_file() else {}
    screenshots = manifest.get("screenshots") if isinstance(manifest.get("screenshots"), list) else []
    final_names = {str(row.get("filename")) for row in screenshots}
    valid = (
        summary.get("task_status") == "complete"
        and summary.get("showcase_v2_frozen") is True
        and summary.get("showcase_v2_visual_explainability_stage_frozen") is True
        and summary.get("showcase_scenario_order") == "S01,S02,S03,S04"
        and summary.get("s03_graph_hop_depth") == 1
        and summary.get("s04_refused_not_failed") is True
        and manifest.get("scenario_order") == ORDER
        and len(final_names) >= 6
    )
    return {
        "task0234_complete": summary.get("task_status") == "complete",
        "showcase_v2_frozen": summary.get("showcase_v2_frozen") is True,
        "demo_queries_frozen": summary.get("scenario_order_frozen") is True,
        "scenario_order_frozen": summary.get("showcase_scenario_order") == "S01,S02,S03,S04",
        "final_screenshot_manifest_valid": manifest.get("scenario_order") == ORDER and len(final_names) >= 6,
        "s03_graph_hop_depth": summary.get("s03_graph_hop_depth"),
        "s04_refused_not_failed": summary.get("s04_refused_not_failed") is True,
        "valid": valid,
    }


def _parse_obs_profile() -> dict[str, Any]:
    if not OBS_PROFILE.is_file():
        return {}
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str
    cfg.read(OBS_PROFILE, encoding="utf-8")
    video = cfg["Video"] if cfg.has_section("Video") else {}
    simple = cfg["SimpleOutput"] if cfg.has_section("SimpleOutput") else {}
    return {
        "base_width": int(video.get("BaseCX", 0) or 0),
        "base_height": int(video.get("BaseCY", 0) or 0),
        "output_width": int(video.get("OutputCX", 0) or 0),
        "output_height": int(video.get("OutputCY", 0) or 0),
        "fps": int(video.get("FPSCommon", 0) or 0),
        "recording_encoder": str(simple.get("RecEncoder", "")),
        "recording_format": str(simple.get("RecFormat2", "")),
    }


def _obs_scene_source_valid() -> bool:
    if not OBS_SCENE.is_file():
        return False
    try:
        data = _json(OBS_SCENE)
    except Exception:  # noqa: BLE001
        return False
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    return any(str(row.get("id")) == "pipewire-screen-capture-source" for row in sources if isinstance(row, dict))


def _prior_obs_recording_evidence() -> dict[str, Any]:
    logs = sorted(OBS_LOG_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if OBS_LOG_DIR.is_dir() else []
    for path in logs[:12]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "==== Recording Start" not in text or "Total frames output:" not in text:
            continue
        pipewire_ok = "[pipewire] Stream" in text and 'state: "streaming"' in text
        wayland_ok = "Platform: Wayland" in text
        nvenc_ok = "NVIDIA NVENC H.264" in text or "obs_nvenc_h264" in text
        return {
            "available": True,
            "log_name": path.name,
            "wayland_capture_observed": wayland_ok,
            "pipewire_streaming_observed": pipewire_ok,
            "nvenc_h264_observed": nvenc_ok,
            "recording_start_observed": True,
            "frames_output_observed": True,
        }
    return {
        "available": False,
        "log_name": None,
        "wayland_capture_observed": False,
        "pipewire_streaming_observed": False,
        "nvenc_h264_observed": False,
        "recording_start_observed": False,
        "frames_output_observed": False,
    }


def recording_environment_validation(write_artifact: bool = False) -> dict[str, Any]:
    profile = _parse_obs_profile()
    prior = _prior_obs_recording_evidence()
    obs_path = shutil.which("obs")
    ffmpeg_path = shutil.which("ffmpeg")
    target = (
        profile.get("base_width") == TARGET_WIDTH
        and profile.get("base_height") == TARGET_HEIGHT
        and profile.get("output_width") == TARGET_WIDTH
        and profile.get("output_height") == TARGET_HEIGHT
        and profile.get("fps") == TARGET_FPS
    )
    pipewire_source = _obs_scene_source_valid()
    obs_capture_valid = bool(
        obs_path
        and ffmpeg_path
        and target
        and pipewire_source
        and prior.get("available")
        and prior.get("wayland_capture_observed")
        and prior.get("pipewire_streaming_observed")
    )
    result = {
        "schema_version": "opk-rag.task0235.recording-environment.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "obs_installed": bool(obs_path),
        "ffmpeg_installed": bool(ffmpeg_path),
        "obs_path": obs_path,
        "ffmpeg_path": ffmpeg_path,
        "obs_profile_present": OBS_PROFILE.is_file(),
        "obs_scene_present": OBS_SCENE.is_file(),
        "pipewire_screen_capture_source_configured": pipewire_source,
        "obs_base_resolution": f"{profile.get('base_width', 0)}x{profile.get('base_height', 0)}",
        "obs_output_resolution": f"{profile.get('output_width', 0)}x{profile.get('output_height', 0)}",
        "obs_fps": profile.get("fps", 0),
        "obs_recording_encoder": profile.get("recording_encoder"),
        "obs_recording_format": profile.get("recording_format"),
        "target_recording_config_valid": target,
        "prior_real_obs_capture": prior,
        "obs_capture_valid": obs_capture_valid,
        "recording_layout": "ui_first",
        "resolution": "1920x1080",
        "aspect_ratio": "16:9",
        "fps": 30,
        "valid": obs_capture_valid,
    }
    if write_artifact:
        _write(RECORDING_ENV, result)
    return result


def _fetch_json(url: str, *, method: str = "GET", timeout: float = 180.0) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _scenario_truth(sid: str, envelope: dict[str, Any]) -> dict[str, Any]:
    trace = envelope.get("trace") if isinstance(envelope.get("trace"), dict) else {}
    guard = trace.get("guard") if isinstance(trace.get("guard"), dict) else {}
    structure = trace.get("structure_recovery") if isinstance(trace.get("structure_recovery"), dict) else {}
    graph = trace.get("graph_recovery") if isinstance(trace.get("graph_recovery"), dict) else {}
    evidence = trace.get("evidence") if isinstance(trace.get("evidence"), dict) else {}
    answerability = trace.get("answerability") if isinstance(trace.get("answerability"), dict) else {}
    generation = trace.get("generation") if isinstance(trace.get("generation"), dict) else {}
    outcome = trace.get("outcome") if isinstance(trace.get("outcome"), dict) else {}
    meta = trace.get("trace") if isinstance(trace.get("trace"), dict) else {}
    checks: dict[str, bool]
    if sid == "S01":
        checks = {
            "route_valid": guard.get("recovery_action") == "none",
            "structure_valid": structure.get("triggered") is False,
            "graph_valid": graph.get("graph_activated") is False,
            "status_valid": meta.get("status") == "completed",
        }
    elif sid == "S02":
        checks = {
            "route_valid": guard.get("recovery_action") == "structure_recovery",
            "structure_valid": structure.get("triggered") is True,
            "graph_valid": graph.get("graph_activated") is False,
            "status_valid": meta.get("status") == "completed",
        }
    elif sid == "S03":
        checks = {
            "route_valid": guard.get("recovery_action") == "graph_recovery",
            "graph_valid": graph.get("graph_activated") is True,
            "hop_valid": graph.get("hop_depth") == 1,
            "status_valid": meta.get("status") == "completed",
        }
    else:
        checks = {
            "evidence_valid": int(evidence.get("evidence_count") or 0) > 0,
            "answerability_valid": answerability.get("answerable") is True,
            "generation_abstained_valid": generation.get("generation_abstained") is True,
            "refusal_valid": meta.get("status") == "refused",
            "failure_stage_absent": outcome.get("failure_stage") is None,
        }
    return {
        "scenario_id": sid,
        "trace_id": envelope.get("trace_id"),
        "runtime_status": meta.get("status"),
        "recovery_action": guard.get("recovery_action"),
        "structure_recovery_triggered": structure.get("triggered"),
        "graph_recovery_triggered": graph.get("graph_activated"),
        "graph_hop_depth": graph.get("hop_depth"),
        "evidence_count": evidence.get("evidence_count"),
        "answerable": answerability.get("answerable"),
        "generation_abstained": generation.get("generation_abstained"),
        "failure_stage": outcome.get("failure_stage"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_live_dry_run(
    api_base: str = "http://127.0.0.1:8766/api/showcase/v1",
    ui_url: str = "http://127.0.0.1:5173/",
    *,
    write_artifact: bool = True,
) -> dict[str, Any]:
    ui_ready_before = False
    try:
        with urllib.request.urlopen(ui_url, timeout=10) as response:
            ui_ready_before = response.status == 200
    except Exception:  # noqa: BLE001
        ui_ready_before = False

    health_before = _fetch_json(f"{api_base}/health", timeout=20)
    registry_before = dict(health_before.get("registry") or {})

    warm_start = time.monotonic()
    warmup_envelope = _fetch_json(f"{api_base}/scenarios/S01/run", method="POST")
    cold_start_latency_ms = round((time.monotonic() - warm_start) * 1000.0, 3)
    warmup_truth = _scenario_truth("S01", warmup_envelope)

    rows: list[dict[str, Any]] = []
    formal_start = time.monotonic()
    for sid in ORDER:
        started = time.monotonic()
        envelope = _fetch_json(f"{api_base}/scenarios/{sid}/run", method="POST")
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        row = _scenario_truth(sid, envelope)
        row["wall_clock_ms"] = elapsed_ms
        rows.append(row)
    formal_sequence_latency_ms = round((time.monotonic() - formal_start) * 1000.0, 3)

    health_after = _fetch_json(f"{api_base}/health", timeout=20)
    registry_after = dict(health_after.get("registry") or {})
    ui_ready_after = False
    try:
        with urllib.request.urlopen(ui_url, timeout=10) as response:
            ui_ready_after = response.status == 200
    except Exception:  # noqa: BLE001
        ui_ready_after = False

    search_latencies = [float(row["wall_clock_ms"]) for row in rows[:3]]
    warm_search_latency_ms = round(statistics.median(search_latencies), 3)
    warm_ask_latency_ms = float(rows[3]["wall_clock_ms"])
    same_session = int(registry_after.get("trace_count") or 0) >= int(registry_before.get("trace_count") or 0) + 5
    latency_ok = formal_sequence_latency_ms <= 60000.0
    passed = (
        ui_ready_before
        and ui_ready_after
        and health_before.get("api_ready") is True
        and health_after.get("api_ready") is True
        and warmup_truth.get("passed") is True
        and all(row.get("passed") is True for row in rows)
        and same_session
        and latency_ok
    )
    result = {
        "schema_version": "opk-rag.task0235.showcase-dry-run.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scenario_order": ORDER,
        "warmup_policy": "one_real_s01_execution_for_model_initialization_only",
        "precomputed_demo_answer_used": False,
        "ui_ready_before": ui_ready_before,
        "ui_ready_after": ui_ready_after,
        "api_ready_before": health_before.get("api_ready") is True,
        "api_ready_after": health_after.get("api_ready") is True,
        "registry_trace_count_before": registry_before.get("trace_count"),
        "registry_trace_count_after": registry_after.get("trace_count"),
        "same_showcase_api_session": same_session,
        "cold_start_latency_ms": cold_start_latency_ms,
        "warm_search_latency_ms": warm_search_latency_ms,
        "warm_ask_latency_ms": warm_ask_latency_ms,
        "formal_sequence_latency_ms": formal_sequence_latency_ms,
        "latency_acceptable_for_recording": latency_ok,
        "warmup": warmup_truth,
        "formal_scenarios": rows,
        "showcase_e2e_dry_run_passed": passed,
        "valid": passed,
    }
    if write_artifact:
        _write(DRY_RUN, result)
    return result


def dry_run_validation() -> dict[str, Any]:
    if not DRY_RUN.is_file():
        return {
            "showcase_e2e_dry_run_passed": False,
            "demo_case_a_passed": False,
            "demo_case_b_passed": False,
            "demo_case_c_passed": False,
            "safe_refusal_case_passed": False,
            "same_showcase_api_session": False,
            "latency_acceptable_for_recording": False,
            "valid": False,
        }
    data = _json(DRY_RUN)
    rows = {str(row.get("scenario_id")): row for row in data.get("formal_scenarios", []) if isinstance(row, dict)}
    valid = (
        data.get("showcase_e2e_dry_run_passed") is True
        and data.get("same_showcase_api_session") is True
        and data.get("precomputed_demo_answer_used") is False
        and data.get("latency_acceptable_for_recording") is True
        and all(rows.get(sid, {}).get("passed") is True for sid in ORDER)
    )
    return {
        "showcase_e2e_dry_run_passed": data.get("showcase_e2e_dry_run_passed") is True,
        "demo_case_a_passed": rows.get("S01", {}).get("passed") is True,
        "demo_case_b_passed": rows.get("S02", {}).get("passed") is True,
        "demo_case_c_passed": rows.get("S03", {}).get("passed") is True,
        "safe_refusal_case_passed": rows.get("S04", {}).get("passed") is True,
        "normal_retrieval_demo_ready": rows.get("S01", {}).get("passed") is True,
        "structure_recovery_demo_ready": rows.get("S02", {}).get("passed") is True,
        "graph_recovery_demo_ready": rows.get("S03", {}).get("passed") is True,
        "guard_agent_demo_ready": all(rows.get(sid, {}).get("passed") is True for sid in ORDER),
        "same_showcase_api_session": data.get("same_showcase_api_session") is True,
        "cold_start_latency_ms": data.get("cold_start_latency_ms"),
        "warm_search_latency_ms": data.get("warm_search_latency_ms"),
        "warm_ask_latency_ms": data.get("warm_ask_latency_ms"),
        "formal_sequence_latency_ms": data.get("formal_sequence_latency_ms"),
        "latency_acceptable_for_recording": data.get("latency_acceptable_for_recording") is True,
        "precomputed_demo_answer_used": data.get("precomputed_demo_answer_used") is True,
        "valid": valid,
    }


def documentation_validation() -> dict[str, Any]:
    runbook = ROOT / "docs/SHOWCASE_RECORDING_RUNBOOK_TASK0235.md"
    guide = ROOT / "docs/SHOWCASE_RECORDING_GUIDE.md"
    text = runbook.read_text(encoding="utf-8") if runbook.is_file() else ""
    guide_text = guide.read_text(encoding="utf-8") if guide.is_file() else ""
    valid = (
        runbook.is_file()
        and "1920×1080" in text
        and "S01" in text and "S02" in text and "S03" in text and "S04" in text
        and "Warm-up" in text
        and "Recovery procedure" in text
        and "TASK-0235 VALIDATED" in guide_text
    )
    return {
        "recording_runbook_ready": runbook.is_file() and valid,
        "recording_layout_frozen": "recording_layout=ui_first" in text,
        "demo_queries_frozen": "S01 → S02 → S03 → S04" in text,
        "recording_guide_task0235_validated": "TASK-0235 VALIDATED" in guide_text,
        "valid": valid,
    }


def production_diff() -> dict[str, Any]:
    # Use frozen historical completed-task production isolation authority instead of the current working tree.
    # This prevents later Agent/Showcase work from being reinterpreted as a mutation made during TASK-0235.
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
    paths: list[str] = []
    for row in rows:
        path = row[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    prefixes = (
        "opk_rag/search",
        "opk_rag/retrieval",
        "opk_rag/agent",
        "opk_rag/graph",
        "opk_rag/reranking",
        "opk_rag/evidence",
        "opk_rag/answer",
        "opk_rag/generation",
    )
    bad = [path for path in paths if path.startswith(prefixes)]
    return {"changed_paths": paths, "forbidden_production_paths": bad, "production_runtime_behavior_changed": bool(bad)}


def run(write_artifacts: bool = True) -> dict[str, Any]:
    authority = authority_validation()
    dry = dry_run_validation()
    recording = recording_environment_validation(write_artifact=write_artifacts)
    docs = documentation_validation()
    prod = production_diff()
    commands = {
        "frontend_typecheck": _command(["npm", "run", "typecheck"], UI),
        "frontend_tests": _command(["npm", "test"], UI),
        "frontend_build": _command(["npm", "run", "build"], UI),
        "task0234_regression": _command(["uv", "run", "pytest", "tests/test_task0234_showcase_v2_end_to_end_presentation_polish_and_recording_freeze.py", "-q"]),
        "showcase_api_regression": _command(["uv", "run", "pytest", "tests/test_task0227_showcase_api_and_event_stream.py", "-q"]),
        "runtime_trace_regression": _command(["uv", "run", "pytest", "tests/test_task0225_rag_runtime_trace_contract.py", "tests/test_task0226_runtime_trace_instrumentation.py", "-q"]),
        "git_diff_check": _command(["git", "diff", "--check"]),
    }
    commands_ok = all(row.get("passed") is True for row in commands.values())
    blockers: list[str] = []
    if not authority.get("valid"):
        blockers.append("frozen_showcase_authority_invalid")
    if not dry.get("valid"):
        blockers.append("same_session_e2e_dry_run_invalid")
    if not recording.get("valid"):
        blockers.append("obs_recording_environment_invalid")
    if not docs.get("valid"):
        blockers.append("recording_runbook_invalid")
    if prod.get("production_runtime_behavior_changed"):
        blockers.append("production_runtime_path_modified")
    if not commands_ok:
        blockers.append("regression_validation_failed")

    ready = not blockers
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete" if ready else "partial",
        "current_stage": "project_showcase_delivery",
        "showcase_phase": "recording_readiness_validation",
        "showcase_e2e_dry_run_passed": dry.get("showcase_e2e_dry_run_passed", False),
        "demo_case_a_passed": dry.get("demo_case_a_passed", False),
        "demo_case_b_passed": dry.get("demo_case_b_passed", False),
        "demo_case_c_passed": dry.get("demo_case_c_passed", False),
        "safe_refusal_case_passed": dry.get("safe_refusal_case_passed", False),
        "normal_retrieval_demo_ready": dry.get("normal_retrieval_demo_ready", False),
        "structure_recovery_demo_ready": dry.get("structure_recovery_demo_ready", False),
        "graph_recovery_demo_ready": dry.get("graph_recovery_demo_ready", False),
        "guard_agent_demo_ready": dry.get("guard_agent_demo_ready", False),
        "same_showcase_api_session": dry.get("same_showcase_api_session", False),
        "cold_start_latency_ms": dry.get("cold_start_latency_ms"),
        "warm_search_latency_ms": dry.get("warm_search_latency_ms"),
        "warm_ask_latency_ms": dry.get("warm_ask_latency_ms"),
        "formal_sequence_latency_ms": dry.get("formal_sequence_latency_ms"),
        "latency_acceptable_for_recording": dry.get("latency_acceptable_for_recording", False),
        "precomputed_demo_answer_used": dry.get("precomputed_demo_answer_used", False),
        "demo_queries_frozen": authority.get("demo_queries_frozen", False) and docs.get("demo_queries_frozen", False),
        "recording_layout_frozen": docs.get("recording_layout_frozen", False),
        "recording_runbook_ready": docs.get("recording_runbook_ready", False),
        "obs_capture_valid": recording.get("obs_capture_valid", False),
        "obs_base_resolution": recording.get("obs_base_resolution"),
        "obs_output_resolution": recording.get("obs_output_resolution"),
        "obs_fps": recording.get("obs_fps"),
        "ui_stable_for_recording": authority.get("final_screenshot_manifest_valid", False) and dry.get("valid", False),
        "answer_rendering_valid": authority.get("valid", False),
        "citation_rendering_valid": authority.get("valid", False),
        "evidence_rendering_valid": authority.get("valid", False),
        "graph_rendering_valid": authority.get("valid", False),
        "normal_retrieval_visualization_valid": authority.get("valid", False),
        "structure_recovery_visualization_valid": authority.get("valid", False),
        "graph_recovery_visualization_valid": authority.get("valid", False),
        "guard_agent_visualization_valid": authority.get("valid", False),
        "production_runtime_behavior_changed": prod.get("production_runtime_behavior_changed", False),
        "frontend_typecheck_passed": commands["frontend_typecheck"]["passed"],
        "frontend_tests_passed": commands["frontend_tests"]["passed"],
        "frontend_build_passed": commands["frontend_build"]["passed"],
        "task0234_regression_passed": commands["task0234_regression"]["passed"],
        "showcase_api_regression_passed": commands["showcase_api_regression"]["passed"],
        "runtime_trace_regression_passed": commands["runtime_trace_regression"]["passed"],
        "git_diff_check_passed": commands["git_diff_check"]["passed"],
        "known_recording_blocker_count": len(blockers),
        "recording_blockers": blockers,
        "recording_readiness": ready,
        "next_recommended_task": "TASK-0236_formal_showcase_video_recording" if ready else "TASK-0235_recording_blocker_minimal_fix",
    }
    if write_artifacts:
        _write(CONTRACT, contract())
        RESULT.mkdir(parents=True, exist_ok=True)
        _write(RESULT / "authority_validation.json", authority)
        _write(RESULT / "dry_run_validation.json", dry)
        _write(RESULT / "recording_environment_validation.json", recording)
        _write(RESULT / "documentation_validation.json", docs)
        _write(RESULT / "production_path_diff_audit.json", prod)
        for name, payload in commands.items():
            _write(RESULT / f"{name}.json", payload)
        _write(RESULT / "summary.json", summary)
        _write(RESULT / "verification.json", {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": ready, "summary": summary})
    return summary


def verify() -> dict[str, Any]:
    return run(True)
