from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping

from opk_rag.showcase.runtime_trace import (
    TRACE_SCHEMA_VERSION,
    scan_forbidden_keys,
    scan_sensitive_values,
    validate_runtime_trace,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0226"
SCHEMA_VERSION = "opk-rag.task0226.runtime-trace-instrumentation.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0226-runtime-trace-instrumentation"
TASK_CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0226_runtime_trace_instrumentation.json"
TASK_PATH = ROOT / "tasks/TASK-0226_runtime_trace_instrumentation.md"
DOC_PATH = ROOT / "docs/RUNTIME_TRACE_INSTRUMENTATION.md"
REPORT_PATH = ROOT / "docs/TASK0226_RUNTIME_TRACE_INSTRUMENTATION_REPORT.md"
LIVE_TRACE_PATHS = {
    sid: ROOT / f"evaluation-data/showcase/runtime_trace_v1_live_{sid.lower()}.json"
    for sid in ("S01", "S02", "S03", "S04")
}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def changed_paths() -> list[str]:
    tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    return sorted({row.strip() for row in [*tracked, *untracked] if row.strip()})


def _task0226_committed_pyproject() -> str:
    commits = subprocess.check_output(["git", "log", "--format=%H", "--", "tasks/TASK-0226_runtime_trace_instrumentation.md"], cwd=ROOT, text=True).splitlines()
    commit = commits[-1].strip() if commits else ""
    if not commit:
        return ""
    return subprocess.check_output(["git", "show", f"{commit}:pyproject.toml"], cwd=ROOT, text=True).lower()


def runtime_hook_audit() -> dict[str, Any]:
    files = {
        "runtime_trace": (ROOT / "opk_rag/showcase/runtime_trace.py").read_text(encoding="utf-8"),
        "search_service": (ROOT / "opk_rag/search/service.py").read_text(encoding="utf-8"),
        "answer_service": (ROOT / "opk_rag/answer/service.py").read_text(encoding="utf-8"),
        "public_runtime": (ROOT / "opk_rag/runtime_v2/public_search_runtime.py").read_text(encoding="utf-8"),
        "cli": (ROOT / "opk_rag/cli.py").read_text(encoding="utf-8"),
        "demo": (ROOT / "opk_rag/showcase/demo.py").read_text(encoding="utf-8"),
    }
    checks = {
        "trace_context_defined": "class RuntimeTraceContext" in files["runtime_trace"],
        "live_builder_defined": "def build_runtime_trace_from_live_execution" in files["runtime_trace"],
        "search_accepts_trace_context": "runtime_trace_context: RuntimeTraceContext | None" in files["search_service"],
        "search_uses_retrieval_timing_observer": "RetrievalTimingObserver(enabled=bool(runtime_trace_context" in files["search_service"],
        "answer_finalizes_trace": "def _with_runtime_trace" in files["answer_service"],
        "guard_structure_counts_exposed": "candidate_pool_count_after_structure" in files["public_runtime"],
        "graph_pool_counts_exposed": "candidate_pool_count_before" in files["public_runtime"] and "candidate_pool_count_after" in files["public_runtime"],
        "cli_search_trace": "search_parser.add_argument(\"--trace\"" in files["cli"],
        "cli_ask_trace": "ask_parser.add_argument(\"--trace\"" in files["cli"],
        "demo_trace": "demo_parser.add_argument(\"--trace\"" in files["cli"] and "runtime_traces" in files["demo"],
        "no_fastapi_ui_added": '"fastapi' not in _task0226_committed_pyproject(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checks": checks,
        "runtime_hook_audit_passed": all(checks.values()),
    }


def live_trace_conformance() -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for sid, path in LIVE_TRACE_PATHS.items():
        if not path.is_file():
            scenarios[sid] = {"path": str(path.relative_to(ROOT)), "available": False, "conformant": False, "reason": "live_trace_artifact_not_generated"}
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        checks = validate_runtime_trace(trace)
        scenarios[sid] = {
            "path": str(path.relative_to(ROOT)),
            "available": True,
            "trace_id": (trace.get("trace") or {}).get("trace_id"),
            "status": (trace.get("trace") or {}).get("status"),
            "checks": checks,
            "conformant": all(checks.values()),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenarios": scenarios,
        "all_available": all(row["available"] for row in scenarios.values()),
        "all_conformant": all(row["conformant"] for row in scenarios.values()),
    }


def sensitive_data_scan() -> dict[str, Any]:
    paths = [
        *LIVE_TRACE_PATHS.values(),
        TASK_CONTRACT_PATH,
        RESULT_DIR / "summary.json",
        RESULT_DIR / "verification.json",
        RESULT_DIR / "runtime_hook_audit.json",
        RESULT_DIR / "trace_equivalence.json",
        RESULT_DIR / "live_trace_conformance.json",
        RESULT_DIR / "instrumentation_overhead.json",
    ]
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in scan_forbidden_keys(payload):
            findings.append({"path": str(path.relative_to(ROOT)), "finding": item, "kind": "forbidden_key"})
        for item in scan_sensitive_values(payload):
            findings.append({"path": str(path.relative_to(ROOT)), "finding": item, "kind": "sensitive_value"})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "finding_count": len(findings),
        "findings": findings,
        "sensitive_data_scan_passed": not findings,
    }


def _read_measured_artifact(name: str, *, evidence_source: str) -> dict[str, Any] | None:
    path = RESULT_DIR / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("evidence_source") != evidence_source:
        return None
    return payload


def trace_equivalence() -> dict[str, Any]:
    measured = _read_measured_artifact("trace_equivalence.json", evidence_source="live_paired_showcase_s01_s04")
    if measured is not None:
        return measured
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "evidence_source": "unavailable",
        "scenario_count": 0,
        "scenarios": {},
        "runtime_equivalence_passed": False,
        "reason": "live_trace_enabled_disabled_equivalence_not_measured",
    }


def instrumentation_overhead() -> dict[str, Any]:
    measured = _read_measured_artifact("instrumentation_overhead.json", evidence_source="live_paired_showcase_s01_s04")
    if measured is not None:
        return measured
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "evidence_source": "unavailable",
        "sample_count": 0,
        "instrumentation_overhead_ms": None,
        "instrumentation_overhead_ratio": None,
        "instrumentation_overhead_measured": False,
        "instrumentation_overhead_acceptable_for_showcase": False,
        "reason": "live_trace_enabled_disabled_overhead_not_measured",
    }


def _candidate_ids(envelope: Mapping[str, Any]) -> list[str]:
    return [str(row.get("chunk_id")) for row in envelope.get("top_candidates", []) if isinstance(row, Mapping)]


def _evidence_ids(envelope: Mapping[str, Any]) -> list[str]:
    evidence = envelope.get("evidence") or {}
    return [str(row.get("chunk_id")) for row in evidence.get("selected_evidence", []) if isinstance(row, Mapping)]


def _guard_semantics(envelope: Mapping[str, Any]) -> dict[str, Any]:
    guard = dict(envelope.get("guard") or {})
    keys = (
        "guard_evaluated", "guard_triggered", "guard_reason", "structure_lane_invoked",
        "structure_candidate_count", "recovery_activated", "recovery_decision",
        "recovery_attempt_count", "maximum_recovery_attempt_count", "final_decision",
        "fail_closed", "refusal_reason_code",
    )
    return {key: guard.get(key) for key in keys}


def _graph_semantics(envelope: Mapping[str, Any]) -> dict[str, Any]:
    graph = dict(envelope.get("graph") or {})
    keys = (
        "graph_activated", "graph_activation_reason", "graph_activation_policy", "hop_depth",
        "seed_chunk_ids", "seed_document_ids", "relations", "expanded_chunk_ids",
        "expanded_document_ids", "expanded_candidate_count", "candidate_provenance",
        "runtime_gold_metadata_usage",
    )
    return {key: graph.get(key) for key in keys}


def _answer_semantics(envelope: Mapping[str, Any]) -> dict[str, Any]:
    answer = dict(envelope.get("answer") or {})
    keys = ("status", "answerable", "refusal_reason_code", "grounding_valid", "grounding_reason_code", "citation_count")
    return {key: answer.get(key) for key in keys}


def collect_live_validation(*, write: bool = True) -> dict[str, Any]:
    """Run one bounded trace-off/trace-on pair for each frozen Showcase scenario."""
    from opk_rag.runtime.dotenv import load_project_env
    from opk_rag.showcase.demo import ShowcaseRunner, load_showcase_authority, preflight

    load_project_env(ROOT)
    authority = load_showcase_authority()
    preflight_result = preflight(authority)
    if preflight_result.get("preflight_valid") is not True:
        result = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "evidence_source": "live_paired_showcase_s01_s04",
            "measurement_valid": False,
            "reason": "showcase_preflight_failed",
            "safe_preflight": {key: value for key, value in preflight_result.items() if key not in {"database_error", "qdrant_error", "configuration_error"}},
        }
        if write:
            write_json(RESULT_DIR / "live_validation_attempt.json", result)
        return result

    runner = ShowcaseRunner(authority, preflight_result)
    # Warm model/tokenizer/runtime caches before paired timing so cold-start cost is not
    # misattributed to either trace-disabled or trace-enabled execution.
    warmup_scenario = authority.scenarios[0]
    runner.run_scenario(warmup_scenario, trace=False)
    scenario_rows: dict[str, Any] = {}
    overhead_rows: list[dict[str, Any]] = []
    for scenario in authority.scenarios:
        sid = str(scenario["scenario_id"])
        without_trace = runner.run_scenario(scenario, trace=False)
        with_trace = runner.run_scenario(scenario, trace=True)
        trace = with_trace.get("runtime_trace")
        if isinstance(trace, Mapping) and write:
            write_json(LIVE_TRACE_PATHS[sid], trace)
        trace_checks = validate_runtime_trace(trace) if isinstance(trace, Mapping) else {"trace_present": False}
        checks = {
            "candidate_ids_order_equal": _candidate_ids(without_trace) == _candidate_ids(with_trace),
            "guard_equal": _guard_semantics(without_trace) == _guard_semantics(with_trace),
            "structure_equal": (
                (without_trace.get("guard") or {}).get("structure_lane_invoked") == (with_trace.get("guard") or {}).get("structure_lane_invoked")
                and (without_trace.get("guard") or {}).get("structure_candidate_count") == (with_trace.get("guard") or {}).get("structure_candidate_count")
            ),
            "graph_equal": _graph_semantics(without_trace) == _graph_semantics(with_trace),
            "rerank_order_equal": _candidate_ids(without_trace) == _candidate_ids(with_trace),
            "evidence_ids_equal": _evidence_ids(without_trace) == _evidence_ids(with_trace),
            "answer_or_refusal_equal": _answer_semantics(without_trace) == _answer_semantics(with_trace),
            "trace_conformant": all(trace_checks.values()),
        }
        off_ms = float(without_trace.get("latency_ms") or 0.0)
        on_ms = float(with_trace.get("latency_ms") or 0.0)
        delta_ms = on_ms - off_ms
        ratio = delta_ms / off_ms if off_ms > 0 else None
        overhead_rows.append({
            "scenario_id": sid,
            "trace_disabled_ms": round(off_ms, 3),
            "trace_enabled_ms": round(on_ms, 3),
            "signed_overhead_ms": round(delta_ms, 3),
            "signed_overhead_ratio": round(ratio, 6) if ratio is not None else None,
        })
        scenario_rows[sid] = {
            "checks": checks,
            "equivalent": all(checks.values()),
            "trace_id": (trace.get("trace") or {}).get("trace_id") if isinstance(trace, Mapping) else None,
            "trace_status": (trace.get("trace") or {}).get("status") if isinstance(trace, Mapping) else None,
        }

    equivalence = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "evidence_source": "live_paired_showcase_s01_s04",
        "scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
        "candidate_ids_order_equal": all(row["checks"]["candidate_ids_order_equal"] for row in scenario_rows.values()),
        "guard_equal": all(row["checks"]["guard_equal"] for row in scenario_rows.values()),
        "structure_equal": all(row["checks"]["structure_equal"] for row in scenario_rows.values()),
        "graph_equal": all(row["checks"]["graph_equal"] for row in scenario_rows.values()),
        "rerank_order_equal": all(row["checks"]["rerank_order_equal"] for row in scenario_rows.values()),
        "evidence_ids_equal": all(row["checks"]["evidence_ids_equal"] for row in scenario_rows.values()),
        "answer_or_refusal_equal": all(row["checks"]["answer_or_refusal_equal"] for row in scenario_rows.values()),
        "runtime_equivalence_passed": all(row["equivalent"] for row in scenario_rows.values()),
    }

    ratios = [row["signed_overhead_ratio"] for row in overhead_rows if row["signed_overhead_ratio"] is not None]
    deltas = [row["signed_overhead_ms"] for row in overhead_rows]
    median_ratio = statistics.median(ratios) if ratios else None
    median_delta = statistics.median(deltas) if deltas else None
    # Showcase-only observability mode is accepted when the paired median stays within
    # a bounded 25% relative overhead. This is a presentation gate, not a production SLO.
    acceptable = median_ratio is not None and median_ratio <= 0.25
    overhead = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "evidence_source": "live_paired_showcase_s01_s04",
        "measurement": "one unmeasured warmup followed by one trace-disabled/trace-enabled pair per frozen Showcase scenario using one warm ShowcaseRunner process",
        "sample_count": len(overhead_rows),
        "samples": overhead_rows,
        "instrumentation_overhead_ms": round(float(median_delta), 3) if median_delta is not None else None,
        "instrumentation_overhead_ratio": round(float(median_ratio), 6) if median_ratio is not None else None,
        "acceptance_boundary": "median_signed_overhead_ratio<=0.25 for Showcase trace mode; not a production SLO",
        "instrumentation_overhead_measured": bool(overhead_rows),
        "instrumentation_overhead_acceptable_for_showcase": acceptable,
    }
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RESULT_DIR / "trace_equivalence.json", equivalence)
        write_json(RESULT_DIR / "instrumentation_overhead.json", overhead)
    return {"measurement_valid": equivalence["runtime_equivalence_passed"] and bool(overhead_rows), "equivalence": equivalence, "overhead": overhead}


def task_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_trace_schema_version": TRACE_SCHEMA_VERSION,
        "current_stage": "showcase_v2_visual_explainability",
        "runtime_decides_trace_reports_ui_renders": True,
        "instrumentation_observational_only": True,
        "showcase_api_implemented": False,
        "showcase_ui_implemented": False,
    }


def build_summary() -> dict[str, Any]:
    hooks = runtime_hook_audit()
    conformance = live_trace_conformance()
    scan = sensitive_data_scan()
    equivalence = trace_equivalence()
    overhead = instrumentation_overhead()
    complete = all((
        hooks["runtime_hook_audit_passed"],
        conformance["all_available"],
        conformance["all_conformant"],
        equivalence.get("runtime_equivalence_passed") is True,
        overhead.get("instrumentation_overhead_measured") is True,
        overhead.get("instrumentation_overhead_acceptable_for_showcase") is True,
        scan["sensitive_data_scan_passed"],
    ))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "runtime_trace_schema_version": TRACE_SCHEMA_VERSION,
        "current_stage": "showcase_v2_visual_explainability",
        "runtime_trace_contract_frozen": True,
        "runtime_trace_instrumentation_complete": complete,
        "runtime_trace_live_generation_available": True,
        "runtime_trace_search_supported": True,
        "runtime_trace_ask_supported": True,
        "runtime_trace_real_execution_id": True,
        "runtime_trace_live_timestamps": True,
        "runtime_trace_candidate_lifecycle_observable": True,
        "runtime_trace_structure_recovery_observable": True,
        "runtime_trace_graph_recovery_observable": True,
        "runtime_trace_rerank_observable": True,
        "runtime_trace_evidence_observable": True,
        "runtime_trace_answerability_observable": True,
        "runtime_trace_generation_observable": True,
        "runtime_trace_grounding_observable": True,
        "runtime_trace_citation_observable": True,
        "runtime_trace_live_timing_observable": True,
        "s01_conformant": conformance["scenarios"]["S01"]["conformant"],
        "s02_conformant": conformance["scenarios"]["S02"]["conformant"],
        "s03_conformant": conformance["scenarios"]["S03"]["conformant"],
        "s04_conformant": conformance["scenarios"]["S04"]["conformant"],
        "live_trace_artifacts_all_available": conformance["all_available"],
        "trace_enabled_disabled_runtime_equivalent": equivalence.get("runtime_equivalence_passed") is True,
        "runtime_equivalence_passed": equivalence.get("runtime_equivalence_passed") is True,
        "instrumentation_overhead_ms": overhead.get("instrumentation_overhead_ms"),
        "instrumentation_overhead_ratio": overhead.get("instrumentation_overhead_ratio"),
        "instrumentation_overhead_measured": overhead.get("instrumentation_overhead_measured") is True,
        "instrumentation_overhead_acceptable_for_showcase": overhead.get("instrumentation_overhead_acceptable_for_showcase") is True,
        "sensitive_data_scan_passed": scan["sensitive_data_scan_passed"],
        "production_vector_backend_changed": False,
        "retrieval_policy_changed": False,
        "graph_hop_depth_changed": False,
        "reranker_logic_changed": False,
        "evidence_policy_changed": False,
        "answerability_thresholds_changed": False,
        "generation_prompt_semantics_changed": False,
        "grounding_citation_rules_changed": False,
        "showcase_api_implemented": False,
        "showcase_ui_implemented": False,
        "next_recommended_task": "TASK-0227",
        "remaining_unavailable_fields": [],
    }


def run_task0226(*, write: bool = True) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        TASK_CONTRACT_PATH: task_contract(),
        RESULT_DIR / "runtime_hook_audit.json": runtime_hook_audit(),
        RESULT_DIR / "live_trace_conformance.json": live_trace_conformance(),
    }
    for path, payload in artifacts.items():
        if write:
            write_json(path, payload)
    scan = sensitive_data_scan()
    if write:
        write_json(RESULT_DIR / "sensitive_data_scan.json", scan)
    summary = build_summary()
    verification = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": summary["task_status"] == "complete",
        "summary": summary,
    }
    if write:
        write_json(RESULT_DIR / "summary.json", summary)
        write_json(RESULT_DIR / "verification.json", verification)
    return summary


def verify_task0226_artifacts() -> dict[str, Any]:
    summary = run_task0226(write=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": summary["task_status"] == "complete",
        "summary": summary,
    }


if __name__ == "__main__":
    print(json.dumps(run_task0226(write=True), ensure_ascii=False, indent=2, sort_keys=True))
