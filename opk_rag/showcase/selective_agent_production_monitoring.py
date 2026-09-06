from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from opk_rag.showcase.selective_agent_production import production_traffic_eligible

MONITOR_SCHEMA_VERSION = "opk-rag.task0255.production-monitoring.v1"
MONITOR_ENABLED_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_MONITORING_ENABLED"
MONITOR_STORE_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_MONITORING_STORE"
MONITOR_MAX_SAMPLES_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_MONITORING_MAX_SAMPLES"
MONITOR_DEFAULT_STORE = Path("runtime/selective-agent-production-monitoring/task0255-production-observations.jsonl")
MONITOR_MIN_FORMAL_SAMPLES = 100
MONITOR_RECOMMENDED_FREEZE_SAMPLES = 300
MONITOR_DEFAULT_MAX_SAMPLES = 1000
TASK0254_SUMMARY_REL = Path("evaluation-data/results/task0254-selective-agent-production-promotion-review-and-release-freeze/summary.json")
TASK0254_RELEASE_REL = Path("evaluation-data/results/task0254-selective-agent-production-promotion-review-and-release-freeze/release_identity.json")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timestamp_bucket() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ProductionMonitoringConfig:
    enabled: bool
    store_path: Path
    max_samples: int


def load_production_monitoring_config(
    env: Mapping[str, str] | None = None, *, root: Path | None = None
) -> ProductionMonitoringConfig:
    values = os.environ if env is None else env
    enabled = _truthy(values.get(MONITOR_ENABLED_ENV))
    raw = (values.get(MONITOR_STORE_ENV) or "").strip()
    store = Path(raw) if raw else MONITOR_DEFAULT_STORE
    if root is not None and not store.is_absolute():
        store = root / store
    try:
        max_samples = int((values.get(MONITOR_MAX_SAMPLES_ENV) or str(MONITOR_DEFAULT_MAX_SAMPLES)).strip())
    except ValueError as exc:
        raise ValueError("invalid_production_monitoring_max_samples") from exc
    if max_samples < MONITOR_MIN_FORMAL_SAMPLES or max_samples > 10000:
        raise ValueError("production_monitoring_max_samples_out_of_bounds")
    return ProductionMonitoringConfig(enabled=enabled, store_path=store, max_samples=max_samples)


def evaluate_task0254_release_readiness(
    *, root: Path, expected_release_fingerprint: str | None = None
) -> dict[str, Any]:
    summary = _read_json(root / TASK0254_SUMMARY_REL)
    release = _read_json(root / TASK0254_RELEASE_REL)
    release_fp = str(release.get("release_fingerprint") or summary.get("release_fingerprint") or "")
    gates = {
        "task0254_complete": summary.get("task_status") == "complete",
        "task0254_promoted": summary.get("candidate_decision") == "promote_and_freeze_agentic_rag_v2",
        "promotion_review_promoted": summary.get("promotion_review_decision") == "promote_and_freeze_agentic_rag_v2",
        "production_promotion_executed": summary.get("production_promotion_executed") is True,
        "production_agentic_v2_active": summary.get("production_agentic_v2_active") is True,
        "release_frozen": summary.get("release_frozen") is True and release.get("release_frozen") is True,
        "entry_gate_passed": summary.get("entry_gate_passed") is True,
        "candidate_identity_match": summary.get("candidate_identity_match") is True,
        "rollback_target_available": summary.get("rollback_target_available") is True,
        "kill_switch_validated": summary.get("kill_switch_mechanism_validated") is True,
        "release_fingerprint_present": bool(release_fp),
        "release_fingerprint_match": expected_release_fingerprint in {None, "", release_fp},
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0255.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "task0254_status": summary.get("task_status"),
        "task0254_decision": summary.get("candidate_decision"),
        "release_id": release.get("release_id") or summary.get("release_id"),
        "release_fingerprint": release_fp or None,
    }


class ProductionMonitoringStore:
    _FORBIDDEN_KEYS = {
        "query", "raw_query", "raw_provider_output", "raw_output", "prompt", "system_prompt",
        "hidden_reasoning", "chain_of_thought", "api_key", "authorization", "secret",
        "evidence_text", "evidence_excerpts", "full_answer", "answer_text",
    }

    def __init__(self, config: ProductionMonitoringConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

    @classmethod
    def _validate(cls, value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key).lower()
                if name in cls._FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden_production_monitoring_key:{'.'.join((*path, name))}")
                cls._validate(item, (*path, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                cls._validate(item, (*path, str(index)))

    def read(self) -> list[dict[str, Any]]:
        if not self.config.store_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.config.store_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append(self, record: Mapping[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        payload = dict(record)
        self._validate(payload)
        with self._lock:
            rows = self.read()
            if len(rows) >= self.config.max_samples:
                return False
            self.config.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.store_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return True


def build_production_observation_record(
    *,
    query: str,
    execution_scope: str,
    production_route: Mapping[str, Any],
    controller_telemetry: Mapping[str, Any],
    release_fingerprint: str,
    source: str = "direct_user_request",
) -> dict[str, Any]:
    if not production_traffic_eligible(execution_scope=execution_scope, source=source):
        raise ValueError("production_monitoring_traffic_not_eligible")
    if production_route.get("production_control_plane_active") is not True:
        raise ValueError("production_control_plane_not_active")
    if not controller_telemetry.get("authoritative_controller_telemetry") is True:
        raise ValueError("authoritative_controller_telemetry_required")
    if str(controller_telemetry.get("release_fingerprint") or "") != release_fingerprint:
        raise ValueError("production_release_fingerprint_mismatch")
    required = (
        "controller_call_count", "necessary_llm_invoked", "recovery_invoked", "veto_invoked",
        "provider_requests", "provider_responses", "provider_valid_decisions", "fallback_required",
        "fallback_success", "agent_induced_production_failure", "terminal", "total_latency_ms",
    )
    missing = [name for name in required if name not in controller_telemetry]
    if missing:
        raise ValueError("missing_authoritative_controller_fields:" + ",".join(missing))
    query_digest = _digest(query)
    return {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "observation_digest": _digest(f"{query_digest}|{execution_scope}|{time.time_ns()}|{os.getpid()}"),
        "timestamp_bucket": _timestamp_bucket(),
        "query_digest": query_digest,
        "execution_scope": execution_scope,
        "traffic_class": f"production_user_{execution_scope}",
        "production_traffic_eligible": True,
        "release_fingerprint": release_fingerprint,
        "controller": {
            "controller_call_count": int(controller_telemetry.get("controller_call_count") or 0),
            "necessary_llm_invoked": bool(controller_telemetry.get("necessary_llm_invoked")),
            "ambiguity_rewrite_invoked": bool(controller_telemetry.get("ambiguity_rewrite_invoked")),
            "recovery_invoked": bool(controller_telemetry.get("recovery_invoked")),
            "recovery_improved": bool(controller_telemetry.get("recovery_improved")),
            "recovery_harmed": bool(controller_telemetry.get("recovery_harmed")),
            "veto_invoked": bool(controller_telemetry.get("veto_invoked")),
            "veto_decision": controller_telemetry.get("veto_decision"),
            "provider_requests": int(controller_telemetry.get("provider_requests") or 0),
            "provider_responses": int(controller_telemetry.get("provider_responses") or 0),
            "provider_valid_decisions": int(controller_telemetry.get("provider_valid_decisions") or 0),
            "controller_latency_ms": float(controller_telemetry.get("controller_latency_ms") or 0.0),
            "input_tokens": int(controller_telemetry.get("input_tokens") or 0),
            "output_tokens": int(controller_telemetry.get("output_tokens") or 0),
        },
        "runtime": {
            "terminal": controller_telemetry.get("terminal"),
            "fallback_required": bool(controller_telemetry.get("fallback_required")),
            "fallback_success": bool(controller_telemetry.get("fallback_success")),
            "agent_induced_production_failure": bool(controller_telemetry.get("agent_induced_production_failure")),
            "top_rerank_score": controller_telemetry.get("top_rerank_score"),
            "candidate_identity_digest": controller_telemetry.get("candidate_identity_digest"),
            "evidence_identity_digest": controller_telemetry.get("evidence_identity_digest"),
            "graph_hop_depth": int(controller_telemetry.get("graph_hop_depth") or 0),
            "grounding_checked": controller_telemetry.get("grounding_checked"),
            "grounding_passed": controller_telemetry.get("grounding_passed"),
            "citation_checked": controller_telemetry.get("citation_checked"),
            "citation_valid": controller_telemetry.get("citation_valid"),
            "total_latency_ms": float(controller_telemetry.get("total_latency_ms") or 0.0),
        },
        "safety": {
            "llm_finish_authority_count": int(controller_telemetry.get("llm_finish_authority_count") or 0),
            "abstain_to_finish_override_count": int(controller_telemetry.get("abstain_to_finish_override_count") or 0),
            "necessary_llm_gate_bypass_count": int(controller_telemetry.get("necessary_llm_gate_bypass_count") or 0),
            "unauthorized_action_execution_count": int(controller_telemetry.get("unauthorized_action_execution_count") or 0),
            "guard_bypass_count": int(controller_telemetry.get("guard_bypass_count") or 0),
            "recovery_pipeline_bypass_count": int(controller_telemetry.get("recovery_pipeline_bypass_count") or 0),
            "graph_hop_violation_count": int(controller_telemetry.get("graph_hop_violation_count") or 0),
            "knowledge_base_mutation_count": int(controller_telemetry.get("knowledge_base_mutation_count") or 0),
            "grounding_bypass_count": int(controller_telemetry.get("grounding_bypass_count") or 0),
            "citation_bypass_count": int(controller_telemetry.get("citation_bypass_count") or 0),
            "benchmark_gold_exposure_count": int(controller_telemetry.get("benchmark_gold_exposure_count") or 0),
            "hidden_reasoning_persisted_count": int(controller_telemetry.get("hidden_reasoning_persisted_count") or 0),
            "raw_provider_output_persisted_count": int(controller_telemetry.get("raw_provider_output_persisted_count") or 0),
            "secret_exposure_count": int(controller_telemetry.get("secret_exposure_count") or 0),
        },
        "raw_query_persisted": False,
        "raw_provider_output_persisted": False,
        "hidden_reasoning_persisted": False,
    }


def maybe_record_selective_agent_production_observation(
    *,
    query: str,
    execution_scope: str,
    production_route: Mapping[str, Any] | None,
    controller_telemetry: Mapping[str, Any] | None,
    source: str = "direct_user_request",
    env: Mapping[str, str] | None = None,
    root: Path,
) -> Mapping[str, Any] | None:
    config = load_production_monitoring_config(env, root=root)
    if not config.enabled:
        return None
    readiness = evaluate_task0254_release_readiness(root=root)
    if not readiness["passed"]:
        return {
            "schema_version": MONITOR_SCHEMA_VERSION,
            "recorded": False,
            "status": "blocked_by_task0254_release_gate",
            "entry_gate_blockers": readiness["blockers"],
        }
    if production_route is None or production_route.get("production_control_plane_active") is not True:
        return {"schema_version": MONITOR_SCHEMA_VERSION, "recorded": False, "status": "production_control_plane_not_active"}
    if controller_telemetry is None or controller_telemetry.get("authoritative_controller_telemetry") is not True:
        return {
            "schema_version": MONITOR_SCHEMA_VERSION,
            "recorded": False,
            "status": "authoritative_controller_telemetry_required",
        }
    try:
        record = build_production_observation_record(
            query=query,
            execution_scope=execution_scope,
            production_route=production_route,
            controller_telemetry=controller_telemetry,
            release_fingerprint=str(readiness["release_fingerprint"]),
            source=source,
        )
        stored = ProductionMonitoringStore(config).append(record)
        return {
            "schema_version": MONITOR_SCHEMA_VERSION,
            "recorded": stored,
            "status": "recorded" if stored else "store_capacity_reached",
            "observation_digest": record["observation_digest"],
            "raw_query_persisted": False,
        }
    except Exception as exc:
        return {
            "schema_version": MONITOR_SCHEMA_VERSION,
            "recorded": False,
            "status": "monitoring_record_rejected",
            "failure_code": f"production_monitoring_{type(exc).__name__}",
        }
