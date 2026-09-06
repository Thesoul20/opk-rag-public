from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from opk_rag.showcase.live_selective_agent_shadow import classify_traffic

CANARY_SCHEMA_VERSION = "opk-rag.task0253.selective-agent-canary.v1"
CANARY_ENABLED_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_ENABLED"
CANARY_MODE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_MODE"
CANARY_EXPOSURE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_EXPOSURE"
CANARY_KILL_SWITCH_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_KILL_SWITCH"
CANARY_STORE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_STORE"
CANARY_MAX_SAMPLES_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_MAX_SAMPLES"
CANARY_DEFAULT_STORE = Path("runtime/selective-agent-canary/task0253-canary-observations.jsonl")
CANARY_MODES = ("off", "shadow", "small_canary", "expanded_canary")
MODE_MAX_EXPOSURE = {"off": 0.0, "shadow": 0.0, "small_canary": 0.05, "expanded_canary": 0.20}
DEFAULT_MAX_SAMPLES = 500
TASK0252_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/summary.json")
TASK0252_GATE_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/controlled_promotion_readiness_gate.json")
TASK0252_CANDIDATE_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/candidate_identity.json")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SelectiveAgentCanaryConfig:
    enabled: bool
    mode: str
    exposure: float
    kill_switch: bool
    store_path: Path
    max_samples: int

    @property
    def max_allowed_exposure(self) -> float:
        return MODE_MAX_EXPOSURE[self.mode]


def load_canary_config(env: Mapping[str, str] | None = None, *, root: Path | None = None) -> SelectiveAgentCanaryConfig:
    values = os.environ if env is None else env
    enabled = _truthy(values.get(CANARY_ENABLED_ENV))
    mode = (values.get(CANARY_MODE_ENV) or "off").strip().lower()
    if mode not in CANARY_MODES:
        raise ValueError("invalid_canary_mode")
    raw_exposure = (values.get(CANARY_EXPOSURE_ENV) or "0").strip()
    try:
        exposure = float(raw_exposure)
    except ValueError as exc:
        raise ValueError("invalid_canary_exposure") from exc
    if exposure < 0 or exposure > 0.20:
        raise ValueError("canary_exposure_out_of_absolute_bounds")
    if exposure > MODE_MAX_EXPOSURE[mode]:
        raise ValueError("canary_exposure_exceeds_mode_bound")
    if mode in {"off", "shadow"} and exposure != 0.0:
        raise ValueError("non_authoritative_mode_requires_zero_exposure")
    kill_switch = _truthy(values.get(CANARY_KILL_SWITCH_ENV, "1"))
    raw_store = (values.get(CANARY_STORE_ENV) or "").strip()
    store = Path(raw_store) if raw_store else CANARY_DEFAULT_STORE
    if root is not None and not store.is_absolute():
        store = root / store
    try:
        max_samples = int((values.get(CANARY_MAX_SAMPLES_ENV) or str(DEFAULT_MAX_SAMPLES)).strip())
    except ValueError as exc:
        raise ValueError("invalid_canary_max_samples") from exc
    if max_samples < 30 or max_samples > 5000:
        raise ValueError("canary_max_samples_out_of_bounds")
    if not enabled:
        mode = "off"
        exposure = 0.0
    return SelectiveAgentCanaryConfig(enabled=enabled, mode=mode, exposure=exposure, kill_switch=kill_switch, store_path=store, max_samples=max_samples)


def evaluate_task0252_readiness(*, root: Path, expected_candidate_fingerprint: str | None = None) -> dict[str, Any]:
    summary = _read_json(root / TASK0252_REL)
    gate = _read_json(root / TASK0252_GATE_REL)
    candidate = _read_json(root / TASK0252_CANDIDATE_REL)
    fingerprint = str(candidate.get("candidate_fingerprint") or summary.get("candidate_fingerprint") or "")
    gates = {
        "task0252_complete": summary.get("task_status") == "complete",
        "task0252_advanced": summary.get("candidate_decision") == "advance_to_canary_execution_readiness",
        "task0252_entry_gate": summary.get("entry_gate_passed") is True,
        "external_evaluation_executed": summary.get("pdfqa_external_evaluation_executed") is True,
        "external_holdout_independence": summary.get("pdfqa_independence_proven") is True,
        "promotion_gate_advanced": gate.get("decision") == "advance_to_canary_execution_readiness",
        "safety_invariants_zero": (gate.get("gates") or {}).get("safety_invariants_zero") is True,
        "provider_reliability": (gate.get("gates") or {}).get("provider_reliability") is True,
        "selectivity_bounded": (gate.get("gates") or {}).get("selectivity_bounded") is True,
        "latency_gate": (gate.get("gates") or {}).get("latency_below_always_on_agent") is True,
        "candidate_fingerprint_present": bool(fingerprint),
        "candidate_fingerprint_match": expected_candidate_fingerprint in {None, "", fingerprint},
    }
    blockers = [name for name, ok in gates.items() if not ok]
    return {
        "schema_version": "opk-rag.task0253.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "task0252_status": summary.get("task_status"),
        "task0252_decision": summary.get("candidate_decision"),
        "approved_candidate_fingerprint": fingerprint or None,
    }


def canary_traffic_class(*, execution_scope: str, source: str) -> tuple[str, bool]:
    traffic_class, live_eligible = classify_traffic(execution_scope=execution_scope, source=source)
    return traffic_class, bool(live_eligible and source == "direct_user_request")


def cohort_bucket(*, query: str, execution_scope: str) -> float:
    digest = hashlib.sha256(f"task0253|{execution_scope}|{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def route_canary_request(
    *, query: str, execution_scope: str, source: str = "direct_user_request",
    env: Mapping[str, str] | None = None, root: Path,
    expected_candidate_fingerprint: str | None = None,
) -> dict[str, Any]:
    try:
        config = load_canary_config(env, root=root)
    except ValueError as exc:
        return {
            "schema_version": CANARY_SCHEMA_VERSION, "enabled": False, "mode": "off", "selected": False,
            "authority_lane": "rule_governed_control", "fallback_available": True,
            "configuration_valid": False, "failure_code": str(exc), "kill_switch_active": True,
            "entry_gate_passed": False, "observed_exposure": 0.0,
        }
    traffic_class, eligible = canary_traffic_class(execution_scope=execution_scope, source=source)
    readiness = evaluate_task0252_readiness(root=root, expected_candidate_fingerprint=expected_candidate_fingerprint)
    bucket = cohort_bucket(query=query, execution_scope=execution_scope)
    selected = bool(
        config.enabled and config.mode in {"small_canary", "expanded_canary"} and not config.kill_switch
        and eligible and readiness["passed"] and bucket < config.exposure
    )
    reason = None
    if not config.enabled or config.mode == "off": reason = "canary_disabled"
    elif config.kill_switch: reason = "kill_switch_active"
    elif not eligible: reason = "traffic_not_eligible"
    elif not readiness["passed"]: reason = "task0252_readiness_not_passed"
    elif config.mode == "shadow": reason = "shadow_non_authoritative"
    elif not selected: reason = "control_cohort"
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "enabled": config.enabled,
        "mode": config.mode,
        "configured_exposure": config.exposure,
        "maximum_mode_exposure": config.max_allowed_exposure,
        "kill_switch_active": config.kill_switch,
        "configuration_valid": True,
        "traffic_class": traffic_class,
        "traffic_eligible": eligible,
        "entry_gate_passed": readiness["passed"],
        "entry_gate_blockers": readiness["blockers"],
        "candidate_fingerprint": readiness["approved_candidate_fingerprint"],
        "cohort_bucket": bucket,
        "cohort_digest": _digest(f"{execution_scope}|{_digest(query)}|{bucket:.12f}"),
        "selected": selected,
        "authority_lane": "selective_agent_canary" if selected else "rule_governed_control",
        "fallback_available": True,
        "reason_code": reason,
        "observed_exposure": config.exposure if selected else 0.0,
        "raw_query_persisted": False,
    }


class CanaryObservationStore:
    _FORBIDDEN_KEYS = {"query", "raw_query", "raw_provider_output", "prompt", "system_prompt", "hidden_reasoning", "chain_of_thought", "api_key", "authorization", "secret", "evidence_text", "answer"}
    def __init__(self, config: SelectiveAgentCanaryConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
    @classmethod
    def _validate(cls, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in cls._FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden_canary_persistence_key:{str(key).lower()}")
                cls._validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value: cls._validate(item)
    def read(self) -> list[dict[str, Any]]:
        if not self.config.store_path.is_file(): return []
        rows=[]
        for line in self.config.store_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(row, dict): rows.append(row)
        return rows
    def append(self, record: Mapping[str, Any]) -> bool:
        payload=dict(record); self._validate(payload)
        with self._lock:
            rows=self.read()
            if len(rows) >= self.config.max_samples: return False
            self.config.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.store_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True)+"\n")
        return True


def maybe_route_selective_agent_canary(
    *, query: str, execution_scope: str, source: str = "direct_user_request",
    env: Mapping[str, str] | None = None, root: Path,
    agent_executor: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    # Preserve the existing Production path with effectively zero Canary overhead when disabled.
    try:
        config = load_canary_config(env, root=root)
    except ValueError:
        config = None
    if config is not None and not config.enabled:
        return None
    route = route_canary_request(query=query, execution_scope=execution_scope, source=source, env=env, root=root)
    result=dict(route)
    result.update({"controller_called": False, "authority_applied": False, "fallback_used": False, "agent_induced_production_failure": False})
    if route.get("selected") is True:
        if agent_executor is None:
            result.update({"fallback_used": True, "failure_code": "authoritative_agent_executor_not_bound", "authority_lane": "rule_governed_fallback"})
        else:
            try:
                decision=dict(agent_executor(route))
                result["controller_called"] = True
                # The executor may apply authority only after its own governed downstream validation.
                result["authority_applied"] = decision.get("downstream_validation_passed") is True and decision.get("authority_applied") is True
                result["agent_result_digest"] = decision.get("result_digest")
                if not result["authority_applied"]:
                    result.update({"fallback_used": True, "authority_lane": "rule_governed_fallback"})
            except Exception as exc:
                result.update({"controller_called": True, "fallback_used": True, "failure_code": f"canary_controller_{type(exc).__name__}", "authority_lane": "rule_governed_fallback"})
    result["timestamp_bucket"] = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    result["raw_query_persisted"] = False
    return result
