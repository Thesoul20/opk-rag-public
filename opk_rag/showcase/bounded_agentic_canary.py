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

CANARY_SCHEMA_VERSION = "opk-rag.task0264.selective-agent-canary.v1"
CANARY_ENABLED_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_ENABLED"
CANARY_MODE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_MODE"
CANARY_EXPOSURE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_EXPOSURE"
CANARY_KILL_SWITCH_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_KILL_SWITCH"
CANARY_STORE_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_STORE"
CANARY_MAX_SAMPLES_ENV = "OPK_RAG_SELECTIVE_AGENT_CANARY_MAX_SAMPLES"
CANARY_DEFAULT_STORE = Path("runtime/selective-agent-canary/task0264-canary-observations.jsonl")
CANARY_MODES = ("off", "shadow", "small_canary", "expanded_canary")
MODE_MAX_EXPOSURE = {"off": 0.0, "shadow": 0.0, "small_canary": 0.05, "expanded_canary": 0.20}
DEFAULT_MAX_SAMPLES = 500
TASK0252_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/summary.json")
TASK0252_GATE_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/controlled_promotion_readiness_gate.json")
TASK0252_CANDIDATE_REL = Path("evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/candidate_identity.json")
TASK0263_SUMMARY_REL = Path("evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification/summary.json")
TASK0263_CANDIDATE_REL = Path("evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification/candidate_fingerprint.json")


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


def evaluate_task0263_readiness(*, root: Path, expected_candidate_fingerprint: str | None = None) -> dict[str, Any]:
    summary = _read_json(root / TASK0263_SUMMARY_REL)
    candidate = _read_json(root / TASK0263_CANDIDATE_REL)
    fingerprint = str(candidate.get("candidate_fingerprint") or summary.get("candidate_fingerprint") or "")
    frozen_hashes = dict(candidate.get("file_sha256") or {})
    current_hashes_match = bool(frozen_hashes) and all(
        (root / rel).is_file() and hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected
        for rel, expected in frozen_hashes.items()
    )
    gates = {
        "task0263_complete": summary.get("task_status") == "complete",
        "task0263_all_blocking_gates_passed": summary.get("all_blocking_gates_passed") is True,
        "task0263_advanced": summary.get("candidate_decision") == "advance_to_bounded_agentic_rag_canary_execution",
        "external_evaluation_executed": summary.get("pdfqa_external_evaluation_executed") is True,
        "external_holdout_independence": summary.get("pdfqa_independence_proven") is True,
        "provider_reliability": float(summary.get("provider_cumulative_execution_success_rate") or 0.0) >= 0.98,
        "runtime_gold_exposure_zero": int(summary.get("runtime_gold_exposure_count") or 0) == 0,
        "production_inactive": summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False,
        "candidate_fingerprint_present": bool(fingerprint),
        "candidate_fingerprint_match": expected_candidate_fingerprint in {None, "", fingerprint},
        "summary_candidate_fingerprint_match": summary.get("candidate_fingerprint") == fingerprint,
        "frozen_candidate_file_hashes_match": current_hashes_match,
    }
    blockers = [name for name, ok in gates.items() if not ok]
    return {"schema_version":"opk-rag.task0264.entry-gate.v1","passed":not blockers,"gates":gates,"blockers":blockers,"task0263_status":summary.get("task_status"),"task0263_decision":summary.get("candidate_decision"),"approved_candidate_fingerprint":fingerprint or None}


def canary_traffic_class(*, execution_scope: str, source: str) -> tuple[str, bool]:
    traffic_class, live_eligible = classify_traffic(execution_scope=execution_scope, source=source)
    return traffic_class, bool(live_eligible and source == "direct_user_request")


def cohort_bucket(*, query: str, execution_scope: str) -> float:
    digest = hashlib.sha256(f"task0264|{execution_scope}|{query}".encode("utf-8")).digest()
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
    readiness = evaluate_task0263_readiness(root=root, expected_candidate_fingerprint=expected_candidate_fingerprint)
    bucket = cohort_bucket(query=query, execution_scope=execution_scope)
    selected = bool(
        config.enabled and config.mode in {"small_canary", "expanded_canary"} and not config.kill_switch
        and eligible and readiness["passed"] and bucket < config.exposure
    )
    reason = None
    if not config.enabled or config.mode == "off": reason = "canary_disabled"
    elif config.kill_switch: reason = "kill_switch_active"
    elif not eligible: reason = "traffic_not_eligible"
    elif not readiness["passed"]: reason = "task0263_readiness_not_passed"
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
            selected_count=sum(bool(row.get("selected")) for row in rows)
            if bool(payload.get("selected")) and selected_count >= self.config.max_samples:
                return False
            self.config.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.store_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True)+"\n")
        return True


_EXECUTOR_TELEMETRY_KEYS = (
    "candidate_executor_executed",
    "controller_call_count",
    "provider_request_count",
    "provider_response_count",
    "provider_valid_decision_count",
    "recovery_invoked",
    "recovery_selected",
    "recovery_harmed",
    "recovery_improved",
    "veto_invoked",
    "veto_decision",
    "authoritative_terminal",
    "total_latency_ms",
    "unsafe_finish",
    "llm_direct_finish_authority",
    "abstain_to_finish_override",
    "graph_hop_violation",
    "kb_mutation",
    "grounding_bypass",
    "fabricated_citation",
    "runtime_gold_exposure",
    "fingerprint_mismatch_executed",
)


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
    result.update({"controller_called": False, "candidate_executor_executed": False, "authority_applied": False, "fallback_used": False, "agent_induced_production_failure": False})
    if route.get("selected") is True:
        if agent_executor is None:
            result.update({"fallback_used": True, "failure_code": "authoritative_agent_executor_not_bound", "authority_lane": "rule_governed_fallback"})
        else:
            try:
                decision=dict(agent_executor(route))
                result["candidate_executor_executed"] = decision.get("candidate_executor_executed") is True
                result["controller_called"] = int(decision.get("controller_call_count") or 0) > 0
                # The executor may apply authority only after its own governed downstream validation.
                result["authority_applied"] = decision.get("downstream_validation_passed") is True and decision.get("authority_applied") is True
                result["agent_result_digest"] = decision.get("result_digest")
                for key in _EXECUTOR_TELEMETRY_KEYS:
                    if key in decision:
                        result[key] = decision[key]
                if decision.get("fallback_used") is True:
                    result["fallback_used"] = True
                if not result["authority_applied"]:
                    result.update({"fallback_used": True, "authority_lane": "rule_governed_fallback"})
            except Exception as exc:
                result.update({"candidate_executor_executed": True, "fallback_used": True, "failure_code": f"canary_controller_{type(exc).__name__}", "authority_lane": "rule_governed_fallback"})
    result["timestamp_bucket"] = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    result["raw_query_persisted"] = False
    result["raw_provider_output_persisted"] = False
    result["hidden_reasoning_persisted"] = False
    result["runtime_gold_exposure"] = False
    result["execution_scope"] = execution_scope
    result["query_digest"] = _digest(query)
    try:
        result["kill_switch_validation_observation"] = False
        result["post_kill_switch_agent_execution_count"] = None
        if config is not None and config.enabled and result.get("traffic_eligible") is True:
            store = CanaryObservationStore(config)
            existing = store.read()
            prior_real_selected = any(
                row.get("selected") is True and row.get("candidate_executor_executed") is True
                for row in existing
            )
            prior_kill_validation = any(row.get("kill_switch_validation_observation") is True for row in existing)
            if result.get("selected") is True:
                result["recorded"] = store.append(result)
            elif config.kill_switch and prior_real_selected and not prior_kill_validation:
                # This is only emitted on a genuine eligible request after real Canary
                # traffic exists. Evaluator mechanism probes use a separate env/store
                # and therefore cannot manufacture real rollback evidence.
                result["kill_switch_validation_observation"] = True
                result["post_kill_switch_agent_execution_count"] = 0
                result["recorded"] = store.append(result)
            else:
                result["recorded"] = False
        else:
            result["recorded"] = False
    except Exception as exc:
        result["recorded"] = False
        result["telemetry_failure_code"] = f"canary_store_{type(exc).__name__}"
    return result
