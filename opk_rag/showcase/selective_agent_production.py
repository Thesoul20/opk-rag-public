from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from opk_rag.showcase.live_selective_agent_shadow import classify_traffic

PRODUCTION_SCHEMA_VERSION = "opk-rag.task0254.selective-agent-production.v1"
PRODUCTION_ENABLED_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_ENABLED"
PRODUCTION_KILL_SWITCH_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_KILL_SWITCH"
PRODUCTION_RELEASE_STATE_ENV = "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_RELEASE_STATE"
PRODUCTION_DEFAULT_RELEASE_STATE = Path("runtime/selective-agent-production/release-state.json")
TASK0253_SUMMARY_REL = Path("evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation/summary.json")
TASK0253_GATE_REL = Path("evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation/production_promotion_review_gate.json")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SelectiveAgentProductionConfig:
    enabled: bool
    kill_switch: bool
    release_state_path: Path


def load_production_config(
    env: Mapping[str, str] | None = None, *, root: Path | None = None
) -> SelectiveAgentProductionConfig:
    values = os.environ if env is None else env
    enabled = _truthy(values.get(PRODUCTION_ENABLED_ENV))
    # Fail-safe default: kill switch is ON unless explicitly disabled.
    kill_switch = _truthy(values.get(PRODUCTION_KILL_SWITCH_ENV, "1"))
    raw_state = (values.get(PRODUCTION_RELEASE_STATE_ENV) or "").strip()
    state_path = Path(raw_state) if raw_state else PRODUCTION_DEFAULT_RELEASE_STATE
    if root is not None and not state_path.is_absolute():
        state_path = root / state_path
    return SelectiveAgentProductionConfig(enabled=enabled, kill_switch=kill_switch, release_state_path=state_path)


def evaluate_task0253_promotion_readiness(
    *, root: Path, expected_candidate_fingerprint: str | None = None
) -> dict[str, Any]:
    summary = _read_json(root / TASK0253_SUMMARY_REL)
    gate = _read_json(root / TASK0253_GATE_REL)
    fingerprint = str(summary.get("candidate_fingerprint") or "")
    safety_zero = bool(summary.get("safety_invariants_preserved") is True)
    gates = {
        "task0253_complete": summary.get("task_status") == "complete",
        "task0253_advanced": summary.get("final_candidate_decision") == "advance_to_production_promotion_review",
        "canary_execution_performed": summary.get("canary_execution_performed") is True,
        "small_canary_minimum": int(summary.get("small_canary_observation_count") or 0) >= 30,
        "expanded_canary_minimum": int(summary.get("expanded_canary_observation_count") or 0) >= 100,
        "small_canary_advanced": summary.get("small_canary_decision") == "advance_to_expanded_canary",
        "expanded_canary_advanced": summary.get("expanded_canary_decision") == "advance_to_production_promotion_review",
        "exposure_within_bound": float(summary.get("observed_canary_exposure") or 0.0) <= 0.20,
        "provider_response_rate": float(summary.get("provider_response_rate") or 0.0) >= 0.98,
        "structured_validity": float(summary.get("final_structured_validity") or 0.0) >= 0.98,
        "fallback_success": float(summary.get("fallback_success_rate") or 0.0) >= 1.0,
        "kill_switch_validated": summary.get("kill_switch_validated") is True,
        "rollback_validated": summary.get("rollback_validated") is True,
        "production_isolation": int(summary.get("agent_induced_production_failure_count") or 0) == 0,
        "reranker_terminal_stability": int(summary.get("reranker_induced_terminal_decision_instability_count") or 0) == 0,
        "safety_invariants_zero": safety_zero,
        "promotion_gate_advanced": gate.get("decision") == "advance_to_production_promotion_review",
        "candidate_fingerprint_present": bool(fingerprint),
        "candidate_fingerprint_match": expected_candidate_fingerprint in {None, "", fingerprint},
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0254.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "task0253_status": summary.get("task_status"),
        "task0253_decision": summary.get("final_candidate_decision"),
        "approved_candidate_fingerprint": fingerprint or None,
        "small_canary_observation_count": int(summary.get("small_canary_observation_count") or 0),
        "expanded_canary_observation_count": int(summary.get("expanded_canary_observation_count") or 0),
    }


def build_release_identity(
    *,
    candidate_fingerprint: str,
    git_commit: str,
    architecture_digest: str,
    config_digest: str,
    evidence_digests: Mapping[str, str],
) -> dict[str, Any]:
    material = {
        "candidate_fingerprint": candidate_fingerprint,
        "git_commit": git_commit,
        "architecture_digest": architecture_digest,
        "config_digest": config_digest,
        "evidence_digests": dict(sorted(evidence_digests.items())),
    }
    fingerprint = canonical_digest(material)
    return {
        "schema_version": "opk-rag.agentic-rag-v2.production-release.v1",
        "release_id": f"agentic-rag-v2-{fingerprint[:16]}",
        "release_fingerprint": fingerprint,
        **material,
    }


def production_traffic_eligible(*, execution_scope: str, source: str) -> bool:
    _, live_eligible = classify_traffic(execution_scope=execution_scope, source=source)
    return bool(live_eligible and source == "direct_user_request")


def route_selective_agent_production(
    *,
    query: str,
    execution_scope: str,
    source: str = "direct_user_request",
    env: Mapping[str, str] | None = None,
    root: Path,
    expected_candidate_fingerprint: str | None = None,
    necessary_llm_invoked: bool = False,
) -> dict[str, Any]:
    config = load_production_config(env, root=root)
    readiness = evaluate_task0253_promotion_readiness(
        root=root, expected_candidate_fingerprint=expected_candidate_fingerprint
    )
    eligible = production_traffic_eligible(execution_scope=execution_scope, source=source)
    active = bool(config.enabled and not config.kill_switch and eligible and readiness["passed"])
    reason = None
    if not config.enabled:
        reason = "production_agent_disabled"
    elif config.kill_switch:
        reason = "kill_switch_active"
    elif not eligible:
        reason = "traffic_not_eligible"
    elif not readiness["passed"]:
        reason = "task0253_promotion_readiness_not_passed"
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "enabled": config.enabled,
        "kill_switch_active": config.kill_switch,
        "traffic_eligible": eligible,
        "entry_gate_passed": readiness["passed"],
        "entry_gate_blockers": readiness["blockers"],
        "candidate_fingerprint": readiness["approved_candidate_fingerprint"],
        "production_control_plane_active": active,
        "authority_lane": "selective_agent_control_plane" if active else "rule_governed_control",
        # Production mode means the deterministic control plane is active, not that every query calls the LLM.
        "necessary_llm_gate_active": active,
        "llm_invocation_required": bool(active and necessary_llm_invoked),
        "llm_invocation_rate_implied": False,
        "fallback_available": True,
        "reason_code": reason,
        "query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "raw_query_persisted": False,
    }


def maybe_route_selective_agent_production(
    *,
    query: str,
    execution_scope: str,
    source: str = "direct_user_request",
    env: Mapping[str, str] | None = None,
    root: Path,
    necessary_llm_invoked: bool = False,
) -> Mapping[str, Any] | None:
    config = load_production_config(env, root=root)
    if not config.enabled:
        return None
    return route_selective_agent_production(
        query=query,
        execution_scope=execution_scope,
        source=source,
        env=env,
        root=root,
        necessary_llm_invoked=necessary_llm_invoked,
    )


def explicit_activation_request(
    *,
    root: Path,
    env: Mapping[str, str],
    release_identity: Mapping[str, Any],
    explicit: bool,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Separate promotion review from activation. Never called by evaluator/verify paths."""
    if not explicit:
        return {"activated": False, "status": "explicit_activation_required"}
    candidate = str(release_identity.get("candidate_fingerprint") or "")
    readiness = evaluate_task0253_promotion_readiness(root=root, expected_candidate_fingerprint=candidate)
    config = load_production_config(env, root=root)
    blockers = list(readiness["blockers"])
    if not config.enabled:
        blockers.append("production_enabled_flag_required")
    if config.kill_switch:
        blockers.append("kill_switch_must_be_explicitly_disabled_for_activation")
    if blockers:
        return {"activated": False, "status": "blocked", "blockers": blockers, "dry_run": dry_run}
    if dry_run:
        return {"activated": False, "status": "dry_run_ready", "blockers": [], "dry_run": True}
    config.release_state_path.parent.mkdir(parents=True, exist_ok=True)
    config.release_state_path.write_text(
        json.dumps(dict(release_identity), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "activated": True,
        "status": "release_state_written",
        "release_state_path": str(config.release_state_path),
        "dry_run": False,
    }
