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

LIVE_SHADOW_SCHEMA_VERSION = "opk-rag.task0251.live-shadow-observation.v1"
LIVE_SHADOW_ENV = "OPK_RAG_SELECTIVE_AGENT_LIVE_SHADOW_ENABLED"
LIVE_SHADOW_STORE_ENV = "OPK_RAG_SELECTIVE_AGENT_LIVE_SHADOW_STORE"
LIVE_SHADOW_MAX_SAMPLES_ENV = "OPK_RAG_SELECTIVE_AGENT_LIVE_SHADOW_MAX_SAMPLES"
LIVE_SHADOW_MIN_SAMPLES = 30
LIVE_SHADOW_DEFAULT_MAX_SAMPLES = 60
LIVE_SHADOW_DEFAULT_STORE = Path("runtime/selective-agent-live-shadow/task0251-live-observations.jsonl")
MATERIAL_RERANK_SCORE_DELTA = 0.15
ELIGIBLE_TRAFFIC_CLASSES = {"live_user_search", "live_user_ask"}
ALL_TRAFFIC_CLASSES = {
    "live_user_search",
    "live_user_ask",
    "frozen_benchmark_runtime_replay",
    "synthetic_test",
    "showcase_scenario",
    "failure_injection",
    "controlled_realistic_dogfooding",
}


def _env_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def live_shadow_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return _env_true(values.get(LIVE_SHADOW_ENV))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_ids(ids: tuple[str, ...], *, sorted_ids: bool = False) -> str:
    values = tuple(sorted(ids)) if sorted_ids else ids
    return _digest("|".join(values))


def _timestamp_bucket() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def classify_traffic(*, execution_scope: str, source: str) -> tuple[str, bool]:
    """Classify request origin. Only direct interactive Search/Ask is promotion eligible."""
    if source == "direct_user_request" and execution_scope == "search":
        return "live_user_search", True
    if source == "direct_user_request" and execution_scope == "ask":
        return "live_user_ask", True
    if source == "showcase_scenario":
        return "showcase_scenario", False
    if source == "frozen_benchmark_runtime_replay":
        return "frozen_benchmark_runtime_replay", False
    if source == "failure_injection":
        return "failure_injection", False
    if source == "controlled_realistic_dogfooding":
        return "controlled_realistic_dogfooding", False
    if source == "controlled_realistic_evaluation":
        return "controlled_realistic_evaluation", False
    return "synthetic_test", False


@dataclass(frozen=True)
class LiveShadowConfig:
    enabled: bool
    store_path: Path
    min_samples: int = LIVE_SHADOW_MIN_SAMPLES
    max_samples: int = LIVE_SHADOW_DEFAULT_MAX_SAMPLES


def load_live_shadow_config(env: Mapping[str, str] | None = None, *, root: Path | None = None) -> LiveShadowConfig:
    values = os.environ if env is None else env
    raw = (values.get(LIVE_SHADOW_STORE_ENV) or "").strip()
    path = Path(raw) if raw else LIVE_SHADOW_DEFAULT_STORE
    if not path.is_absolute() and root is not None:
        path = root / path
    try:
        max_samples = int((values.get(LIVE_SHADOW_MAX_SAMPLES_ENV) or str(LIVE_SHADOW_DEFAULT_MAX_SAMPLES)).strip())
    except ValueError:
        max_samples = LIVE_SHADOW_DEFAULT_MAX_SAMPLES
    max_samples = min(max(max_samples, LIVE_SHADOW_MIN_SAMPLES), 500)
    return LiveShadowConfig(enabled=live_shadow_enabled(values), store_path=path, max_samples=max_samples)


class LiveShadowObservationStore:
    """Bounded privacy-safe JSONL store. It never accepts raw query/provider output fields."""

    _FORBIDDEN_KEYS = {
        "query",
        "raw_query",
        "raw_provider_output",
        "raw_output",
        "prompt",
        "system_prompt",
        "hidden_reasoning",
        "chain_of_thought",
        "api_key",
        "authorization",
        "secret",
        "evidence_text",
        "evidence_excerpts",
    }

    def __init__(self, config: LiveShadowConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

    def read(self) -> list[dict[str, Any]]:
        path = self.config.store_path
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def append(self, record: Mapping[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        payload = dict(record)
        self._validate_privacy(payload)
        with self._lock:
            current = self.read()
            eligible_count = sum(bool(row.get("live_traffic_eligible")) for row in current)
            if bool(payload.get("live_traffic_eligible")) and eligible_count >= self.config.max_samples:
                return False
            self.config.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.store_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    @classmethod
    def _validate_privacy(cls, value: Any, *, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).lower()
                if key_text in cls._FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden live-shadow persistence key: {'.'.join((*path, key_text))}")
                cls._validate_privacy(item, path=(*path, key_text))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                cls._validate_privacy(item, path=(*path, str(index)))


def _ids(search_response: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates = tuple(str(item.chunk_id) for item in search_response.results)
    bundle = search_response.evidence_bundle
    evidence = tuple(str(item.chunk_id) for item in bundle.items) if bundle is not None else ()
    return candidates, evidence


def _overlap(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    aset, bset = set(a), set(b)
    return len(aset & bset) / max(1, len(aset | bset))


def _rule_from_response(search_response: Any, answerability: Any, *, latency_ms: float = 0.0):
    from opk_rag.evaluation.task0246_runtime_support import RuleGovernedResult

    evidence_items = search_response.evidence_bundle.items if search_response.evidence_bundle else ()
    signals = search_response.evidence_signals
    guard = dict(search_response.guard_trace or {})
    graph = dict(search_response.graph_trace or {})
    return RuleGovernedResult(
        status="finished" if answerability.status in {"answerable", "partially_answerable"} else "abstained",
        candidate_count=len(search_response.results),
        evidence_count=len(evidence_items),
        source_count=int(signals.selected_source_document_count) if signals is not None else len({str(x.document_id) for x in evidence_items}),
        answerability_status=answerability.status,
        answerability_reason_code=answerability.reason_code,
        structure_lane_invoked=guard.get("structure_lane_invoked") is True,
        graph_activated=graph.get("graph_activated") is True,
        graph_hop_depth=int(graph.get("hop_depth") or 0),
        top_rerank_score=signals.top_reranker_score if signals is not None else None,
        latency_ms=latency_ms,
    )


def _seed(run_id: str, query: str, rule: Any):
    from opk_rag.agentic_v2.observation import build_agent_observation
    from opk_rag.agentic_v2.state import AgentState

    state = AgentState(
        run_id=run_id,
        original_query=query,
        current_query=query,
        retrieval_call_count=1,
        candidate_count=rule.candidate_count,
        evidence_count=rule.evidence_count,
        source_count=rule.source_count,
        top_rerank_score=rule.top_rerank_score,
        answerability_status=rule.answerability_status,
    )
    observation = build_agent_observation(
        state,
        structure_context_available=rule.structure_lane_invoked,
        graph_relation_available=rule.graph_activated,
    )
    return state, observation


def _improved(before: Any, after: Any) -> bool:
    if after is None:
        return False
    return bool(
        ((after.top_rerank_score or 0.0) - (before.top_rerank_score or 0.0)) >= 0.05
        or after.evidence_count > before.evidence_count
        or (before.answerability_status != "answerable" and after.answerability_status == "answerable")
    )


def _harmed(before: Any, after: Any) -> bool:
    if after is None:
        return False
    order = {"answerable": 3, "partially_answerable": 2, "insufficient_evidence": 1, "unanswerable": 0, "unknown": 0}
    return bool(
        ((before.top_rerank_score or 0.0) - (after.top_rerank_score or 0.0)) >= 0.05
        or after.evidence_count < before.evidence_count
        or order.get(after.answerability_status, 0) < order.get(before.answerability_status, 0)
    )


class _CaptureAdapter:
    def __init__(self, binding: Any) -> None:
        from opk_rag.evaluation.task0246_runtime_support import AnswerabilityAwareSearchAdapter

        class Capture(AnswerabilityAwareSearchAdapter):
            def __init__(self, inner_binding):
                super().__init__(inner_binding)
                self.last_search_response = None
                self.last_answerability = None

            def _search(self, action):
                response, answerability, rule = super()._search(action)
                self.last_search_response = response
                self.last_answerability = answerability
                return response, answerability, rule

        self.adapter = Capture(binding)


def _run_shadow_controller(*, query: str, search_response: Any, answerability: Any, binding: Any, execution_scope: str) -> dict[str, Any]:
    """Execute bounded real DS4 Recovery/Veto after Production result is already fixed."""
    from opk_rag.agentic_v2.guard import AgentGuard
    from opk_rag.agentic_v2.policy_prompt import AGENTIC_V2_POLICY_PROMPT_V2_VERSION
    from opk_rag.agentic_v2.policy_provider import OpenAICompatibleV2PolicyProvider
    from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime
    from opk_rag.agentic_v2.runtime_context import AgentRuntimeContext
    from opk_rag.agentic_v2.state_transition import AgentStateTransitionEngine
    from opk_rag.agentic_v2.tool_executor import GuardedToolExecutor
    from opk_rag.agentic_v2.tools import AgentToolContext, default_agent_tool_registry
    from opk_rag.evaluation.task0247_selective_runtime import RecoveryOnlyAllowedActionResolver, SelectiveRecoveryController
    from opk_rag.evaluation.task0248_selective_runtime import AbstentionVetoDecision, compact_evidence_excerpts
    from opk_rag.evaluation.task0249_selective_reliability_and_conflict import EvaluationStructuredTransport
    from opk_rag.evaluation.task0258_runtime_support import (
        BoundedReliableNarrowPolicyV3,
        ConflictGateV3,
        ConflictSignalDetectorV3,
        veto_prompt_v3,
    )
    from opk_rag.showcase.selective_agent_shadow import observe_search_response

    started = time.monotonic()
    gate_observation = observe_search_response(search_response, execution_scope=execution_scope)
    production_rule = _rule_from_response(search_response, answerability)
    production_candidates, production_evidence = _ids(search_response)
    final_rule = production_rule
    final_response = search_response
    final_answerability = answerability
    recovery_called = False
    recovery_selected = None
    recovery_failure = None
    recovery_improved = False
    recovery_harmed = False
    recovery_latency = recovery_input = recovery_output = recovery_requests = recovery_responses = recovery_valid = 0
    recovery_transport_retries = recovery_structural_repairs = 0
    recovery_diagnostics: list[dict[str, Any]] = []

    # Direct Showcase Search/Ask has no conversation context. Ambiguity is recorded but never rewritten without context.
    ambiguity_context_available = False
    if gate_observation.get("shadow_role") == "recovery_selection":
        capture = _CaptureAdapter(binding).adapter
        resolver = RecoveryOnlyAllowedActionResolver()
        recovery_policy = LLMAgentPolicyRuntime(
            provider=OpenAICompatibleV2PolicyProvider(timeout_seconds=30, temperature=0.0, max_tokens=512),
            max_transport_retries=1,
            prompt_version=AGENTIC_V2_POLICY_PROMPT_V2_VERSION,
        )
        controller = SelectiveRecoveryController(
            policy=recovery_policy,
            resolver=resolver,
            guard=AgentGuard(resolver=resolver),
            tool_executor=GuardedToolExecutor(registry=default_agent_tool_registry()),
            state_transition=AgentStateTransitionEngine(resolver=resolver),
        )
        run_id = f"task0251-{_digest(query)[:16]}"
        state, observation = _seed(run_id, query, production_rule)
        execution = controller.run(
            state=state,
            observation=observation,
            context=AgentRuntimeContext(
                tool_context=AgentToolContext(run_id=run_id, retrieval_adapter=capture),
                observation_signal_resolver=capture.observation_signals,
            ),
            baseline_terminal=production_rule.status,
        )
        recovery_called = execution.policy_called
        recovery_selected = execution.recovery_selected
        recovery_failure = execution.policy_failure_code
        recovery_latency = execution.policy_latency_ms
        recovery_input = execution.policy_input_tokens
        recovery_output = execution.policy_output_tokens
        policy_rows = tuple(recovery_policy.metrics.records)
        recovery_diagnostics = [d.model_dump(mode="json") for d in recovery_policy.last_diagnostics]
        recovery_requests = sum(int(row.get("provider_request_count") or 1) for row in policy_rows)
        recovery_responses = sum(int(row.get("provider_response_count") or 0) for row in policy_rows)
        recovery_transport_retries = sum(bool(row.get("transport_retry")) for row in policy_rows)
        recovery_structural_repairs = sum(bool(row.get("repair_attempt")) for row in policy_rows)
        recovery_valid = 1 if execution.policy_called and execution.policy_failure_code is None else 0
        if execution.policy_called and execution.recovery_selected and capture.last_rule_result is not None:
            final_rule = capture.last_rule_result
            final_response = capture.last_search_response or search_response
            final_answerability = capture.last_answerability or answerability
            recovery_improved = _improved(production_rule, final_rule)
            recovery_harmed = _harmed(production_rule, final_rule)

    excerpts = compact_evidence_excerpts(final_response, max_items=3, max_chars=800)
    detector = ConflictSignalDetectorV3()
    signals = detector.detect(
        query=query,
        evidence_excerpts=excerpts,
        answerability_status=final_rule.answerability_status,
        top_rerank_score=final_rule.top_rerank_score,
        graph_relation_available=final_rule.graph_activated,
        recovery_attempted=recovery_called,
        recovery_improved_evidence=recovery_improved,
    )
    veto_gate = ConflictGateV3().decide(terminal=final_rule.status, signals=signals)
    shadow_terminal = final_rule.status
    veto_called = veto_gate.invoke_veto
    veto_valid: bool | None = None
    veto_decision = None
    veto_failure = None
    veto_latency = veto_input = veto_output = veto_requests = veto_responses = 0
    veto_transport_retries = veto_structural_repairs = 0
    if veto_called:
        veto_policy = BoundedReliableNarrowPolicyV3(
            transport=EvaluationStructuredTransport(mode="json_object", timeout_seconds=30, max_tokens=320),
            max_transport_retries=1,
            max_structural_repairs=1,
        )
        veto = veto_policy.decide(
            prompt=veto_prompt_v3(query=query, signals=signals, evidence_excerpts=excerpts),
            model_type=AbstentionVetoDecision,
            function_name="decide_abstention_veto",
        )
        veto_valid = veto.valid
        veto_failure = veto.failure_code
        veto_latency = veto.latency_ms
        veto_input = veto.input_tokens
        veto_output = veto.output_tokens
        veto_requests = veto.provider_requests
        veto_responses = veto.response_count
        veto_transport_retries = veto.transport_retry_count
        veto_structural_repairs = veto.structural_repair_count
        if veto.valid and isinstance(veto.decision, AbstentionVetoDecision):
            veto_decision = veto.decision.decision
            if veto.decision.decision == "abstain" and shadow_terminal == "finished":
                shadow_terminal = "abstained"

    final_candidates, final_evidence = _ids(final_response)
    controller_calls = int(recovery_called) + int(veto_called)
    provider_requests = recovery_requests + veto_requests
    provider_responses = recovery_responses + veto_responses
    provider_valid_decisions = recovery_valid + (1 if veto_valid else 0)
    return {
        "gate": gate_observation,
        "ambiguity_context_available": ambiguity_context_available,
        "controller_call_count": controller_calls,
        "recovery_policy_called": recovery_called,
        "recovery_selected": recovery_selected,
        "recovery_failure_code": recovery_failure,
        "recovery_diagnostics": recovery_diagnostics,
        "recovery_improved": recovery_improved,
        "recovery_harmed": recovery_harmed,
        "veto_invoked": veto_called,
        "veto_valid": veto_valid,
        "veto_decision": veto_decision,
        "veto_failure_code": veto_failure,
        "conflict_reason_codes": tuple(signals.reason_codes),
        "conflict_gate_version": "opk-rag.task0258.conflict-gate-v3.v1",
        "shadow_terminal": shadow_terminal,
        "shadow_answerability_status": final_rule.answerability_status,
        "shadow_top_rerank_score": final_rule.top_rerank_score,
        "candidate_overlap": _overlap(production_candidates, final_candidates),
        "evidence_overlap": _overlap(production_evidence, final_evidence),
        "candidate_count_delta": len(final_candidates) - len(production_candidates),
        "evidence_count_delta": len(final_evidence) - len(production_evidence),
        "controller_latency_ms": recovery_latency + veto_latency,
        "controller_input_tokens": recovery_input + veto_input,
        "controller_output_tokens": recovery_output + veto_output,
        "provider_requests": provider_requests,
        "provider_responses": provider_responses,
        "provider_valid_decisions": provider_valid_decisions,
        "recovery_provider_requests": recovery_requests,
        "recovery_transport_retry_count": recovery_transport_retries,
        "recovery_structural_repair_count": recovery_structural_repairs,
        "veto_provider_requests": veto_requests,
        "veto_transport_retry_count": veto_transport_retries,
        "veto_structural_repair_count": veto_structural_repairs,
        "elapsed_shadow_ms": max(0, int((time.monotonic() - started) * 1000)),
        # Internal-only candidate artifacts for TASK-0264 authoritative Canary binding.
        # build_live_observation_record() intentionally does not persist these objects.
        "_final_search_response": final_response,
        "_final_answerability": final_answerability,
        "_final_rule": final_rule,
    }


def execute_selective_agent_candidate(*, query: str, search_response: Any, answerability: Any, binding: Any, execution_scope: str) -> dict[str, Any]:
    """Run the frozen Selective Guarded Agent candidate before downstream authority.

    This reuses the same Recovery/Veto implementation used by TASK-0257 through
    TASK-0262. The returned SearchResponse is process-local and must never be
    persisted by Canary telemetry.
    """
    return _run_shadow_controller(
        query=query,
        search_response=search_response,
        answerability=answerability,
        binding=binding,
        execution_scope=execution_scope,
    )


def build_live_observation_record(
    *,
    query: str,
    search_response: Any,
    answerability: Any,
    binding: Any,
    execution_scope: str,
    source: str = "direct_user_request",
    production_latency_ms: float | None = None,
) -> dict[str, Any]:
    traffic_class, eligible = classify_traffic(execution_scope=execution_scope, source=source)
    if traffic_class not in ALL_TRAFFIC_CLASSES:
        raise ValueError("unsupported traffic class")
    candidates, evidence = _ids(search_response)
    top_score = search_response.evidence_signals.top_reranker_score if search_response.evidence_signals is not None else None
    production_terminal = "finished" if answerability.status in {"answerable", "partially_answerable"} else "abstained"
    shadow = _run_shadow_controller(
        query=query,
        search_response=search_response,
        answerability=answerability,
        binding=binding,
        execution_scope=execution_scope,
    )
    query_digest = _digest(query)
    return {
        "schema_version": LIVE_SHADOW_SCHEMA_VERSION,
        "observation_digest": _digest(f"{query_digest}|{time.time_ns()}|{execution_scope}"),
        "timestamp_bucket": _timestamp_bucket(),
        "request_digest": _digest(f"{query_digest}|{execution_scope}|{time.time_ns()}"),
        "query_digest": query_digest,
        "execution_scope": execution_scope,
        "traffic_class": traffic_class,
        "live_traffic_eligible": eligible,
        "raw_query_persisted": False,
        "raw_provider_output_persisted": False,
        "hidden_reasoning_persisted": False,
        "secret_exposure_count": 0,
        "production": {
            "terminal": production_terminal,
            "answerability_status": answerability.status,
            "candidate_count": len(candidates),
            "evidence_count": len(evidence),
            "top_rerank_score": top_score,
            "candidate_identity_digest": _digest_ids(candidates),
            "candidate_set_digest": _digest_ids(candidates, sorted_ids=True),
            "top1_candidate_digest": _digest(candidates[0]) if candidates else None,
            "evidence_identity_digest": _digest_ids(evidence),
            "latency_ms": production_latency_ms,
        },
        "shadow": {
            "terminal": shadow["shadow_terminal"],
            "answerability_status": shadow["shadow_answerability_status"],
            "top_rerank_score": shadow["shadow_top_rerank_score"],
            "llm_invocation_recommended": shadow["gate"].get("llm_invocation_recommended"),
            "shadow_role": shadow["gate"].get("shadow_role"),
            "reason_code": shadow["gate"].get("reason_code"),
            "ambiguity_detected": shadow["gate"].get("ambiguity_detected"),
            "ambiguity_context_available": shadow["ambiguity_context_available"],
            "controller_call_count": shadow["controller_call_count"],
            "recovery_policy_called": shadow["recovery_policy_called"],
            "recovery_selected": shadow["recovery_selected"],
            "recovery_failure_code": shadow["recovery_failure_code"],
            "recovery_diagnostics": shadow.get("recovery_diagnostics", []),
            "recovery_improved": shadow["recovery_improved"],
            "recovery_harmed": shadow["recovery_harmed"],
            "veto_invoked": shadow["veto_invoked"],
            "veto_valid": shadow["veto_valid"],
            "veto_decision": shadow["veto_decision"],
            "veto_failure_code": shadow["veto_failure_code"],
            "conflict_reason_codes": list(shadow["conflict_reason_codes"]),
            "conflict_gate_version": shadow.get("conflict_gate_version", "opk-rag.task0249.conflict-gate-v2.v1"),
            "candidate_overlap": shadow["candidate_overlap"],
            "evidence_overlap": shadow["evidence_overlap"],
            "candidate_count_delta": shadow["candidate_count_delta"],
            "evidence_count_delta": shadow["evidence_count_delta"],
            "controller_latency_ms": shadow["controller_latency_ms"],
            "controller_input_tokens": shadow["controller_input_tokens"],
            "controller_output_tokens": shadow["controller_output_tokens"],
            "provider_requests": shadow["provider_requests"],
            "provider_responses": shadow["provider_responses"],
            "provider_valid_decisions": shadow["provider_valid_decisions"],
            "recovery_provider_requests": shadow.get("recovery_provider_requests", 0),
            "recovery_transport_retry_count": shadow.get("recovery_transport_retry_count", 0),
            "recovery_structural_repair_count": shadow.get("recovery_structural_repair_count", 0),
            "veto_provider_requests": shadow.get("veto_provider_requests", 0),
            "veto_transport_retry_count": shadow.get("veto_transport_retry_count", 0),
            "veto_structural_repair_count": shadow.get("veto_structural_repair_count", 0),
            "elapsed_shadow_ms": shadow["elapsed_shadow_ms"],
        },
        "authority": {
            "shadow_authoritative": False,
            "production_mutation_allowed": False,
            "llm_finish_authority": False,
            "abstain_to_finish_override_allowed": False,
            "max_graph_hop": 1,
        },
    }


def maybe_record_live_shadow_observation(
    *,
    query: str,
    search_response: Any,
    answerability: Any,
    binding: Any,
    execution_scope: str,
    source: str = "direct_user_request",
    production_latency_ms: float | None = None,
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> Mapping[str, Any] | None:
    config = load_live_shadow_config(env, root=root)
    if not config.enabled:
        return None
    try:
        record = build_live_observation_record(
            query=query,
            search_response=search_response,
            answerability=answerability,
            binding=binding,
            execution_scope=execution_scope,
            source=source,
            production_latency_ms=production_latency_ms,
        )
        stored = LiveShadowObservationStore(config).append(record)
        return {
            "schema_version": LIVE_SHADOW_SCHEMA_VERSION,
            "recorded": stored,
            "traffic_class": record["traffic_class"],
            "live_traffic_eligible": record["live_traffic_eligible"],
            "observation_digest": record["observation_digest"],
            "raw_query_persisted": False,
            "raw_provider_output_persisted": False,
            "shadow_authoritative": False,
            "candidate_decision_effect": "none",
        }
    except Exception as exc:
        # Live-shadow failure is observational only and may never fail Production Search/Ask.
        return {
            "schema_version": LIVE_SHADOW_SCHEMA_VERSION,
            "recorded": False,
            "traffic_class": classify_traffic(execution_scope=execution_scope, source=source)[0],
            "live_traffic_eligible": classify_traffic(execution_scope=execution_scope, source=source)[1],
            "shadow_failure": True,
            "failure_code": f"live_shadow_{type(exc).__name__}",
            "raw_query_persisted": False,
            "raw_provider_output_persisted": False,
            "shadow_authoritative": False,
            "candidate_decision_effect": "none",
        }


def reranker_stability_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("live_traffic_eligible") is True]
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible:
        by_query.setdefault(str(row.get("query_digest") or ""), []).append(row)
    repeated = {key: group for key, group in by_query.items() if len(group) >= 2 and key}
    same_identity_groups = 0
    material_score_groups = 0
    top1_change_groups = 0
    topk_set_change_groups = 0
    rank_change_groups = 0
    max_score_delta = 0.0
    for group in repeated.values():
        production = [dict(row.get("production") or {}) for row in group]
        identities = {str(item.get("candidate_identity_digest") or "") for item in production}
        sets = {str(item.get("candidate_set_digest") or "") for item in production}
        top1 = {str(item.get("top1_candidate_digest") or "") for item in production}
        scores = [float(item["top_rerank_score"]) for item in production if item.get("top_rerank_score") is not None]
        delta = max(scores) - min(scores) if len(scores) >= 2 else 0.0
        max_score_delta = max(max_score_delta, delta)
        if len(identities) == 1:
            same_identity_groups += 1
            if delta >= MATERIAL_RERANK_SCORE_DELTA:
                material_score_groups += 1
        if len(top1) > 1:
            top1_change_groups += 1
        if len(sets) > 1:
            topk_set_change_groups += 1
        if len(sets) == 1 and len(identities) > 1:
            rank_change_groups += 1
    return {
        "schema_version": "opk-rag.task0251.reranker-runtime-stability.v1",
        "material_score_delta_threshold": MATERIAL_RERANK_SCORE_DELTA,
        "eligible_live_query_count": len(eligible),
        "repeated_query_group_count": len(repeated),
        "same_candidate_identity_repeat_group_count": same_identity_groups,
        "same_candidate_identity_material_score_delta_count": material_score_groups,
        "reranker_runtime_variability_rate": material_score_groups / max(1, same_identity_groups),
        "top1_identity_change_group_count": top1_change_groups,
        "top_k_set_change_group_count": topk_set_change_groups,
        "rank_change_with_same_set_group_count": rank_change_groups,
        "max_observed_same_query_score_delta": max_score_delta,
        "task0250_known_risk": "reranker_score_variability_without_candidate_or_evidence_identity_change",
    }
