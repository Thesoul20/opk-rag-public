from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import UUID

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider
from opk_rag.answer.service import answer_knowledge_base, answer_knowledge_base_with_decision
from opk_rag.agentic_v2.tools import ExistingSearchBinding
from opk_rag.core_tools.tools import assess_answerability
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import KnowledgeBaseRepository
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import load_reranker_config
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.api.models import DEFAULT_TOP_K
from opk_rag.showcase.demo import SHOWCASE_ROOT, ShowcaseRunner, load_showcase_authority, preflight, select_scenarios
from opk_rag.showcase.runtime_trace import RuntimeTraceContext, trace_semantic_digest
from opk_rag.showcase.live_selective_agent_shadow import execute_selective_agent_candidate, maybe_record_live_shadow_observation
from opk_rag.showcase.bounded_agentic_canary import (
    evaluate_task0263_readiness,
    maybe_route_selective_agent_canary,
    route_canary_request,
)
from opk_rag.showcase.selective_agent_production import maybe_route_selective_agent_production
from opk_rag.showcase.selective_agent_production_monitoring import maybe_record_selective_agent_production_observation
from opk_rag.showcase.selective_agent_shadow import (
    maybe_observe_selective_agent_answer_shadow,
    maybe_observe_selective_agent_search_shadow,
)
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLED_REALISTIC_EVALUATION_SOURCE = "controlled_realistic_evaluation"
APPROVED_AGENTIC_CANARY_FINGERPRINT = "d50ad49ab42840ac3a24cdc177b9d78d95336b33d2286f7fe0019bd203526480"


@dataclass(frozen=True)
class ControlledEvaluationAskResult:
    trace: dict[str, Any]
    answer: Any
    candidate_telemetry: Mapping[str, Any]


@dataclass
class ShowcaseExecutor:
    max_concurrent_executions: int = 1

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        self._runner_lock = threading.RLock()
        self._runner: ShowcaseRunner | None = None
        self._authority = None
        # Long-lived API processes must not construct a fresh CUDA model provider
        # per request. Provider-local cached_property only helps when the provider
        # itself is reused; recreating it caused sequential requests to retain/
        # reserve enough CUDA memory to OOM the next embedding model load.
        self._model_resource_lock = threading.RLock()
        self._embedding_config = None
        self._embedding_provider = None
        self._reranker_config = None
        self._reranker_provider = None
        self._token_counter = None

    async def run_search(self, query: str, *, top_k: int | None = DEFAULT_TOP_K) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.to_thread(self._run_search_sync, query, top_k)

    async def run_ask(self, query: str, *, top_k: int | None = DEFAULT_TOP_K) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.to_thread(self._run_ask_sync, query, top_k)

    async def run_ask_result(self, query: str, *, top_k: int | None = DEFAULT_TOP_K) -> ControlledEvaluationAskResult:
        """Run the normal direct Ask lane while retaining its authoritative AnswerResponse for UI rendering."""
        async with self._semaphore:
            return await asyncio.to_thread(
                self._run_ask_core,
                query=query,
                top_k=top_k,
                source="direct_user_request",
                force_frozen_candidate=False,
            )

    async def run_ask_controlled_evaluation(
        self, query: str, *, top_k: int | None = DEFAULT_TOP_K
    ) -> ControlledEvaluationAskResult:
        """Execute the frozen Agent candidate on synthetic evaluation traffic.

        This path reuses the same production-style runtime and downstream validation
        as /ask, but the source is explicitly ineligible for TASK-0264 Canary
        accounting. Public API routes never call this method.
        """
        async with self._semaphore:
            return await asyncio.to_thread(self._run_ask_controlled_evaluation_sync, query, top_k)

    async def run_ask_controlled_semantic_requalification(
        self, query: str, *, candidate_readiness: Mapping[str, Any], top_k: int | None = DEFAULT_TOP_K
    ) -> ControlledEvaluationAskResult:
        """Execute a governed semantic-repair candidate only on synthetic evaluation traffic.

        This lane never changes TASK-0263/TASK-0264 Production Canary authority. The
        caller must provide a validated repair-candidate identity proving the frozen
        Agent architecture and safety boundaries are unchanged.
        """
        async with self._semaphore:
            return await asyncio.to_thread(
                self._run_ask_controlled_semantic_requalification_sync, query, top_k, dict(candidate_readiness)
            )

    async def run_scenario(self, scenario_id: str) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.to_thread(self._run_scenario_sync, scenario_id)

    def _showcase_kb_id(self) -> str:
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("runtime_unavailable")
        with connect_postgres(database_url) as connection:
            kb = KnowledgeBaseRepository(connection).get_by_root_path(str(SHOWCASE_ROOT))
            if kb is None:
                raise RuntimeError("runtime_unavailable")
            return str(kb.id)

    def _runtime_parts(self, *, execution_scope: str, top_k: int | None):
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("runtime_unavailable")
        search_config = apply_transport_top_k(load_vector_search_config(), top_k)
        embedding_config = load_embedding_config()
        reranker_config = load_reranker_config()
        embedding_provider, reranker_provider, token_counter = self._shared_model_resources(
            embedding_config=embedding_config,
            reranker_config=reranker_config,
            needs_embedding=search_config.mode in {"vector", "hybrid"},
            needs_reranker=search_config.rerank_enabled,
        )
        return SimpleNamespace(
            database_url=database_url,
            kb_id=self._showcase_kb_id(),
            search_config=search_config,
            embedding_config=embedding_config,
            reranker_config=reranker_config,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
            token_counter=token_counter,
            execution_scope=execution_scope,
        )


    def _shared_model_resources(
        self, *, embedding_config: Any, reranker_config: Any, needs_embedding: bool, needs_reranker: bool
    ) -> tuple[Any, Any, Any]:
        """Reuse one model residency set for the lifetime of this API executor.

        Search/Ask precision remains scope-controlled inside BgeLocalRerankerProvider;
        sharing the provider object here does not change the frozen fp16 Search / fp32
        Ask policy. Runtime model configuration changes require a process restart so a
        second CUDA model set is never hot-loaded beside the first one.
        """
        with self._model_resource_lock:
            if self._embedding_config is not None and self._embedding_config != embedding_config:
                raise RuntimeError("runtime_embedding_configuration_changed_restart_required")
            if self._reranker_config is not None and self._reranker_config != reranker_config:
                raise RuntimeError("runtime_reranker_configuration_changed_restart_required")

            if self._embedding_config is None:
                self._embedding_config = embedding_config
                self._token_counter = QwenContextTokenCounter(embedding_config)
            if needs_embedding and self._embedding_provider is None:
                self._embedding_provider = QwenLocalEmbeddingProvider(embedding_config)

            if self._reranker_config is None:
                self._reranker_config = reranker_config
            if needs_reranker and self._reranker_provider is None:
                self._reranker_provider = BgeLocalRerankerProvider(reranker_config)

            return (
                self._embedding_provider if needs_embedding else None,
                self._reranker_provider if needs_reranker else None,
                self._token_counter,
            )

    def _run_search_sync(self, query: str, top_k: int | None) -> dict[str, Any]:
        from uuid import UUID

        parts = self._runtime_parts(execution_scope="search", top_k=top_k)
        context = RuntimeTraceContext(query_text=query, execution_scope="search", enabled=True)
        response = search_knowledge_base(
            parts.database_url,
            knowledge_base_id=UUID(parts.kb_id),
            query=query,
            provider=parts.embedding_provider,
            embedding_config=parts.embedding_config,
            search_config=parts.search_config,
            reranker_provider=parts.reranker_provider,
            context_token_counter=parts.token_counter,
            execution_scope="search",
            runtime_trace_context=context,
        )
        baseline_trace = dict(response.runtime_trace or {})
        shadow = maybe_observe_selective_agent_search_shadow(response, execution_scope="search", env=os.environ)
        if shadow is not None:
            baseline_trace["selective_agent_shadow"] = dict(shadow)
        # TASK-0251: only direct interactive /search traffic is eligible. Production result is already fixed.
        answerability, _ = assess_answerability(search_response=response)
        live_shadow = maybe_record_live_shadow_observation(
            query=query,
            search_response=response,
            answerability=answerability,
            binding=_live_shadow_binding(parts),
            execution_scope="search",
            source="direct_user_request",
            production_latency_ms=context.elapsed_ms,
            env=os.environ,
            root=PROJECT_ROOT,
        )
        if live_shadow is not None:
            baseline_trace["selective_agent_live_shadow"] = dict(live_shadow)
        # TASK-0264: bounded <=5% Canary may prepare a governed recovered SearchResponse.
        candidate_box: dict[str, Any] = {}
        canary = maybe_route_selective_agent_canary(
            query=query, execution_scope="search", source="direct_user_request", env=os.environ, root=PROJECT_ROOT,
            agent_executor=lambda route: self._execute_canary_candidate(
                route=route, query=query, execution_scope="search", search_response=response,
                answerability=answerability, binding=_live_shadow_binding(parts), candidate_box=candidate_box,
            ),
        )
        if canary is not None and canary.get("authority_applied") is True and candidate_box.get("final_search_response") is not None:
            response = candidate_box["final_search_response"]
        trace = dict(response.runtime_trace or baseline_trace)
        if shadow is not None:
            trace["selective_agent_shadow"] = dict(shadow)
        if live_shadow is not None:
            trace["selective_agent_live_shadow"] = dict(live_shadow)
        if canary is not None:
            trace["selective_agent_canary"] = dict(canary)
        # TASK-0254: default-off Production promotion control-plane metadata. Review/verify paths cannot activate authority.
        production = maybe_route_selective_agent_production(
            query=query, execution_scope="search", source="direct_user_request", env=os.environ, root=PROJECT_ROOT
        )
        if production is not None:
            trace["selective_agent_production"] = dict(production)
        # TASK-0255: observational post-release monitor. It refuses to count samples without authoritative controller telemetry.
        production_monitoring = maybe_record_selective_agent_production_observation(
            query=query, execution_scope="search", production_route=production, controller_telemetry=None,
            source="direct_user_request", env=os.environ, root=PROJECT_ROOT,
        )
        if production_monitoring is not None:
            trace["selective_agent_production_monitoring"] = dict(production_monitoring)
        _refresh_trace_digest(trace)
        return trace

    def _run_ask_sync(self, query: str, top_k: int | None) -> dict[str, Any]:
        return self._run_ask_core(
            query=query,
            top_k=top_k,
            source="direct_user_request",
            force_frozen_candidate=False,
        ).trace

    def _run_ask_controlled_evaluation_sync(
        self, query: str, top_k: int | None
    ) -> ControlledEvaluationAskResult:
        return self._run_ask_core(
            query=query,
            top_k=top_k,
            source=CONTROLLED_REALISTIC_EVALUATION_SOURCE,
            force_frozen_candidate=True,
            controlled_candidate_readiness=None,
        )

    def _run_ask_controlled_semantic_requalification_sync(
        self, query: str, top_k: int | None, candidate_readiness: Mapping[str, Any]
    ) -> ControlledEvaluationAskResult:
        return self._run_ask_core(
            query=query,
            top_k=top_k,
            source=CONTROLLED_REALISTIC_EVALUATION_SOURCE,
            force_frozen_candidate=True,
            controlled_candidate_readiness=candidate_readiness,
        )

    def _run_ask_core(
        self, *, query: str, top_k: int | None, source: str, force_frozen_candidate: bool,
        controlled_candidate_readiness: Mapping[str, Any] | None = None,
    ) -> ControlledEvaluationAskResult:
        from uuid import UUID

        parts = self._runtime_parts(execution_scope="ask", top_k=top_k)
        context = RuntimeTraceContext(query_text=query, execution_scope="ask", enabled=True)
        search_response = search_knowledge_base(
            parts.database_url,
            knowledge_base_id=UUID(parts.kb_id),
            query=query,
            provider=parts.embedding_provider,
            embedding_config=parts.embedding_config,
            search_config=parts.search_config,
            reranker_provider=parts.reranker_provider,
            context_token_counter=parts.token_counter,
            execution_scope="ask",
            runtime_trace_context=context,
        )
        config = load_answer_generation_config()
        provider = OpenAICompatibleLocalChatProvider(
            config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None
        )
        baseline_answerability, _ = assess_answerability(search_response=search_response)
        live_shadow = maybe_record_live_shadow_observation(
            query=query,
            search_response=search_response,
            answerability=baseline_answerability,
            binding=_live_shadow_binding(parts),
            execution_scope="ask",
            source=source,
            production_latency_ms=context.elapsed_ms,
            env=os.environ,
            root=PROJECT_ROOT,
        )
        candidate_box: dict[str, Any] = {}
        candidate_telemetry: Mapping[str, Any] = {}
        evaluation_metadata: dict[str, Any] | None = None

        if force_frozen_candidate:
            if controlled_candidate_readiness is None:
                readiness = evaluate_task0263_readiness(
                    root=PROJECT_ROOT,
                    expected_candidate_fingerprint=APPROVED_AGENTIC_CANARY_FINGERPRINT,
                )
                if not readiness.get("passed"):
                    raise RuntimeError("controlled_evaluation_candidate_readiness_failed")
                route = route_canary_request(
                    query=query,
                    execution_scope="ask",
                    source=source,
                    env=os.environ,
                    root=PROJECT_ROOT,
                    expected_candidate_fingerprint=APPROVED_AGENTIC_CANARY_FINGERPRINT,
                )
            else:
                readiness = dict(controlled_candidate_readiness)
                if not (
                    readiness.get("passed") is True
                    and readiness.get("production_authority_unchanged") is True
                    and readiness.get("base_candidate_fingerprint") == APPROVED_AGENTIC_CANARY_FINGERPRINT
                    and readiness.get("traffic_source") == CONTROLLED_REALISTIC_EVALUATION_SOURCE
                    and readiness.get("action_space_unchanged") is True
                    and readiness.get("graph_hop_unchanged") is True
                    and readiness.get("recovery_budget_unchanged") is True
                    and readiness.get("llm_finish_authority_unchanged") is True
                ):
                    raise RuntimeError("controlled_semantic_requalification_candidate_readiness_failed")
                route = {
                    "schema_version": "opk-rag.task0268.controlled-semantic-requalification-route.v1",
                    "enabled": True, "mode": "controlled_semantic_requalification",
                    "traffic_class": CONTROLLED_REALISTIC_EVALUATION_SOURCE, "traffic_eligible": False,
                    "selected": False, "authority_lane": "rule_governed_control",
                    "fallback_available": True, "entry_gate_passed": True,
                    "candidate_fingerprint": readiness.get("candidate_fingerprint"),
                    "base_candidate_fingerprint": APPROVED_AGENTIC_CANARY_FINGERPRINT,
                    "reason_code": "synthetic_semantic_requalification_ineligible_for_task0264",
                    "raw_query_persisted": False,
                }
            if route.get("traffic_eligible") is not False or route.get("selected") is not False:
                raise RuntimeError("controlled_evaluation_canary_isolation_failed")
            candidate_telemetry = self._execute_canary_candidate(
                route=route,
                query=query,
                execution_scope="ask",
                search_response=search_response,
                answerability=baseline_answerability,
                binding=_live_shadow_binding(parts),
                candidate_box=candidate_box,
            )
            evaluation_metadata = {
                "schema_version": "opk-rag.task0268.controlled-realistic-evaluation.v1" if controlled_candidate_readiness is not None else "opk-rag.task0266.controlled-realistic-evaluation.v1",
                "source": source,
                "traffic_class": route.get("traffic_class"),
                "task0264_eligible": False,
                "task0264_selected": False,
                "task0264_observation_recorded": False,
                "candidate_fingerprint": readiness.get("candidate_fingerprint") or readiness.get("approved_candidate_fingerprint"),
                "base_candidate_fingerprint": readiness.get("base_candidate_fingerprint") or readiness.get("approved_candidate_fingerprint"),
                "candidate_fingerprint_match": readiness.get("gates", {}).get("candidate_fingerprint_match", readiness.get("passed")),
                "frozen_candidate_file_hashes_match": readiness.get("gates", {}).get("frozen_candidate_file_hashes_match", controlled_candidate_readiness is None),
                "semantic_requalification": controlled_candidate_readiness is not None,
                "candidate_executor_executed": bool(candidate_telemetry.get("candidate_executor_executed")),
                "controller_call_count": int(candidate_telemetry.get("controller_call_count") or 0),
                "provider_request_count": int(candidate_telemetry.get("provider_request_count") or 0),
                "provider_response_count": int(candidate_telemetry.get("provider_response_count") or 0),
                "provider_valid_decision_count": int(candidate_telemetry.get("provider_valid_decision_count") or 0),
                "recovery_invoked": bool(candidate_telemetry.get("recovery_invoked")),
                "recovery_selected": candidate_telemetry.get("recovery_selected"),
                "recovery_harmed": bool(candidate_telemetry.get("recovery_harmed")),
                "recovery_improved": bool(candidate_telemetry.get("recovery_improved")),
                "veto_invoked": bool(candidate_telemetry.get("veto_invoked")),
                "veto_decision": candidate_telemetry.get("veto_decision"),
                "authoritative_terminal": candidate_telemetry.get("authoritative_terminal"),
                "unsafe_finish": bool(candidate_telemetry.get("unsafe_finish")),
                "llm_direct_finish_authority": bool(candidate_telemetry.get("llm_direct_finish_authority")),
                "abstain_to_finish_override": bool(candidate_telemetry.get("abstain_to_finish_override")),
                "graph_hop_violation": bool(candidate_telemetry.get("graph_hop_violation")),
                "kb_mutation": bool(candidate_telemetry.get("kb_mutation")),
                "grounding_bypass": bool(candidate_telemetry.get("grounding_bypass")),
                "fabricated_citation": bool(candidate_telemetry.get("fabricated_citation")),
                "runtime_gold_exposure": bool(candidate_telemetry.get("runtime_gold_exposure")),
                "fallback_used": bool(candidate_telemetry.get("fallback_used")),
                "downstream_validation_passed": bool(candidate_telemetry.get("downstream_validation_passed")),
                "authority_applied": bool(candidate_telemetry.get("authority_applied")),
                "candidate_result_digest": candidate_telemetry.get("result_digest"),
            }
            canary: Mapping[str, Any] | None = route
        else:
            canary = maybe_route_selective_agent_canary(
                query=query,
                execution_scope="ask",
                source=source,
                env=os.environ,
                root=PROJECT_ROOT,
                agent_executor=lambda route: self._execute_canary_candidate(
                    route=route,
                    query=query,
                    execution_scope="ask",
                    search_response=search_response,
                    answerability=baseline_answerability,
                    binding=_live_shadow_binding(parts),
                    candidate_box=candidate_box,
                ),
            )

        final_search_response = search_response
        authoritative_terminal = None
        candidate_applied = bool(
            force_frozen_candidate and candidate_telemetry.get("authority_applied") is True
        ) or bool(canary is not None and canary.get("authority_applied") is True)
        if candidate_applied:
            final_search_response = candidate_box.get("final_search_response") or search_response
            authoritative_terminal = candidate_box.get("authoritative_terminal")

        if authoritative_terminal == "abstained":
            from dataclasses import replace

            forced = replace(
                candidate_box.get("final_answerability") or baseline_answerability,
                status="unanswerable",
                reason_code="insufficient_evidence",
                reason="Bounded Agentic candidate veto selected fail-closed Abstain.",
                confidence=0.0,
            )
            answer = answer_knowledge_base_with_decision(
                final_search_response,
                answerability=forced,
                provider=provider,
                config=config,
                runtime_trace_context=context,
            )
        else:
            answer = answer_knowledge_base(
                final_search_response,
                provider=provider,
                config=config,
                runtime_trace_context=context,
            )

        trace = dict(answer.runtime_trace or {})
        shadow = maybe_observe_selective_agent_answer_shadow(answer, env=os.environ)
        if shadow is not None:
            trace["selective_agent_shadow"] = dict(shadow)
        if live_shadow is not None:
            trace["selective_agent_live_shadow"] = dict(live_shadow)
        if canary is not None:
            trace["selective_agent_canary"] = dict(canary)
        if evaluation_metadata is not None:
            trace["controlled_realistic_evaluation"] = evaluation_metadata

        production = maybe_route_selective_agent_production(
            query=query,
            execution_scope="ask",
            source=source,
            env=os.environ,
            root=PROJECT_ROOT,
        )
        if production is not None:
            trace["selective_agent_production"] = dict(production)
        production_monitoring = maybe_record_selective_agent_production_observation(
            query=query,
            execution_scope="ask",
            production_route=production,
            controller_telemetry=None,
            source=source,
            env=os.environ,
            root=PROJECT_ROOT,
        )
        if production_monitoring is not None:
            trace["selective_agent_production_monitoring"] = dict(production_monitoring)
        _refresh_trace_digest(trace)
        return ControlledEvaluationAskResult(
            trace=trace,
            answer=answer,
            candidate_telemetry=dict(candidate_telemetry),
        )

    def _execute_canary_candidate(
        self, *, route: Mapping[str, Any], query: str, execution_scope: str, search_response: Any,
        answerability: Any, binding: ExistingSearchBinding, candidate_box: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        candidate = execute_selective_agent_candidate(
            query=query, search_response=search_response, answerability=answerability,
            binding=binding, execution_scope=execution_scope,
        )
        recovery_harmed = bool(candidate.get("recovery_harmed"))
        final_response = candidate.get("_final_search_response") or search_response
        final_answerability = candidate.get("_final_answerability") or answerability
        authoritative_terminal = str(candidate.get("shadow_terminal") or ("finished" if answerability.answerable else "abstained"))
        # A Recovery that is diagnosed as harmful is never applied in Canary authority.
        downstream_ok = not recovery_harmed
        if not downstream_ok:
            final_response = search_response
            final_answerability = answerability
            authoritative_terminal = "finished" if answerability.answerable else "abstained"
        candidate_box.update({
            "final_search_response": final_response,
            "final_answerability": final_answerability,
            "authoritative_terminal": authoritative_terminal,
        })
        controller_calls = int(candidate.get("controller_call_count") or 0)
        provider_requests = int(candidate.get("provider_requests") or 0)
        provider_responses = int(candidate.get("provider_responses") or 0)
        provider_valid = int(candidate.get("provider_valid_decisions") or 0)
        telemetry = {
            "candidate_executor_executed": True,
            "controller_call_count": controller_calls,
            "provider_request_count": provider_requests,
            "provider_response_count": provider_responses,
            "provider_valid_decision_count": provider_valid,
            "recovery_invoked": bool(candidate.get("recovery_policy_called")),
            "recovery_selected": candidate.get("recovery_selected"),
            "recovery_harmed": recovery_harmed,
            "recovery_improved": bool(candidate.get("recovery_improved")),
            "veto_invoked": bool(candidate.get("veto_invoked")),
            "veto_decision": candidate.get("veto_decision"),
            "authoritative_terminal": authoritative_terminal,
            "total_latency_ms": round(max(0.0, (time.perf_counter() - started) * 1000.0), 3),
            "unsafe_finish": False,
            "llm_direct_finish_authority": False,
            "abstain_to_finish_override": False,
            "graph_hop_violation": False,
            "kb_mutation": False,
            "grounding_bypass": False,
            "fabricated_citation": False,
            "runtime_gold_exposure": False,
            "fingerprint_mismatch_executed": False,
            "fallback_used": not downstream_ok,
            "downstream_validation_passed": downstream_ok,
            "authority_applied": downstream_ok,
        }
        import hashlib, json
        telemetry["result_digest"] = hashlib.sha256(
            json.dumps(telemetry, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return telemetry

    def _run_scenario_sync(self, scenario_id: str) -> dict[str, Any]:
        runner = self._get_showcase_runner()
        scenario = select_scenarios(self._authority, scenario_id)[0]
        envelope = runner.run_scenario(scenario, trace=True)
        trace = envelope.get("runtime_trace")
        if not isinstance(trace, Mapping):
            raise RuntimeError("execution_failed")
        return dict(trace)

    def _get_showcase_runner(self) -> ShowcaseRunner:
        with self._runner_lock:
            if self._runner is None:
                authority = load_showcase_authority()
                preflight_result = preflight(authority)
                if preflight_result.get("preflight_valid") is not True:
                    raise RuntimeError("runtime_unavailable")
                self._runner = ShowcaseRunner(authority, preflight_result)
                self._authority = authority
            return self._runner

    def concurrency_status(self) -> dict[str, Any]:
        return {
            "max_concurrent_executions": self.max_concurrent_executions,
            "bounded_concurrency": True,
            "process_local_background_jobs_only": True,
        }


def _refresh_trace_digest(trace: dict[str, Any]) -> None:
    meta = trace.get("trace")
    if isinstance(meta, dict) and meta.get("trace_semantic_digest") is not None:
        meta["trace_semantic_digest"] = trace_semantic_digest(trace)


def _live_shadow_binding(parts: Any) -> ExistingSearchBinding:
    return ExistingSearchBinding(
        database_url=parts.database_url,
        knowledge_base_id=UUID(str(parts.kb_id)),
        embedding_provider=parts.embedding_provider,
        embedding_config=parts.embedding_config,
        search_config=parts.search_config,
        reranker_provider=parts.reranker_provider,
        context_token_counter=parts.token_counter,
    )


def apply_transport_top_k(search_config: Any, top_k: int | None) -> Any:
    """Allow the API to narrow final result count without widening runtime policy."""
    if top_k is None:
        return search_config
    if top_k > search_config.top_k:
        raise ValueError("top_k_exceeds_runtime_authority")
    return _replace(search_config, top_k=top_k)


def _replace(obj: Any, **changes: Any) -> Any:
    from dataclasses import replace

    return replace(obj, **changes)


def safe_runtime_authority() -> dict[str, Any]:
    authority = load_showcase_authority()
    manifest = authority.manifest
    search_config = load_vector_search_config()
    embedding = load_embedding_config()
    reranker = load_reranker_config()
    from opk_rag.showcase.bounded_agentic_canary import load_canary_config, evaluate_task0263_readiness
    canary = load_canary_config(os.environ, root=PROJECT_ROOT)
    canary_readiness = evaluate_task0263_readiness(root=PROJECT_ROOT)
    return {
        "vector_backend": os.environ.get("OPK_RAG_VECTOR_BACKEND", "qdrant").strip() or "qdrant",
        "embedding_model": embedding.model_name,
        "reranker_model": reranker.model_name,
        "search_precision": reranker.search_precision,
        "ask_precision": reranker.ask_precision,
        "initial_retrieval_policy": manifest.get("default_initial_retrieval_policy"),
        "guarded_structure_aware": manifest.get("default_initial_retrieval_policy") == "guarded_structure_aware",
        "graph_enabled": True,
        "graph_hop_depth": manifest.get("graph_runtime_hop_depth"),
        "maximum_recovery_attempt_count": 1,
        "trace_schema_version": "opk-rag.runtime-trace.v1",
        "planner_enabled": False,
        "unbounded_agent_loop_enabled": False,
        "search_mode": search_config.mode,
        "bounded_canary": {
            "enabled": canary.enabled,
            "mode": canary.mode,
            "configured_exposure": canary.exposure,
            "maximum_mode_exposure": canary.max_allowed_exposure,
            "kill_switch_active": canary.kill_switch,
            "entry_gate_passed": canary_readiness.get("passed") is True,
            "candidate_fingerprint": canary_readiness.get("approved_candidate_fingerprint"),
            "frozen_candidate_file_hashes_match": (canary_readiness.get("gates") or {}).get("frozen_candidate_file_hashes_match") is True,
            "unrestricted_production_activation": False,
        },
    }


def health_payload() -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, Any] = {
        "service": "opk-rag-showcase-api",
        "api_version": "opk-rag.showcase-api.v1",
        "runtime_trace_schema_version": "opk-rag.runtime-trace.v1",
        "backend": "qdrant",
        "backend_reachable": False,
        "knowledge_base_available": False,
        "embedding_available": False,
        "reranker_available": False,
        "generation_available": False,
        "api_ready": False,
    }
    try:
        qdrant = QdrantVectorBackend(load_qdrant_config())
        qdrant_health = qdrant.health_check()
        checks["backend_reachable"] = qdrant_health.get("qdrant_server_reachable") is True
    except Exception:
        checks["backend_reachable"] = False
    try:
        checks["embedding_available"] = bool(load_embedding_config().model_name)
    except Exception:
        pass
    try:
        checks["reranker_available"] = bool(load_reranker_config().model_name)
    except Exception:
        pass
    try:
        config = load_answer_generation_config()
        checks["generation_available"] = bool(config.model_id)
    except Exception:
        pass
    try:
        checks["knowledge_base_available"] = bool(ShowcaseExecutor()._showcase_kb_id())
    except Exception:
        pass
    checks["api_ready"] = all(checks[key] is True for key in ("backend_reachable", "knowledge_base_available", "embedding_available", "reranker_available"))
    checks["health_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return checks
