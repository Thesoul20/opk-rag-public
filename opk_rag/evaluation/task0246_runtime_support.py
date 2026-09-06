from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from opk_rag.runtime.dotenv import load_project_env
from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.guard import AgentGuard
from opk_rag.agentic_v2.policy_provider import OpenAICompatibleV2PolicyProvider
from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime
from opk_rag.agentic_v2.runtime import AgenticV2Runtime
from opk_rag.agentic_v2.runtime_context import AgentRuntimeContext
from opk_rag.agentic_v2.state_transition import AgentStateTransitionEngine
from opk_rag.agentic_v2.state_transition_contracts import AgentTransitionObservationSignals
from opk_rag.agentic_v2.tool_contracts import AgentToolResult
from opk_rag.agentic_v2.tool_executor import GuardedToolExecutor
from opk_rag.agentic_v2.tools import AgentToolContext, ExistingSearchBinding, default_agent_tool_registry
from opk_rag.core_tools.tools import assess_answerability, search_knowledge_base
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.reranking.config import load_reranker_config
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.evaluation import task0217_public_production_search_guarded_structure_aware_graph_v1_runtime_integration as task0217

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuleGovernedResult:
    status: str
    candidate_count: int
    evidence_count: int
    source_count: int
    answerability_status: str
    answerability_reason_code: str
    structure_lane_invoked: bool
    graph_activated: bool
    graph_hop_depth: int
    top_rerank_score: float | None
    latency_ms: float


class AnswerabilityAwareSearchAdapter:
    """Evaluation-only adapter over the same governed Search + Answerability authorities.

    It keeps SearchResponse/Answerability only in process memory and exposes to AgentState
    only the already-versioned AgentTransitionObservationSignals contract. Gold labels are
    never read by this adapter.
    """
    def __init__(self, binding: ExistingSearchBinding) -> None:
        self.binding = binding
        self.call_count = 0
        self.signals_by_execution_id: dict[str, AgentTransitionObservationSignals] = {}
        self.last_rule_result: RuleGovernedResult | None = None

    def execute_retrieval(self, *, action: AgentAction, run_id: str, execution_id: str) -> AgentToolResult:
        started = perf_counter(); self.call_count += 1
        response, answerability, rule = self._search(action)
        self.last_rule_result = rule
        guard = dict(response.guard_trace or {}); graph = dict(response.graph_trace or {})
        status = "success"; failure_code = None
        if action.action == "structure_search" and guard.get("structure_lane_invoked") is not True:
            status = "no_result"
        if action.action == "graph_search":
            if int(graph.get("hop_depth") or 0) != 1:
                status = "failed"; failure_code = "graph_hop_contract_violation"
            elif graph.get("graph_activated") is not True:
                status = "no_result"
        candidate_ids = tuple(str(item.chunk_id) for item in response.results)
        evidence_items = response.evidence_bundle.items if response.evidence_bundle is not None else ()
        evidence_ids = tuple(str(item.chunk_id) for item in evidence_items)
        signals = response.evidence_signals
        source_count = int(signals.selected_source_document_count) if signals is not None else len({str(x.document_id) for x in evidence_items})
        top = signals.top_reranker_score if signals is not None else None
        self.signals_by_execution_id[execution_id] = AgentTransitionObservationSignals(
            run_id=run_id, execution_id=execution_id, action=action.action,
            source_count=source_count, top_rerank_score=top,
            answerability_status=answerability.status,
            structure_context_available=guard.get("structure_lane_invoked") is True,
            graph_relation_available=graph.get("graph_activated") is True,
        )
        return AgentToolResult(
            run_id=run_id, execution_id=execution_id, action=action.action, status=status,
            candidate_count=len(candidate_ids), evidence_count=len(evidence_ids), candidate_ids=candidate_ids, evidence_ids=evidence_ids,
            result_summary="Governed Search + Answerability evaluation completed.", failure_code=failure_code,
        )

    def observation_signals(self, _state, _action, tool_result: AgentToolResult) -> AgentTransitionObservationSignals | None:
        return self.signals_by_execution_id.get(tool_result.execution_id)

    def run_rule_governed(self, query: str) -> RuleGovernedResult:
        action = _hybrid_action(query)
        _response, _answerability, rule = self._search(action)
        self.last_rule_result = rule
        return rule

    def _search(self, action: AgentAction):
        args=action.arguments; cfg=self.binding.search_config
        updates: dict[str, Any] = {"mode":"hybrid"}
        if action.action == "hybrid_search":
            top_k=args.top_k; updates.update(top_k=top_k,candidate_k=max(cfg.candidate_k,cfg.rerank_top_n,top_k),bm25_candidate_k=max(cfg.bm25_candidate_k,cfg.rerank_top_n,top_k))
        cfg=replace(cfg,**updates); started=perf_counter()
        response,_=search_knowledge_base(database_url=self.binding.database_url,knowledge_base_id=self.binding.knowledge_base_id,query=args.query,provider=self.binding.embedding_provider,embedding_config=self.binding.embedding_config,search_config=cfg,reranker_provider=self.binding.reranker_provider,context_token_counter=self.binding.context_token_counter)
        answerability,_=assess_answerability(search_response=response)
        evidence_items=response.evidence_bundle.items if response.evidence_bundle else ()
        sig=response.evidence_signals; guard=dict(response.guard_trace or {}); graph=dict(response.graph_trace or {})
        rule=RuleGovernedResult(
            status="finished" if answerability.status in {"answerable","partially_answerable"} else "abstained",
            candidate_count=len(response.results), evidence_count=len(evidence_items),
            source_count=int(sig.selected_source_document_count) if sig is not None else len({str(x.document_id) for x in evidence_items}),
            answerability_status=answerability.status, answerability_reason_code=answerability.reason_code,
            structure_lane_invoked=guard.get("structure_lane_invoked") is True, graph_activated=graph.get("graph_activated") is True,
            graph_hop_depth=int(graph.get("hop_depth") or 0), top_rerank_score=sig.top_reranker_score if sig is not None else None,
            latency_ms=max(0.0,(perf_counter()-started)*1000.0),
        )
        return response,answerability,rule


def build_binding() -> ExistingSearchBinding:
    load_project_env(ROOT); os.environ["OPK_RAG_VECTOR_BACKEND"]="qdrant"
    env=task0217._env(); kb=task0217.resolve_showcase_kb(env); emb=load_embedding_config()
    return ExistingSearchBinding(
        database_url=os.environ["DATABASE_URL"], knowledge_base_id=UUID(str(kb["knowledge_base_id"])),
        embedding_provider=QwenLocalEmbeddingProvider(emb), embedding_config=emb, search_config=load_vector_search_config(),
        reranker_provider=BgeLocalRerankerProvider(load_reranker_config()), context_token_counter=QwenContextTokenCounter(emb),
    )


def build_real_runtime(*, prompt_version: str, temperature: float = 0.0):
    binding=build_binding(); adapter=AnswerabilityAwareSearchAdapter(binding)
    provider=OpenAICompatibleV2PolicyProvider(timeout_seconds=60,temperature=temperature,max_tokens=512)
    policy=LLMAgentPolicyRuntime(provider=provider,prompt_version=prompt_version)
    runtime=AgenticV2Runtime(policy=policy,guard=AgentGuard(),tool_executor=GuardedToolExecutor(registry=default_agent_tool_registry()),state_transition=AgentStateTransitionEngine())
    return runtime,adapter,provider


def context_for(run_id: str, adapter: AnswerabilityAwareSearchAdapter) -> AgentRuntimeContext:
    return AgentRuntimeContext(tool_context=AgentToolContext(run_id=run_id,retrieval_adapter=adapter),observation_signal_resolver=adapter.observation_signals)


def _hybrid_action(query: str) -> AgentAction:
    from opk_rag.agentic_v2.action import validate_agent_action
    return validate_agent_action({"contract_version":"opk-rag.agentic-v2.action.v1","action":"hybrid_search","arguments":{"query":query,"top_k":10},"reason_code":"initial_retrieval_needed"})
