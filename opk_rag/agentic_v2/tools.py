from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol
from uuid import UUID

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.tool_contracts import AgentToolDefinition, AgentToolResult
from opk_rag.agentic_v2.tool_registry import AgentToolRegistry

TOOL_VERSION = "1.0.0"


@dataclass(frozen=True)
class EvidenceInspectionSnapshot:
    candidate_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_count: int = 0
    coverage: float | None = None


@dataclass(frozen=True)
class AgentToolContext:
    run_id: str
    execution_id: str = "pending"
    retrieval_adapter: "RetrievalAdapter | None" = None
    evidence_snapshot: EvidenceInspectionSnapshot | None = None


class RetrievalAdapter(Protocol):
    def execute_retrieval(self, *, action: AgentAction, run_id: str, execution_id: str) -> AgentToolResult:
        ...


@dataclass(frozen=True)
class ExistingSearchBinding:
    database_url: str
    knowledge_base_id: UUID
    embedding_provider: Any
    embedding_config: Any
    search_config: Any
    reranker_provider: Any = None
    context_token_counter: Any = None


class ExistingSearchServiceAdapter:
    """Thin read-only adapter over the existing governed Core RAG search capability."""

    def __init__(self, binding: ExistingSearchBinding) -> None:
        self.binding = binding
        self.call_count = 0

    def execute_retrieval(self, *, action: AgentAction, run_id: str, execution_id: str) -> AgentToolResult:
        from opk_rag.core_tools.tools import search_knowledge_base
        from opk_rag.search.config import VectorSearchConfig

        self.call_count += 1
        args = action.arguments
        cfg = self.binding.search_config
        updates: dict[str, Any] = {"mode": "hybrid"}
        if action.action == "hybrid_search":
            top_k = args.top_k
            candidate_k = max(cfg.candidate_k, cfg.rerank_top_n, top_k)
            updates.update(top_k=top_k, candidate_k=candidate_k, bm25_candidate_k=max(cfg.bm25_candidate_k, cfg.rerank_top_n, top_k))
        cfg = replace(cfg, **updates)
        response, _payload = search_knowledge_base(
            database_url=self.binding.database_url,
            knowledge_base_id=self.binding.knowledge_base_id,
            query=args.query,
            provider=self.binding.embedding_provider,
            embedding_config=self.binding.embedding_config,
            search_config=cfg,
            reranker_provider=self.binding.reranker_provider,
            context_token_counter=self.binding.context_token_counter,
        )
        if action.action == "structure_search":
            guard = dict(response.guard_trace or {})
            if guard.get("structure_lane_invoked") is not True:
                return _search_result(response, action, run_id, execution_id, status="no_result", summary="Existing governed Search did not activate Structure Recovery for this query.")
        if action.action == "graph_search":
            graph = dict(response.graph_trace or {})
            if int(graph.get("hop_depth") or 0) != 1:
                return AgentToolResult(run_id=run_id, execution_id=execution_id, action=action.action, status="failed", failure_code="graph_hop_contract_violation")
            if graph.get("graph_activated") is not True:
                return _search_result(response, action, run_id, execution_id, status="no_result", summary="Existing governed Search did not activate one-hop Graph Recovery for this query.")
        return _search_result(response, action, run_id, execution_id, status="success", summary="Existing governed Search capability completed through the shared reranking/evidence path.")


def _search_result(response, action: AgentAction, run_id: str, execution_id: str, *, status: str, summary: str) -> AgentToolResult:
    candidate_ids = tuple(str(result.chunk_id) for result in response.results)
    evidence_ids = tuple(str(item.chunk_id) for item in (response.evidence_bundle.items if response.evidence_bundle is not None else ()))
    return AgentToolResult(
        run_id=run_id, execution_id=execution_id, action=action.action, status=status,
        candidate_count=len(candidate_ids), evidence_count=len(evidence_ids), candidate_ids=candidate_ids, evidence_ids=evidence_ids,
        result_summary=summary,
    )


class RetrievalTool:
    def __init__(self, action: str, capability: str) -> None:
        self.definition = AgentToolDefinition(action=action, tool_name=f"agentic_v2.{action}", tool_version=TOOL_VERSION, capability=capability)

    def execute(self, *, action: AgentAction, context: AgentToolContext) -> AgentToolResult:
        if context.retrieval_adapter is None:
            return AgentToolResult(run_id=context.run_id, execution_id=context.execution_id, action=action.action, status="failed", failure_code="tool_runtime_unavailable")
        return context.retrieval_adapter.execute_retrieval(action=action, run_id=context.run_id, execution_id=context.execution_id)


class RewriteQueryTool:
    definition = AgentToolDefinition(action="rewrite_query", tool_name="agentic_v2.rewrite_query", tool_version=TOOL_VERSION, capability="Accept the already Guard-approved rewritten query without another model or retrieval call.")
    def execute(self, *, action: AgentAction, context: AgentToolContext) -> AgentToolResult:
        return AgentToolResult(run_id=context.run_id, execution_id=context.execution_id, action="rewrite_query", status="success", rewritten_query=action.arguments.query, result_summary="Guard-approved query rewrite intent accepted; AgentState remains unchanged.")


class InspectEvidenceTool:
    definition = AgentToolDefinition(action="inspect_evidence", tool_name="agentic_v2.inspect_evidence", tool_version=TOOL_VERSION, capability="Read only governed evidence identity/count summary.")
    def execute(self, *, action: AgentAction, context: AgentToolContext) -> AgentToolResult:
        snap = context.evidence_snapshot or EvidenceInspectionSnapshot()
        status = "success" if snap.evidence_ids else "no_result"
        return AgentToolResult(run_id=context.run_id, execution_id=context.execution_id, action="inspect_evidence", status=status,
            candidate_count=len(snap.candidate_ids), evidence_count=len(snap.evidence_ids), candidate_ids=snap.candidate_ids, evidence_ids=snap.evidence_ids,
            result_summary=f"Governed evidence summary: sources={snap.source_count}, coverage={'unavailable' if snap.coverage is None else round(snap.coverage, 4)}.")


class TerminalTool:
    def __init__(self, action: str) -> None:
        self.definition = AgentToolDefinition(action=action, tool_name=f"agentic_v2.{action}", tool_version=TOOL_VERSION, capability="Return terminal intent only; do not invoke answer generation.")
    def execute(self, *, action: AgentAction, context: AgentToolContext) -> AgentToolResult:
        return AgentToolResult(run_id=context.run_id, execution_id=context.execution_id, action=action.action, status="success", terminal_intent=action.action, result_summary=f"Terminal intent {action.action} accepted; downstream runtime not executed by Tool Executor.")


def default_agent_tool_registry() -> AgentToolRegistry:
    return AgentToolRegistry(tools=(
        RetrievalTool("hybrid_search", "Invoke existing governed hybrid Search capability; no direct Qdrant/SQL authority."),
        RetrievalTool("structure_search", "Invoke existing governed Search capability and require Structure Recovery lane observation."),
        RetrievalTool("graph_search", "Invoke existing governed Search capability and require authoritative one-hop Graph Recovery observation."),
        RewriteQueryTool(), InspectEvidenceTool(), TerminalTool("finish"), TerminalTool("abstain"),
    ))
