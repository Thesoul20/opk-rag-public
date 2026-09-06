from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.models import AnswerGeneratorProvider
from opk_rag.core_tools.contracts import TOOL_TRACE_SCHEMA_VERSION, CoreToolError, ToolInvocation, ToolResult, ToolTraceEvent
from opk_rag.core_tools.serialization import answer_response_payload, answerability_payload, grounding_payload, search_response_payload
from opk_rag.core_tools.tools import (
    assess_answerability,
    generate_grounded_answer,
    search_knowledge_base,
    verify_grounding,
)
from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingProvider
from opk_rag.reranking.provider import RerankerProvider
from opk_rag.search.config import VectorSearchConfig


@dataclass
class CoreRagToolRuntime:
    database_url: str
    knowledge_base_id: UUID
    embedding_provider: EmbeddingProvider | None
    embedding_config: EmbeddingConfig
    search_config: VectorSearchConfig
    answer_provider: AnswerGeneratorProvider
    answer_config: AnswerGenerationConfig
    reranker_provider: RerankerProvider | None = None
    context_token_counter: Any = None
    runtime_identity: dict[str, Any] = field(default_factory=dict)
    trace_events: list[ToolTraceEvent] = field(default_factory=list)

    def identity(self) -> dict[str, Any]:
        identity = {
            "search_mode": self.search_config.mode,
            "search_top_k": self.search_config.top_k,
            "search_candidate_k": self.search_config.candidate_k,
            "rerank_enabled": self.search_config.rerank_enabled,
            "embedding_model_id": self.embedding_provider.model_id if self.embedding_provider is not None else None,
            "answer_provider_id": self.answer_provider.provider_id,
            "answer_model_id": self.answer_provider.model_id,
            "answer_model_revision": self.answer_provider.model_revision,
            "answer_prompt_version": self.answer_config.prompt_version,
            "answer_output_schema_version": self.answer_config.output_schema_version,
            "grounding_enabled": self.answer_config.grounding.enabled,
        }
        return {**identity, **self.runtime_identity}


def run_toolized_core_pipeline(
    *,
    query: str,
    runtime: CoreRagToolRuntime,
):
    search_response, search_payload = _invoke(
        runtime,
        "search_knowledge_base",
        {"knowledge_base_id": str(runtime.knowledge_base_id), "query": query},
        lambda: search_knowledge_base(
            database_url=runtime.database_url,
            knowledge_base_id=runtime.knowledge_base_id,
            query=query,
            provider=runtime.embedding_provider,
            embedding_config=runtime.embedding_config,
            search_config=runtime.search_config,
            reranker_provider=runtime.reranker_provider,
            context_token_counter=runtime.context_token_counter,
        ),
    )
    answerability, answerability_payload_value = _invoke(
        runtime,
        "assess_answerability",
        {"search_response": search_response_payload(search_response)},
        lambda: assess_answerability(search_response=search_response, config=runtime.answer_config.answerability),
    )
    if not answerability.answerable:
        from opk_rag.answer.service import answer_knowledge_base_with_decision

        answer = answer_knowledge_base_with_decision(
            search_response,
            answerability=answerability,
            provider=runtime.answer_provider,
            config=runtime.answer_config,
        )
        _record(
            runtime,
            "generate_grounded_answer",
            {"search_response": search_response_payload(search_response), "answerability": answerability_payload(answerability)},
            {"answer_response": answer_response_payload(answer)},
        )
        return answer
    answer, answer_payload = _invoke(
        runtime,
        "generate_grounded_answer",
        {"search_response": search_response_payload(search_response), "answerability": answerability_payload_value},
        lambda: generate_grounded_answer(
            search_response=search_response,
            answerability=answerability,
            provider=runtime.answer_provider,
            config=runtime.answer_config,
        ),
    )
    _invoke(
        runtime,
        "verify_grounding",
        {
            "answer_hash": answer_payload["answer_hash"],
            "citations": [citation.citation_id for citation in answer.citations],
            "trusted_evidence": search_payload["trusted_evidence"],
        },
        lambda: verify_grounding(
            answer_text=answer.answer,
            citations=[citation.citation_id for citation in answer.citations],
            evidence_bundle=search_response.evidence_bundle,
            config=runtime.answer_config.grounding,
        ),
    )
    return answer


def _invoke(runtime: CoreRagToolRuntime, tool_name, arguments: dict[str, Any], call):
    try:
        value, payload = call()
    except CoreToolError as exc:
        result = ToolResult(tool_name=tool_name, ok=False, error=exc.error)
        invocation = ToolInvocation(tool_name=tool_name, arguments=arguments, invocation_id=str(uuid4()), runtime_identity=runtime.identity())
        runtime.trace_events.append(ToolTraceEvent(schema_version=TOOL_TRACE_SCHEMA_VERSION, invocation=invocation, result=result))
        raise
    _record(runtime, tool_name, arguments, payload)
    return value, payload


def _record(runtime: CoreRagToolRuntime, tool_name, arguments: dict[str, Any], payload: dict[str, Any]) -> None:
    result = ToolResult(tool_name=tool_name, ok=True, output=payload)
    invocation = ToolInvocation(tool_name=tool_name, arguments=arguments, invocation_id=str(uuid4()), runtime_identity=runtime.identity())
    runtime.trace_events.append(ToolTraceEvent(schema_version=TOOL_TRACE_SCHEMA_VERSION, invocation=invocation, result=result))
