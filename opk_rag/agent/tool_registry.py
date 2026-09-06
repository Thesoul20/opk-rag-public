from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from opk_rag.agent.contracts import AgentAction, AgentRuntimeConfig, AgentState
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.core_tools.contracts import CoreToolError
from opk_rag.core_tools.registry import default_tool_registry as core_default_tool_registry
from opk_rag.core_tools.runtime import CoreRagToolRuntime
from opk_rag.core_tools.serialization import answer_response_payload, answerability_payload, grounding_payload, search_response_payload
from opk_rag.core_tools.tools import assess_answerability, expand_document_section, generate_grounded_answer, search_knowledge_base, verify_grounding
from opk_rag.agent.transition import EXPLICIT_ELIGIBLE_ABSTAIN_REASONS, derive_generation_eligibility, resolve_available_actions

AGENT_TOOL_REGISTRY_VERSION = "opk-rag.agent-tool-registry.v1"

ACTION_TO_TOOL = {
    "search": "search_knowledge_base",
    "expand_evidence": "expand_document_section",
    "evaluate_answerability": "assess_answerability",
    "generate_grounded_answer": "generate_grounded_answer",
    "verify_grounding": "verify_grounding",
}

ALLOWED_ARGS = {
    "search": {"query", "top_k"},
    "expand_evidence": {"chunk_id", "max_items"},
    "evaluate_answerability": set(),
    "generate_grounded_answer": set(),
    "verify_grounding": set(),
    "finish_answer": set(),
    "finish_abstain": set(),
    "finish_failure": set(),
}

FORBIDDEN_ARGS = {
    "database_url",
    "sql",
    "provider",
    "answer_provider",
    "embedding_provider",
    "retriever",
    "disable_answerability",
    "disable_grounding",
    "grounding_enabled",
    "citation_rules",
    "system_prompt",
    "model_id",
    "api_key",
    "authorization",
}


class AgentCoreToolExecutor(Protocol):
    def execute(self, tool_name: str, arguments: dict[str, Any], memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ...


@dataclass(frozen=True)
class AgentToolDefinition:
    tool_name: str
    core_tool_name: str
    tool_contract_version: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_agent_actions: tuple[str, ...]
    timeout_policy: dict[str, Any]
    failure_mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "core_tool_name": self.core_tool_name,
            "tool_contract_version": self.tool_contract_version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "allowed_agent_actions": list(self.allowed_agent_actions),
            "timeout_policy": self.timeout_policy,
            "failure_mapping": self.failure_mapping,
        }


@dataclass
class AgentToolRegistry:
    definitions: tuple[AgentToolDefinition, ...]
    executor: AgentCoreToolExecutor
    version: str = AGENT_TOOL_REGISTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "core_tool_contract": core_default_tool_registry().to_dict(),
            "tools": [definition.to_dict() for definition in self.definitions],
        }

    def execute(
        self,
        action: AgentAction,
        state: AgentState,
        memory: dict[str, Any],
        config: AgentRuntimeConfig,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.validate_action(action, state, config)
        tool_name = ACTION_TO_TOOL[action.action]
        try:
            return self.executor.execute(tool_name, action.arguments, memory)
        except CoreToolError as exc:
            code = "core_tool_timeout" if exc.error.code == "provider_failure" and "timeout" in exc.error.message.lower() else "core_tool_failure"
            raise AgentRuntimeError(code, exc.error.message, origin=_core_origin(exc.error.code), detail=exc.error.to_dict()) from exc
        except TimeoutError as exc:
            raise AgentRuntimeError("core_tool_timeout", str(exc), origin="infrastructure") from exc
        except AgentRuntimeError:
            raise
        except Exception as exc:
            raise AgentRuntimeError("infrastructure_failure", f"Tool adapter failed: {type(exc).__name__}", origin="infrastructure") from exc

    def validate_action(self, action: AgentAction, state: AgentState, config: AgentRuntimeConfig) -> None:
        if action.contract_version != "opk-rag.agent-action.v1":
            raise AgentRuntimeError("agent_invalid_action", "Unknown agent action contract version.", origin="agent_runtime")
        if action.action not in ALLOWED_ARGS:
            raise AgentRuntimeError("agent_invalid_action", f"Unknown agent action: {action.action}", origin="agent_policy")
        unknown = set(action.arguments) - ALLOWED_ARGS[action.action]
        forbidden = set(action.arguments) & FORBIDDEN_ARGS
        if unknown or forbidden:
            raise AgentRuntimeError(
                "agent_tool_input_contract_failure",
                "Agent action arguments violate the closed input contract.",
                origin="tool_adapter",
                detail={"unknown_arguments": sorted(unknown), "forbidden_arguments": sorted(forbidden)},
            )
        if state.is_terminal and action.action not in {"finish_answer", "finish_abstain", "finish_failure"}:
            raise AgentRuntimeError("agent_state_transition_failure", "No tool may execute after terminal state.", origin="agent_runtime")
        if action.action in ACTION_TO_TOOL and state.tool_call_count >= config.max_tool_calls:
            raise AgentRuntimeError("agent_budget_exhausted", "Maximum tool call budget exhausted.", origin="agent_runtime")
        if action.action == "expand_evidence" and state.expansion_count >= config.max_expansions:
            raise AgentRuntimeError("agent_budget_exhausted", "Maximum evidence expansion budget exhausted.", origin="agent_runtime")
        available_actions = resolve_available_actions(state, config)
        if action.action not in available_actions:
            eligibility = derive_generation_eligibility(state, config)
            if (
                action.action == "finish_abstain"
                and eligibility.generation_eligible
                and action.reason_code in EXPLICIT_ELIGIBLE_ABSTAIN_REASONS
            ):
                return
            raise AgentRuntimeError(
                "agent_state_transition_failure",
                "Action is not available in the current runtime phase.",
                origin="agent_runtime",
                detail={
                    "action": action.action,
                    "available_actions": list(available_actions),
                    "generation_eligible": eligibility.generation_eligible,
                    "eligibility_reason_code": eligibility.eligibility_reason_code,
                },
            )
        if action.action == "search" and state.evidence_state.get("searched") and not config.model_planning_enabled:
            raise AgentRuntimeError("agent_state_transition_failure", "Initial search may execute only once.", origin="agent_runtime")
        if action.action == "search" and config.model_planning_enabled and state.search_round_count >= config.max_search_rounds:
            raise AgentRuntimeError("retrieval_round_budget_exhausted", "Maximum search round budget exhausted.", origin="agent_runtime")
        if action.action == "search":
            top_k = action.arguments.get("top_k")
            if not isinstance(action.arguments.get("query"), str) or not action.arguments["query"].strip():
                raise AgentRuntimeError("agent_tool_input_contract_failure", "search.query must be a non-empty string.", origin="tool_adapter")
            if top_k is not None and (not isinstance(top_k, int) or top_k < 1 or top_k > 20):
                raise AgentRuntimeError("agent_tool_input_contract_failure", "search.top_k must be between 1 and 20.", origin="tool_adapter")
        if action.action == "expand_evidence":
            if not state.evidence_state.get("searched"):
                raise AgentRuntimeError("agent_state_transition_failure", "Evidence expansion requires search evidence.", origin="agent_runtime")
            if not isinstance(action.arguments.get("chunk_id"), str) or not action.arguments["chunk_id"]:
                raise AgentRuntimeError("agent_tool_input_contract_failure", "expand_evidence.chunk_id is required.", origin="tool_adapter")
        if action.action == "evaluate_answerability" and not state.evidence_state.get("searched"):
            raise AgentRuntimeError("agent_state_transition_failure", "Answerability requires search evidence.", origin="agent_runtime")
        if action.action == "generate_grounded_answer":
            if not config.answerability_required or state.answerability_state.get("status") not in {"answerable", "partially_answerable"}:
                raise AgentRuntimeError("agent_state_transition_failure", "Generation requires answerability permission.", origin="agent_runtime")
        if action.action == "verify_grounding" and not state.generation_state.get("attempted"):
            raise AgentRuntimeError("agent_state_transition_failure", "Grounding verification requires generated answer.", origin="agent_runtime")
        if action.action == "finish_answer":
            if config.grounding_verification_required and state.verification_state.get("valid") is not True:
                raise AgentRuntimeError("agent_state_transition_failure", "Final answers require successful grounding verification.", origin="agent_runtime")


class LiveCoreToolExecutor:
    def __init__(self, runtime: CoreRagToolRuntime) -> None:
        self.runtime = runtime

    def execute(self, tool_name: str, arguments: dict[str, Any], memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if tool_name == "search_knowledge_base":
            search_config = self.runtime.search_config
            response, payload = search_knowledge_base(
                database_url=self.runtime.database_url,
                knowledge_base_id=self.runtime.knowledge_base_id,
                query=arguments["query"],
                provider=self.runtime.embedding_provider,
                embedding_config=self.runtime.embedding_config,
                search_config=search_config,
                reranker_provider=self.runtime.reranker_provider,
                context_token_counter=self.runtime.context_token_counter,
            )
            memory["search_response"] = response
            memory["evidence_bundle"] = response.evidence_bundle
            return payload, {"tool_name": tool_name, "trusted_evidence": payload.get("trusted_evidence"), "result_count": payload.get("result_count")}
        if tool_name == "expand_document_section":
            bundle = memory["evidence_bundle"]
            expanded, payload = expand_document_section(evidence_bundle=bundle, chunk_id=arguments["chunk_id"], max_items=arguments.get("max_items", 1))
            memory["evidence_bundle"] = expanded
            search_response = memory["search_response"]
            memory["search_response"] = replace(search_response, evidence_bundle=expanded, context_token_count=expanded.context_token_count)
            return payload, {"tool_name": tool_name, "trusted_evidence": payload.get("trusted_evidence")}
        if tool_name == "assess_answerability":
            decision, payload = assess_answerability(search_response=memory["search_response"], config=self.runtime.answer_config.answerability)
            memory["answerability"] = decision
            return payload, {"tool_name": tool_name, "answerability": payload}
        if tool_name == "generate_grounded_answer":
            answer, payload = generate_grounded_answer(
                search_response=memory["search_response"],
                answerability=memory["answerability"],
                provider=self.runtime.answer_provider,
                config=self.runtime.answer_config,
            )
            memory["answer_response"] = answer
            return payload, {"tool_name": tool_name, "answer_response": payload}
        if tool_name == "verify_grounding":
            answer = memory["answer_response"]
            grounding, payload = verify_grounding(
                answer_text=answer.answer,
                citations=[citation.citation_id for citation in answer.citations],
                evidence_bundle=memory["search_response"].evidence_bundle,
                config=self.runtime.answer_config.grounding,
            )
            memory["grounding"] = grounding
            return payload, {"tool_name": tool_name, "grounding": payload}
        raise AgentRuntimeError("agent_unknown_tool", f"Unknown registered tool: {tool_name}", origin="tool_adapter")


def default_agent_tool_registry(executor: AgentCoreToolExecutor) -> AgentToolRegistry:
    core = {tool.name: tool for tool in core_default_tool_registry().list_tools()}
    return AgentToolRegistry(
        definitions=(
            _definition("search", "search_knowledge_base", core["search_knowledge_base"], required=("query",), optional=("top_k",)),
            _definition("expand_evidence", "expand_document_section", core["expand_document_section"], required=("chunk_id",), optional=("max_items",)),
            _definition("evaluate_answerability", "assess_answerability", core["assess_answerability"], required=(), optional=()),
            _definition("generate_grounded_answer", "generate_grounded_answer", core["generate_grounded_answer"], required=(), optional=()),
            _definition("verify_grounding", "verify_grounding", core["verify_grounding"], required=(), optional=()),
        ),
        executor=executor,
    )


def _definition(agent_action: str, core_name: str, core_definition, *, required: tuple[str, ...], optional: tuple[str, ...]) -> AgentToolDefinition:
    return AgentToolDefinition(
        tool_name=agent_action,
        core_tool_name=core_name,
        tool_contract_version=core_definition.version,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": {name: {"description": f"{name} agent argument"} for name in (*required, *optional)},
        },
        output_schema=core_definition.output_schema,
        allowed_agent_actions=(agent_action,),
        timeout_policy={"timeout_seconds": 60, "on_timeout": "core_tool_timeout"},
        failure_mapping={
            "CoreToolError": "core_tool_failure",
            "TimeoutError": "core_tool_timeout",
            "Exception": "infrastructure_failure",
        },
    )


def _core_origin(code: str) -> str:
    if code == "provider_failure":
        return "provider"
    if code in {"knowledge_base_not_found", "document_not_found"}:
        return "infrastructure"
    return "core_rag"
