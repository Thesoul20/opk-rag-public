from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from opk_rag.agentic_v2.action import ActionName, AgentAction
from opk_rag.agentic_v2.tool_contracts import AgentToolDefinition, AgentToolResult

TOOL_REGISTRY_VERSION = "opk-rag.agentic-v2.tool-registry.v1"
FROZEN_TOOL_ACTIONS: tuple[ActionName, ...] = (
    "hybrid_search", "structure_search", "graph_search", "rewrite_query", "inspect_evidence", "finish", "abstain",
)


class AgentTool(Protocol):
    definition: AgentToolDefinition

    def execute(self, *, action: AgentAction, context) -> AgentToolResult:
        ...


class ToolRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentToolRegistry:
    tools: tuple[AgentTool, ...]
    version: str = TOOL_REGISTRY_VERSION

    def __post_init__(self) -> None:
        actions = tuple(tool.definition.action for tool in self.tools)
        if len(actions) != len(set(actions)):
            raise ToolRegistryError("duplicate Agentic V2 tool action registration")
        if set(actions) != set(FROZEN_TOOL_ACTIONS):
            raise ToolRegistryError("Agentic V2 registry must contain exactly the frozen action space")

    def resolve(self, action: ActionName | str) -> AgentTool:
        for tool in self.tools:
            if tool.definition.action == action:
                return tool
        raise ToolRegistryError(f"tool_not_registered:{action}")

    def definitions(self) -> tuple[AgentToolDefinition, ...]:
        return tuple(tool.definition for tool in self.tools)

    @property
    def dynamic_registration_allowed(self) -> bool:
        return False
