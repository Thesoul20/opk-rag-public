from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, RootModel, TypeAdapter

from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_ACTION_CONTRACT_VERSION = "opk-rag.agentic-v2.action.v1"
MAX_QUERY_LENGTH = 512

ActionName = Literal[
    "hybrid_search",
    "structure_search",
    "graph_search",
    "rewrite_query",
    "inspect_evidence",
    "finish",
    "abstain",
]


class HybridSearchArgs(StrictContract):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=10, ge=1, le=20)


class StructureSearchArgs(StrictContract):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    max_context_items: int = Field(default=2, ge=1, le=5)


class GraphSearchArgs(StrictContract):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    hop_limit: Literal[1] = 1


class RewriteQueryArgs(StrictContract):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)


class InspectEvidenceArgs(StrictContract):
    focus: str | None = Field(default=None, max_length=160)


class FinishArgs(StrictContract):
    reason_code: str = Field(min_length=1, max_length=96)


class AbstainArgs(StrictContract):
    reason_code: str = Field(min_length=1, max_length=96)


class _ActionBase(StrictContract):
    contract_version: Literal[AGENTIC_V2_ACTION_CONTRACT_VERSION] = AGENTIC_V2_ACTION_CONTRACT_VERSION
    reason_code: str = Field(min_length=1, max_length=96)


class HybridSearchAction(_ActionBase):
    action: Literal["hybrid_search"]
    arguments: HybridSearchArgs


class StructureSearchAction(_ActionBase):
    action: Literal["structure_search"]
    arguments: StructureSearchArgs


class GraphSearchAction(_ActionBase):
    action: Literal["graph_search"]
    arguments: GraphSearchArgs


class RewriteQueryAction(_ActionBase):
    action: Literal["rewrite_query"]
    arguments: RewriteQueryArgs


class InspectEvidenceAction(_ActionBase):
    action: Literal["inspect_evidence"]
    arguments: InspectEvidenceArgs


class FinishAction(_ActionBase):
    action: Literal["finish"]
    arguments: FinishArgs


class AbstainAction(_ActionBase):
    action: Literal["abstain"]
    arguments: AbstainArgs


ActionVariant = Annotated[
    Union[
        HybridSearchAction,
        StructureSearchAction,
        GraphSearchAction,
        RewriteQueryAction,
        InspectEvidenceAction,
        FinishAction,
        AbstainAction,
    ],
    Field(discriminator="action"),
]
ACTION_ADAPTER = TypeAdapter(ActionVariant)


class AgentAction(RootModel[ActionVariant]):
    root: ActionVariant

    @property
    def action(self) -> ActionName:
        return self.root.action

    @property
    def arguments(self):
        return self.root.arguments

    @property
    def reason_code(self) -> str:
        return self.root.reason_code


def validate_agent_action(value: object) -> AgentAction:
    return AgentAction(root=ACTION_ADAPTER.validate_python(value))
