from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, RootModel, TypeAdapter

from opk_rag.agentic_v2.action import (
    AbstainArgs,
    FinishArgs,
    GraphSearchArgs,
    HybridSearchArgs,
    InspectEvidenceArgs,
    RewriteQueryArgs,
    StructureSearchArgs,
)
from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_DECISION_CONTRACT_VERSION = "opk-rag.agentic-v2.decision.v1"


class _DecisionBase(StrictContract):
    contract_version: Literal[AGENTIC_V2_DECISION_CONTRACT_VERSION] = AGENTIC_V2_DECISION_CONTRACT_VERSION
    reason_code: str = Field(min_length=1, max_length=96)
    confidence: float = Field(ge=0.0, le=1.0)
    short_reason: str | None = Field(default=None, max_length=200, description="Public audit summary only; never hidden chain-of-thought.")


class HybridSearchDecision(_DecisionBase):
    proposed_action: Literal["hybrid_search"]
    arguments: HybridSearchArgs


class StructureSearchDecision(_DecisionBase):
    proposed_action: Literal["structure_search"]
    arguments: StructureSearchArgs


class GraphSearchDecision(_DecisionBase):
    proposed_action: Literal["graph_search"]
    arguments: GraphSearchArgs


class RewriteQueryDecision(_DecisionBase):
    proposed_action: Literal["rewrite_query"]
    arguments: RewriteQueryArgs


class InspectEvidenceDecision(_DecisionBase):
    proposed_action: Literal["inspect_evidence"]
    arguments: InspectEvidenceArgs


class FinishDecision(_DecisionBase):
    proposed_action: Literal["finish"]
    arguments: FinishArgs


class AbstainDecision(_DecisionBase):
    proposed_action: Literal["abstain"]
    arguments: AbstainArgs


DecisionVariant = Annotated[
    Union[
        HybridSearchDecision,
        StructureSearchDecision,
        GraphSearchDecision,
        RewriteQueryDecision,
        InspectEvidenceDecision,
        FinishDecision,
        AbstainDecision,
    ],
    Field(discriminator="proposed_action"),
]
DECISION_ADAPTER = TypeAdapter(DecisionVariant)


class AgentDecision(RootModel[DecisionVariant]):
    root: DecisionVariant

    @property
    def proposed_action(self) -> str:
        return self.root.proposed_action


def validate_agent_decision(value: object) -> AgentDecision:
    return AgentDecision(root=DECISION_ADAPTER.validate_python(value))
