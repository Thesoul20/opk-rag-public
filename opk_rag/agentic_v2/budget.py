from __future__ import annotations

from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_BUDGET_CONTRACT_VERSION = "opk-rag.agentic-v2.budget.v1"


class AgentBudget(StrictContract):
    contract_version: Literal[AGENTIC_V2_BUDGET_CONTRACT_VERSION] = AGENTIC_V2_BUDGET_CONTRACT_VERSION
    max_steps: Literal[3] = 3
    max_retrieval_calls: Literal[2] = 2
    max_query_rewrites: Literal[1] = 1
    max_graph_calls: Literal[1] = 1
    max_graph_hops: Literal[1] = 1


class RemainingBudget(StrictContract):
    steps: int = Field(ge=0, le=3)
    retrieval_calls: int = Field(ge=0, le=2)
    query_rewrites: int = Field(ge=0, le=1)
    graph_calls: int = Field(ge=0, le=1)
    graph_hops: int = Field(ge=0, le=1)
