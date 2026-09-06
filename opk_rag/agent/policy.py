from __future__ import annotations

from typing import Protocol

from opk_rag.agent.contracts import AgentAction, AgentState


class AgentPolicy(Protocol):
    name: str
    version: str

    def next_action(self, state: AgentState) -> AgentAction:
        ...
