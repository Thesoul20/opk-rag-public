from __future__ import annotations

from typing import Protocol

from opk_rag.agentic_v2.decision import AgentDecision
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime


class AgentPolicy(Protocol):
    def decide(self, *, observation: AgentObservation) -> AgentDecision:
        ...


class LLMAgentPolicy:
    def __init__(self, runtime: LLMAgentPolicyRuntime) -> None:
        self.runtime = runtime
        self.last_trace = ()

    def decide(self, *, observation: AgentObservation) -> AgentDecision:
        result = self.runtime.decide(observation=observation)
        self.last_trace = result.trace
        return result.decision
