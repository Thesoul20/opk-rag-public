from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.budget import AgentBudget
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.state_transition_contracts import AgentTransitionObservationSignals
from opk_rag.agentic_v2.tool_contracts import AgentToolResult
from opk_rag.agentic_v2.tools import AgentToolContext


ObservationSignalResolver = Callable[[AgentState, AgentAction, AgentToolResult], AgentTransitionObservationSignals | None]


@dataclass(frozen=True)
class AgentRuntimeContext:
    """System-owned runtime dependencies; none of this object is model-visible."""

    tool_context: AgentToolContext
    budget: AgentBudget = field(default_factory=AgentBudget)
    observation_signal_resolver: ObservationSignalResolver | None = None
