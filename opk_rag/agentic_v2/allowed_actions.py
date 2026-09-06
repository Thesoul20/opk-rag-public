from __future__ import annotations

from opk_rag.agentic_v2.action import ActionName, AgentAction
from opk_rag.agentic_v2.budget import AgentBudget
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.state import AgentState


class AllowedActionResolver:
    """Deterministic, side-effect-free authority for currently eligible V2 actions."""

    def resolve(
        self,
        *,
        state: AgentState,
        observation: AgentObservation,
        budget: AgentBudget | None = None,
        action_history: tuple[AgentAction, ...] = (),
    ) -> tuple[ActionName, ...]:
        budget = budget or AgentBudget()
        if state.status != "running":
            return ()
        remaining = state.remaining_budget(budget)
        actions: list[ActionName] = []

        # Abstention is always a safe terminal proposal while the run is active.
        actions.append("abstain")
        if remaining.steps <= 0:
            return tuple(actions)

        if remaining.retrieval_calls > 0:
            actions.append("hybrid_search")
            actions.append("structure_search")
            if (
                remaining.graph_calls > 0
                and remaining.graph_hops > 0
                and state.graph_call_count < budget.max_graph_calls
                and state.graph_hop_count < budget.max_graph_hops
                and "graph_search" not in state.previous_actions
                and not any(item.action == "graph_search" for item in action_history)
            ):
                actions.append("graph_search")

        if remaining.query_rewrites > 0 and state.rewrite_count < budget.max_query_rewrites:
            actions.append("rewrite_query")

        if observation.evidence_count > 0:
            actions.append("inspect_evidence")
            if observation.answerability_status in {"answerable", "partially_answerable"}:
                actions.append("finish")

        # Preserve the frozen public action-space order for stable prompts/traces.
        frozen_order: tuple[ActionName, ...] = (
            "hybrid_search", "structure_search", "graph_search", "rewrite_query",
            "inspect_evidence", "finish", "abstain",
        )
        return tuple(action for action in frozen_order if action in actions)
