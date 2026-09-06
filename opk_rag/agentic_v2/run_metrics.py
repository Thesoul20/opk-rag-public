from __future__ import annotations

from statistics import mean


class AgentRunMetrics:
    def __init__(self) -> None:
        self._runs: list[dict[str, object]] = []

    def record(self, *, run_status: str, steps: int, policy_decisions: int, tool_executions: int, transitions: int, action_sequence: tuple[str, ...], latency_ms: float) -> None:
        self._runs.append({
            "run_status": run_status,
            "steps": steps,
            "policy_decisions": policy_decisions,
            "tool_executions": tool_executions,
            "transitions": transitions,
            "action_sequence": action_sequence,
            "latency_ms": max(0.0, float(latency_ms)),
        })

    def summary(self) -> dict[str, object]:
        rows = list(self._runs)
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        steps = [int(row["steps"]) for row in rows]
        actions = [tuple(row["action_sequence"]) for row in rows]
        def count_status(name: str) -> int:
            return sum(row["run_status"] == name for row in rows)
        return {
            "agent_run_count": len(rows),
            "finished_run_count": count_status("finished"),
            "abstained_run_count": count_status("abstained"),
            "budget_exhausted_run_count": count_status("budget_exhausted"),
            "guard_rejected_run_count": count_status("guard_rejected"),
            "runtime_failure_run_count": count_status("runtime_failure"),
            "invalid_policy_output_run_count": count_status("invalid_policy_output"),
            "single_step_run_count": sum(step == 1 for step in steps),
            "multi_step_run_count": sum(step > 1 for step in steps),
            "policy_decision_count": sum(int(row["policy_decisions"]) for row in rows),
            "tool_execution_count": sum(int(row["tool_executions"]) for row in rows),
            "state_transition_count": sum(int(row["transitions"]) for row in rows),
            "average_steps_per_run": mean(steps) if steps else 0.0,
            "max_steps_observed": max(steps, default=0),
            "rewrite_used_run_count": sum("rewrite_query" in seq for seq in actions),
            "graph_used_run_count": sum("graph_search" in seq for seq in actions),
            "p50_agent_runtime_ms": _percentile(latencies, 0.50),
            "p95_agent_runtime_ms": _percentile(latencies, 0.95),
        }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
    return values[index]
