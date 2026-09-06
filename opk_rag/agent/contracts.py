from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from opk_rag.agent.errors import AgentFailure

AGENT_STATE_CONTRACT_VERSION = "opk-rag.agent-state.v1"
AGENT_ACTION_CONTRACT_VERSION = "opk-rag.agent-action.v1"
AGENT_TRACE_CONTRACT_VERSION = "opk-rag.agent-trace.v1"
AGENT_RUNTIME_CONTRACT_VERSION = "opk-rag.agent-runtime.v1"

AgentStatus = Literal["running", "answered", "abstained", "failed"]
AgentActionName = Literal[
    "search",
    "expand_evidence",
    "evaluate_answerability",
    "generate_grounded_answer",
    "verify_grounding",
    "finish_answer",
    "finish_abstain",
    "finish_failure",
]
FinalAction = Literal["answer", "abstain", "failure"]


@dataclass(frozen=True)
class AgentRuntimeConfig:
    contract_version: str = AGENT_RUNTIME_CONTRACT_VERSION
    policy_name: str = "deterministic_baseline_v1"
    policy_version: str = "1.0.0"
    max_steps: int = 8
    max_tool_calls: int = 6
    max_expansions: int = 1
    search_top_k: int = 10
    expansion_max_items: int = 1
    answerability_required: bool = True
    grounding_verification_required: bool = True
    citation_rules_overridable: bool = False
    model_planning_enabled: bool = False
    max_search_rounds: int = 2
    max_reformulations: int = 1
    max_queries_per_reformulation: int = 2
    max_reformulated_query_length: int = 256
    max_query_plan_repairs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class AgentAction:
    action: AgentActionName
    reason_code: str
    arguments: dict[str, Any] = field(default_factory=dict)
    contract_version: str = AGENT_ACTION_CONTRACT_VERSION
    expected_state_transition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class AgentState:
    run_id: str
    question: str
    sample_id: str | None = None
    step_index: int = 0
    status: AgentStatus = "running"
    evidence_state: dict[str, Any] = field(default_factory=dict)
    answerability_state: dict[str, Any] = field(default_factory=dict)
    generation_state: dict[str, Any] = field(default_factory=dict)
    verification_state: dict[str, Any] = field(default_factory=dict)
    tool_call_count: int = 0
    expansion_count: int = 0
    search_round_count: int = 0
    final_action: FinalAction | None = None
    failure: AgentFailure | None = None
    contract_version: str = AGENT_STATE_CONTRACT_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.status in {"answered", "abstained", "failed"}

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        if self.failure is not None:
            payload["failure"] = self.failure.to_dict()
        return payload


def state_digest(state: AgentState) -> str:
    encoded = stable_json(state.to_dict()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, AgentFailure):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in sorted(value.__dataclass_fields__)}
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value
