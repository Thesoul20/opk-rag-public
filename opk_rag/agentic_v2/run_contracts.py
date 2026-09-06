from __future__ import annotations

from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.termination import AgentTermination
from opk_rag.agentic_v2.tool_contracts import ToolStatus

AGENTIC_V2_STEP_RECORD_VERSION = "opk-rag.agentic-v2.step-record.v1"
AGENTIC_V2_RUN_RESULT_VERSION = "opk-rag.agentic-v2.run-result.v1"

AgentRunStatus = Literal[
    "finished",
    "abstained",
    "budget_exhausted",
    "guard_rejected",
    "invalid_policy_output",
    "runtime_failure",
]


class AgentStepRecord(StrictContract):
    contract_version: Literal[AGENTIC_V2_STEP_RECORD_VERSION] = AGENTIC_V2_STEP_RECORD_VERSION
    step_index: int = Field(ge=0, le=3)
    observation_digest: str = Field(min_length=64, max_length=64)
    allowed_actions: tuple[ActionName, ...]
    policy_decision_digest: str | None = Field(default=None, min_length=64, max_length=64)
    policy_decision_ids: tuple[str, ...] = ()
    policy_validation_status: Literal["passed", "failed"]
    policy_failure_code: str | None = Field(default=None, max_length=96)
    proposed_action: ActionName | None = None
    guard_outcome: Literal["allow", "modify", "reject", "terminate"] | None = None
    guard_reason_code: str | None = Field(default=None, max_length=96)
    guard_decision_id: str | None = Field(default=None, max_length=128)
    validated_action_digest: str | None = Field(default=None, min_length=64, max_length=64)
    execution_id: str | None = Field(default=None, max_length=128)
    tool_status: ToolStatus | None = None
    transition_status: Literal["applied", "rejected", "terminated"] | None = None
    next_state_digest: str | None = Field(default=None, min_length=64, max_length=64)
    next_observation_digest: str | None = Field(default=None, min_length=64, max_length=64)


class AgentRunResult(StrictContract):
    contract_version: Literal[AGENTIC_V2_RUN_RESULT_VERSION] = AGENTIC_V2_RUN_RESULT_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    initial_state_digest: str = Field(min_length=64, max_length=64)
    final_state: AgentState
    final_state_digest: str = Field(min_length=64, max_length=64)
    final_observation: AgentObservation
    final_observation_digest: str = Field(min_length=64, max_length=64)
    termination: AgentTermination
    step_count: int = Field(ge=0, le=3)
    policy_decision_count: int = Field(ge=0, le=3)
    tool_execution_count: int = Field(ge=0, le=3)
    transition_count: int = Field(ge=0, le=3)
    action_sequence: tuple[ActionName, ...] = ()
    run_status: AgentRunStatus
    step_records: tuple[AgentStepRecord, ...] = ()
    agent_run_digest: str = Field(min_length=64, max_length=64)
