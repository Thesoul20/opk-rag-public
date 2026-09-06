from __future__ import annotations

from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.base import StrictContract, stable_digest
from opk_rag.agentic_v2.run_contracts import AgentRunStatus, AgentStepRecord
from opk_rag.agentic_v2.termination import AgentTermination

AGENTIC_V2_RUN_TRACE_VERSION = "opk-rag.agentic-v2.run-trace.v1"


class AgentRunTrace(StrictContract):
    contract_version: Literal[AGENTIC_V2_RUN_TRACE_VERSION] = AGENTIC_V2_RUN_TRACE_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    initial_state_digest: str = Field(min_length=64, max_length=64)
    final_state_digest: str = Field(min_length=64, max_length=64)
    final_observation_digest: str = Field(min_length=64, max_length=64)
    run_status: AgentRunStatus
    termination: AgentTermination
    step_records: tuple[AgentStepRecord, ...]
    semantic_digest: str = Field(min_length=64, max_length=64)


def run_semantic_digest(*, initial_state_digest: str, final_state_digest: str, final_observation_digest: str, run_status: str, termination: AgentTermination, step_records: tuple[AgentStepRecord, ...]) -> str:
    return stable_digest({
        "initial_state_digest": initial_state_digest,
        "final_state_digest": final_state_digest,
        "final_observation_digest": final_observation_digest,
        "run_status": run_status,
        "termination": termination.model_dump(mode="json"),
        "step_records": [_semantic_step_record(record) for record in step_records],
    })


def _semantic_step_record(record: AgentStepRecord) -> dict[str, object]:
    payload = record.model_dump(mode="json")
    # Correlation IDs are intentionally non-semantic: Policy/Tool layers may use UUIDs.
    payload.pop("policy_decision_ids", None)
    payload.pop("guard_decision_id", None)
    payload.pop("execution_id", None)
    return payload
