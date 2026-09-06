from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_TERMINATION_CONTRACT_VERSION = "opk-rag.agentic-v2.termination.v1"
TerminationKind = Literal[
    "finished_answer",
    "abstained",
    "budget_exhausted",
    "invalid_policy_output",
    "guard_rejected",
    "insufficient_evidence",
    "runtime_failure",
]


class AgentTermination(StrictContract):
    contract_version: Literal[AGENTIC_V2_TERMINATION_CONTRACT_VERSION] = AGENTIC_V2_TERMINATION_CONTRACT_VERSION
    kind: TerminationKind
    reason_code: str = Field(min_length=1, max_length=96)
    is_runtime_failure: bool

    @model_validator(mode="after")
    def validate_failure_semantics(self) -> "AgentTermination":
        expected = self.kind == "runtime_failure"
        if self.is_runtime_failure is not expected:
            raise ValueError("is_runtime_failure must be true only for runtime_failure termination")
        return self
