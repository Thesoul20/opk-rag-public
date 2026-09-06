from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_GUARD_DECISION_CONTRACT_VERSION = "opk-rag.agentic-v2.guard-decision.v1"


class AgentGuardDecision(StrictContract):
    contract_version: Literal[AGENTIC_V2_GUARD_DECISION_CONTRACT_VERSION] = AGENTIC_V2_GUARD_DECISION_CONTRACT_VERSION
    decision: Literal["allow", "modify", "reject", "terminate"]
    reason_code: str = Field(min_length=1, max_length=96)
    validated_action: AgentAction | None = None

    @model_validator(mode="after")
    def validate_action_presence(self) -> "AgentGuardDecision":
        if self.decision in {"allow", "modify"} and self.validated_action is None:
            raise ValueError("allow/modify guard decisions require validated_action")
        if self.decision in {"reject", "terminate"} and self.validated_action is not None:
            raise ValueError("reject/terminate guard decisions must not carry validated_action")
        return self
