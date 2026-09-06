from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPolicyFailure:
    code: str
    message: str


class AgentPolicyRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.failure = AgentPolicyFailure(code=code, message=message)
