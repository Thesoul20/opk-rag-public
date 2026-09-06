from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from opk_rag.agent.policy_provider import (
    AgentPolicyProvider as HistoricalAgentPolicyProvider,
    OpenAICompatiblePolicyProvider as HistoricalOpenAICompatiblePolicyProvider,
)
from opk_rag.agentic_v2.policy_input import AgentPolicyInput


@dataclass(frozen=True)
class AgentPolicyProviderResult:
    raw_output: str
    provider_name: str
    model_name: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "request_id_present": self.request_id is not None,
            "error_code": self.error_code,
            "metadata": self.metadata,
        }


class AgentPolicyProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_decision(
        self,
        *,
        prompt: str,
        policy_input: AgentPolicyInput,
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> AgentPolicyProviderResult:
        ...


class HistoricalPolicyProviderAdapter:
    """Reuse the proven V1 provider transport without reusing the V1 Agent runtime."""

    def __init__(self, provider: HistoricalAgentPolicyProvider) -> None:
        self.provider = provider
        self.provider_name = getattr(provider, "provider_name", "historical_policy_provider_adapter")
        self.model_name = getattr(provider, "model_name", "unknown-policy-model")

    def generate_decision(
        self,
        *,
        prompt: str,
        policy_input: AgentPolicyInput,
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> AgentPolicyProviderResult:
        result = self.provider.decide(
            prompt=prompt,
            state_view=policy_input.model_dump(mode="json"),
            output_schema=output_schema,
            repair_context=repair_context,
        )
        return AgentPolicyProviderResult(
            raw_output=result.raw_output,
            provider_name=result.provider_name,
            model_name=result.model_name,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
            error_code=result.error_code,
            error_message=result.error_message,
            metadata={
                "adapter": "historical_v1_provider_transport",
                "json_output_enabled": result.json_output_enabled,
                "thinking_mode": result.thinking_mode,
            },
        )


class OpenAICompatibleV2PolicyProvider(HistoricalPolicyProviderAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(HistoricalOpenAICompatiblePolicyProvider(**kwargs))


class FakePolicyProvider:
    provider_name = "fake_agentic_v2_policy_provider"
    model_name = "offline-scripted-agentic-v2-policy"

    def __init__(
        self,
        outputs: list[str | dict[str, Any]] | None = None,
        *,
        errors: list[str | None] | None = None,
    ) -> None:
        self.outputs = list(outputs or [])
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []

    def generate_decision(
        self,
        *,
        prompt: str,
        policy_input: AgentPolicyInput,
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> AgentPolicyProviderResult:
        started = time.monotonic()
        self.calls.append({
            "prompt_digest_input_length": len(prompt),
            "policy_input": policy_input.model_dump(mode="json"),
            "repair_context": repair_context,
        })
        error = self.errors.pop(0) if self.errors else None
        if error:
            return AgentPolicyProviderResult(
                raw_output="",
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                error_code=error,
                error_message="scripted provider error",
            )
        if self.outputs:
            output = self.outputs.pop(0)
            raw = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, sort_keys=True)
        else:
            raw = json.dumps(_default_decision(policy_input), ensure_ascii=False, sort_keys=True)
        return AgentPolicyProviderResult(
            raw_output=raw,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(raw) // 4),
            finish_reason="stop",
            request_id=f"fake-v2-{len(self.calls)}",
            metadata={"offline_contract_fixture": True},
        )


def _default_decision(policy_input: AgentPolicyInput) -> dict[str, Any]:
    action = "hybrid_search"
    arguments: dict[str, Any] = {"query": policy_input.current_query, "top_k": 10}
    reason = "initial_retrieval_needed"
    if policy_input.answerability_status in {"answerable", "partially_answerable"} and policy_input.evidence_count > 0:
        action = "finish"
        arguments = {"reason_code": "sufficient_evidence"}
        reason = "sufficient_evidence"
    elif policy_input.remaining_budget.retrieval_calls == 0:
        action = "abstain"
        arguments = {"reason_code": "budget_limited"}
        reason = "budget_limited"
    return {
        "contract_version": "opk-rag.agentic-v2.decision.v1",
        "proposed_action": action,
        "arguments": arguments,
        "reason_code": reason,
        "confidence": 0.8,
        "short_reason": "Offline scripted decision for contract validation.",
    }
