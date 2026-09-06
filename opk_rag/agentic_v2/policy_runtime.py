from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.decision import AgentDecision, validate_agent_decision
from opk_rag.agentic_v2.policy_errors import AgentPolicyRuntimeError
from opk_rag.agentic_v2.policy_input import AgentPolicyInput, build_policy_input
from opk_rag.agentic_v2.policy_metrics import AgentPolicyMetrics
from opk_rag.agentic_v2.policy_prompt import AGENTIC_V2_POLICY_PROMPT_VERSION, SUPPORTED_POLICY_PROMPT_VERSIONS, build_policy_prompt, build_policy_repair_prompt, prompt_digest
from opk_rag.agentic_v2.policy_diagnostics import AgentPolicyValidationDiagnostic, diagnose_policy_output, parse_policy_json_value
from opk_rag.agentic_v2.policy_provider import AgentPolicyProvider, AgentPolicyProviderResult
from opk_rag.agentic_v2.policy_trace import AgentPolicyTrace, public_decision
from opk_rag.agentic_v2.observation import AgentObservation

AGENTIC_V2_LLM_POLICY_VERSION = "opk-rag.agentic-v2.llm-policy.v1"
MAX_STRUCTURAL_REPAIRS = 1
MAX_TRANSPORT_RETRIES = 1
ALLOWED_REASON_CODES = {
    "initial_retrieval_needed",
    "structural_context_needed",
    "cross_document_relation_needed",
    "query_ambiguity",
    "weak_evidence",
    "sufficient_evidence",
    "insufficient_evidence",
    "budget_limited",
}


@dataclass(frozen=True)
class AgentPolicyDecisionResult:
    decision: AgentDecision
    trace: tuple[AgentPolicyTrace, ...]
    diagnostics: tuple[AgentPolicyValidationDiagnostic, ...] = ()


class LLMAgentPolicyRuntime:
    def __init__(self, *, provider: AgentPolicyProvider, max_structural_repairs: int = MAX_STRUCTURAL_REPAIRS, max_transport_retries: int = 0, prompt_version: str = AGENTIC_V2_POLICY_PROMPT_VERSION) -> None:
        if max_structural_repairs != 1:
            raise ValueError("Agentic V2 policy structural repair budget is frozen at exactly one.")
        if max_transport_retries not in (0, 1):
            raise ValueError("Agentic V2 policy transport retry budget must be 0 or 1.")
        if prompt_version not in SUPPORTED_POLICY_PROMPT_VERSIONS:
            raise ValueError(f"unsupported policy prompt version: {prompt_version}")
        self.provider = provider
        self.prompt_version = prompt_version
        self.last_diagnostics: tuple[AgentPolicyValidationDiagnostic, ...] = ()
        self.max_structural_repairs = max_structural_repairs
        self.max_transport_retries = max_transport_retries
        self.metrics = AgentPolicyMetrics()

    def decide(
        self,
        *,
        observation: AgentObservation,
        allowed_actions: tuple[ActionName, ...] | None = None,
    ) -> AgentPolicyDecisionResult:
        policy_input = build_policy_input(observation, allowed_actions=allowed_actions)
        output_schema = AgentDecision.model_json_schema()
        prompt = build_policy_prompt(policy_input=policy_input, output_schema=output_schema, version=self.prompt_version)
        traces: list[AgentPolicyTrace] = []
        diagnostics: list[AgentPolicyValidationDiagnostic] = []
        last_failure_code = "policy_schema_validation_failed"
        previous_output_digest = "none"
        transport_retry_count = 0

        for attempt in range(self.max_structural_repairs + 1):
            repair_attempt = attempt > 0
            if repair_attempt:
                prompt = build_policy_repair_prompt(
                    policy_input=policy_input,
                    output_schema=output_schema,
                    failure_code=last_failure_code,
                    previous_output_digest=previous_output_digest,
                    version=self.prompt_version,
                    failure_detail=diagnostics[-1].failure_detail if diagnostics else None,
                    field_paths=diagnostics[-1].field_paths if diagnostics else (),
                    validation_error_types=diagnostics[-1].validation_error_types if diagnostics else (),
                )
            while True:
                result = self.provider.generate_decision(
                    prompt=prompt,
                    policy_input=policy_input,
                    output_schema=output_schema,
                    repair_context=None if not repair_attempt else {"failure_code": last_failure_code, "previous_output_digest": previous_output_digest},
                )
                effective_error = result.error_code
                failure_detail = "provider_failure"
                if effective_error is None and not result.raw_output.strip():
                    effective_error = "agent_policy_empty_structured_content"
                    failure_detail = "empty_structured_content"
                if effective_error is None:
                    break
                self._record_metric(
                    result, repair_attempt=repair_attempt, status="failure", failure_code=effective_error,
                    failure_detail=failure_detail, transport_retry=transport_retry_count > 0,
                )
                traces.append(self._trace(observation, policy_input, prompt, result, None, "failed", attempt, effective_error))
                diagnostic = diagnose_policy_output(
                    result.raw_output, attempt_index=attempt, repair_attempt=repair_attempt,
                    failure_code=effective_error, failure_detail=failure_detail,
                )
                diagnostics.append(diagnostic); self.last_diagnostics = tuple(diagnostics)
                if _transport_retryable(effective_error) and transport_retry_count < self.max_transport_retries:
                    transport_retry_count += 1
                    continue
                raise AgentPolicyRuntimeError(effective_error, result.error_message or "Policy provider failed.")

            try:
                parsed = parse_policy_json_value(result.raw_output)
                if not isinstance(parsed, dict):
                    raise AgentPolicyRuntimeError("policy_invalid_json", "Policy output must be one JSON object.")
                decision = validate_agent_decision(parsed)
                self._validate_policy_contract(decision, policy_input)
            except json.JSONDecodeError as exc:
                failure = AgentPolicyRuntimeError("policy_invalid_json", "Policy output is not valid JSON.")
                detail = "invalid_json"
                validation_error = None
            except ValidationError as exc:
                failure = AgentPolicyRuntimeError("policy_schema_validation_failed", "Policy output failed AgentDecision validation.")
                detail = _classify_validation_error(exc)
                validation_error = exc
            except AgentPolicyRuntimeError as exc:
                failure = exc
                detail = exc.failure.code
                validation_error = None
            else:
                diagnostic = diagnose_policy_output(result.raw_output, attempt_index=attempt, repair_attempt=repair_attempt, contract_valid=True)
                diagnostics.append(diagnostic); self.last_diagnostics = tuple(diagnostics)
                self._record_metric(result, repair_attempt=repair_attempt, status="success", failure_code=None)
                traces.append(self._trace(observation, policy_input, prompt, result, decision, "passed", attempt, None))
                return AgentPolicyDecisionResult(decision=decision, trace=tuple(traces), diagnostics=tuple(diagnostics))

            diagnostic = diagnose_policy_output(result.raw_output, attempt_index=attempt, repair_attempt=repair_attempt, validation_error=validation_error, failure_code=failure.failure.code, failure_detail=detail)
            diagnostics.append(diagnostic); self.last_diagnostics = tuple(diagnostics)
            last_failure_code = failure.failure.code
            previous_output_digest = diagnostic.raw_output_digest
            self._record_metric(result, repair_attempt=repair_attempt, status="failure", failure_code=last_failure_code, failure_detail=detail)
            traces.append(self._trace(observation, policy_input, prompt, result, None, "failed", attempt, last_failure_code))
            if attempt >= self.max_structural_repairs:
                raise AgentPolicyRuntimeError("policy_structural_repair_exhausted", "Policy structural repair budget exhausted.") from failure

        raise AgentPolicyRuntimeError("policy_structural_repair_exhausted", "Policy structural repair budget exhausted.")

    @staticmethod
    def _validate_policy_contract(decision: AgentDecision, policy_input: AgentPolicyInput) -> None:
        if decision.proposed_action not in policy_input.allowed_actions:
            raise AgentPolicyRuntimeError("policy_action_not_allowed", "Policy proposed an action outside allowed_actions.")
        if decision.root.reason_code not in ALLOWED_REASON_CODES:
            raise AgentPolicyRuntimeError("policy_reason_code_invalid", "Policy reason_code is outside the V2 policy contract.")

    def _trace(
        self,
        observation: AgentObservation,
        policy_input: AgentPolicyInput,
        prompt: str,
        result: AgentPolicyProviderResult,
        decision: AgentDecision | None,
        validation_status: str,
        repair_attempt_count: int,
        failure_code: str | None,
    ) -> AgentPolicyTrace:
        return AgentPolicyTrace(
            run_id=observation.run_id,
            decision_id=str(uuid4()),
            policy_version=AGENTIC_V2_LLM_POLICY_VERSION,
            provider_name=result.provider_name,
            model_name=result.model_name,
            prompt_version=self.prompt_version,
            prompt_digest=prompt_digest(prompt),
            observation_digest=stable_digest(observation),
            allowed_actions=tuple(policy_input.allowed_actions),
            decision=public_decision(decision),
            validation_status=validation_status,
            repair_attempt_count=repair_attempt_count,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            failure_code=failure_code,
        )

    def _record_metric(self, result: AgentPolicyProviderResult, *, repair_attempt: bool, status: str, failure_code: str | None, failure_detail: str | None = None, transport_retry: bool = False) -> None:
        self.metrics.record(
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            repair_attempt=repair_attempt,
            transport_retry=transport_retry,
            provider_request_count=1,
            provider_response_count=_provider_response_count(failure_code),
            status=status,
            failure_code=failure_code,
            failure_detail=failure_detail,
        )


def _provider_response_count(failure_code: str | None) -> int:
    if failure_code in {"agent_policy_timeout", "agent_policy_provider_failure"}:
        return 0
    return 1


def _transport_retryable(failure_code: str) -> bool:
    return failure_code in {"agent_policy_timeout", "agent_policy_provider_failure", "agent_policy_empty_structured_content"}


def _classify_validation_error(exc: ValidationError) -> str:
    text = str(exc).lower()
    if "proposed_action" in text and ("union_tag_invalid" in text or "input tag" in text):
        return "unknown_action"
    if "extra_forbidden" in text or "extra inputs are not permitted" in text:
        return "forbidden_argument"
    return "schema_mismatch"
