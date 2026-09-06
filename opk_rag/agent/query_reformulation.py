from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
import urllib.error
import urllib.request
from typing import Any, Protocol

from opk_rag.agent.contracts import stable_digest, stable_json
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.query_plan import QUERY_PLAN_CONTRACT_VERSION, QueryPlan, QueryPlanValidationConfig, parse_query_plan_json, validate_query_plan
from opk_rag.agent.query_privacy import GOLD_LEAKAGE_TERMS, SECRET_TERMS

QUERY_REFORMULATION_PROMPT_VERSION = "opk-rag.query-reformulation-prompt.v1"


@dataclass(frozen=True)
class QueryReformulationResult:
    raw_output: str
    provider_name: str = "fake_query_reformulation_provider"
    model_name: str | None = "offline"
    latency_ms: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    token_source: str = "provider_reported"
    error_code: str | None = None
    error_message: str | None = None

    def to_trace_dict(self) -> dict[str, Any]:
        token_source = self.token_source
        if token_source not in {"provider_reported", "estimated", "unavailable"}:
            token_source = "unavailable"
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name or "unknown",
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_source": token_source,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class QueryReformulationProvider(Protocol):
    def reformulate(self, *, prompt: str, state_view: dict[str, Any], repair_context: dict[str, Any] | None = None) -> QueryReformulationResult:
        ...


class FakeQueryReformulationProvider:
    def __init__(self, raw_outputs: list[str] | None = None, *, error_code: str | None = None) -> None:
        self.raw_outputs = raw_outputs or []
        self.error_code = error_code
        self.calls = 0

    def reformulate(self, *, prompt: str, state_view: dict[str, Any], repair_context: dict[str, Any] | None = None) -> QueryReformulationResult:
        self.calls += 1
        if self.error_code is not None:
            return QueryReformulationResult(raw_output="", error_code=self.error_code, error_message=self.error_code)
        if self.raw_outputs:
            return QueryReformulationResult(raw_output=self.raw_outputs[min(self.calls - 1, len(self.raw_outputs) - 1)])
        question = str(state_view.get("question") or "")
        payload = {
            "contract_version": QUERY_PLAN_CONTRACT_VERSION,
            "strategy": "focused_reformulation",
            "reason_code": "initial_evidence_insufficient",
            "queries": [{"query": f"{question} 相关配置位置", "query_role": "primary_rewrite"}],
            "source": "deterministic_reformulation",
            "expected_information_gain": "scope_localization",
            "stop_after_round": 2,
        }
        return QueryReformulationResult(raw_output=json.dumps(payload, ensure_ascii=False))


class OpenAICompatibleQueryReformulationProvider:
    provider_name = "openai_compatible_query_reformulation_provider"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        thinking_mode: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPK_RAG_QUERY_REFORMULATION_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPK_RAG_QUERY_REFORMULATION_API_KEY")
        self.model = model or os.getenv("OPK_RAG_QUERY_REFORMULATION_MODEL")
        self.timeout_seconds = timeout_seconds or float(os.getenv("OPK_RAG_QUERY_REFORMULATION_TIMEOUT_SECONDS", "60"))
        self.temperature = temperature if temperature is not None else float(os.getenv("OPK_RAG_QUERY_REFORMULATION_TEMPERATURE", "0"))
        self.thinking_mode = thinking_mode if thinking_mode is not None else os.getenv("OPK_RAG_QUERY_REFORMULATION_THINKING_MODE")
        self.json_output = True
        self.seed_supported = False

    def reformulate(self, *, prompt: str, state_view: dict[str, Any], repair_context: dict[str, Any] | None = None) -> QueryReformulationResult:
        if not self.base_url or not self.api_key or not self.model:
            return QueryReformulationResult(raw_output="", provider_name=self.provider_name, model_name=self.model, error_code="query_reformulation_provider_failure", error_message="Missing query reformulation provider configuration.")
        started = time.monotonic()
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
            if self.thinking_mode and self.thinking_mode != "disabled":
                payload["thinking"] = {"type": self.thinking_mode}
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = body.get("usage") or {}
            raw_output = message.get("content") or ""
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            return QueryReformulationResult(
                raw_output=raw_output,
                provider_name=self.provider_name,
                model_name=self.model,
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                token_source="provider_reported" if usage else "unavailable",
            )
        except TimeoutError:
            return QueryReformulationResult(raw_output="", provider_name=self.provider_name, model_name=self.model, latency_ms=max(0, int((time.monotonic() - started) * 1000)), error_code="query_reformulation_timeout", error_message="TimeoutError")
        except urllib.error.HTTPError as exc:
            return QueryReformulationResult(raw_output="", provider_name=self.provider_name, model_name=self.model, latency_ms=max(0, int((time.monotonic() - started) * 1000)), error_code="query_reformulation_provider_failure", error_message=f"HTTP {exc.code}")
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return QueryReformulationResult(raw_output="", provider_name=self.provider_name, model_name=self.model, latency_ms=max(0, int((time.monotonic() - started) * 1000)), error_code="query_reformulation_provider_failure", error_message=type(exc).__name__)


class GovernedQueryReformulator:
    def __init__(
        self,
        *,
        provider: QueryReformulationProvider,
        validation_config: QueryPlanValidationConfig | None = None,
        max_structural_repairs: int = 1,
    ) -> None:
        self.provider = provider
        self.validation_config = validation_config or QueryPlanValidationConfig()
        self.max_structural_repairs = max_structural_repairs
        self.last_trace: dict[str, Any] | None = None
        self.call_traces: list[dict[str, Any]] = []

    def reformulate(self, *, question: str, initial_query: str, summary: dict[str, Any]) -> QueryPlan:
        state_view = build_query_reformulation_state_view(question=question, initial_query=initial_query, summary=summary, validation_config=self.validation_config)
        prompt = build_query_reformulation_prompt(state_view=state_view)
        repair_context = None
        last_error: AgentRuntimeError | None = None
        for attempt in range(self.max_structural_repairs + 1):
            result = self.provider.reformulate(prompt=prompt, state_view=state_view, repair_context=repair_context)
            if result.error_code is not None:
                error = AgentRuntimeError(result.error_code, result.error_message or "Query reformulation provider failed.", origin="query_reformulation")
                self._record(result, prompt, state_view, attempt=attempt, failure=error.failure.to_dict())
                raise error
            try:
                payload = parse_query_plan_json(result.raw_output)
                plan = validate_query_plan(payload, original_query=initial_query, config=self.validation_config)
            except AgentRuntimeError as exc:
                last_error = exc
                self._record(result, prompt, state_view, attempt=attempt, failure=exc.failure.to_dict())
                if _structural_repair_allowed(exc) and attempt < self.max_structural_repairs:
                    prompt = build_query_reformulation_repair_prompt(state_view=state_view, previous_output_digest=stable_digest(result.raw_output), failure=exc.failure.to_dict())
                    repair_context = {"failure": exc.failure.to_dict(), "instruction": "Return exactly one repaired Query Plan JSON object."}
                    continue
                if attempt >= self.max_structural_repairs and _structural_repair_allowed(exc):
                    raise AgentRuntimeError("query_plan_repair_exhausted", "Query plan repair budget exhausted.", origin="query_reformulation", detail={"last_failure": exc.failure.to_dict()}) from exc
                raise
            self._record(result, prompt, state_view, attempt=attempt, plan=plan.to_dict(redact_queries=True))
            return plan
        assert last_error is not None
        raise last_error

    def _record(self, result: QueryReformulationResult, prompt: str, state_view: dict[str, Any], *, attempt: int, plan: dict[str, Any] | None = None, failure: dict[str, Any] | None = None) -> None:
        trace = {
            "schema_version": "opk-rag.query-reformulation-trace.v1",
            "prompt_version": QUERY_REFORMULATION_PROMPT_VERSION,
            "prompt_digest": query_reformulation_prompt_digest(prompt),
            "state_view_digest": stable_digest(state_view),
            "attempt": attempt,
            "repair_attempt": attempt > 0,
            "provider": result.to_trace_dict(),
            "raw_output_digest": stable_digest(result.raw_output),
            "raw_output_length": len(result.raw_output),
            "plan": plan,
            "failure": failure,
        }
        self.call_traces.append(trace)
        self.last_trace = trace


def build_query_reformulation_state_view(*, question: str, initial_query: str, summary: dict[str, Any], validation_config: QueryPlanValidationConfig | None = None) -> dict[str, Any]:
    validation_config = validation_config or QueryPlanValidationConfig()
    state_view = {
        "contract_version": "opk-rag.query-reformulation-state-view.v1",
        "question": _sanitize(question),
        "initial_query_digest": stable_digest(initial_query),
        "initial_query_length": len(initial_query.strip()),
        "answerability_status": summary.get("answerability_status"),
        "retrieval_summary": {
            "documents_found": int(summary.get("documents_found") or 0),
            "evidence_identities_found": int(summary.get("evidence_identities_found") or 0),
            "scope_localization_status": str(summary.get("scope_localization_status") or "unknown"),
            "duplicate_ratio": float(summary.get("duplicate_ratio") or 0.0),
        },
        "remaining_budget": {
            "search_rounds": max(0, validation_config.max_search_rounds - 1),
            "queries": validation_config.max_queries,
        },
        "limits": {
            "max_query_length": validation_config.max_query_length,
            "max_queries": validation_config.max_queries,
        },
    }
    _assert_no_leakage(stable_json(state_view))
    return state_view


def build_query_reformulation_prompt(*, state_view: dict[str, Any]) -> str:
    prompt = "\n".join(
        [
            f"Query reformulation prompt version: {QUERY_REFORMULATION_PROMPT_VERSION}",
            "You generate governed search queries only. Do not answer the user question.",
            "Return exactly one JSON object and no Markdown, comments, citations, final answers, or hidden reasoning.",
            "The JSON object must conform to opk-rag.agent-query-plan.v1.",
            "Do not wrap the object in query_plan, result, data, choices, or any other outer field.",
            "Use exactly these top-level fields: contract_version, strategy, reason_code, queries, source, expected_information_gain, stop_after_round.",
            "Allowed strategies: focused_reformulation, entity_disambiguation, scope_narrowing, terminology_expansion, document_locator_rewrite, safe_no_reformulation.",
            "Allowed reason codes: initial_evidence_insufficient, initial_query_too_broad, missing_scope_localization, ambiguous_entity, terminology_mismatch, document_found_chunk_missing, reformulation_not_expected_to_help, budget_exhausted.",
            "Generate at most two non-duplicate queries. Queries must not repeat the original query digest, contain SQL, shell/file commands, URLs, prompt instructions, budget changes, citations, or direct answers.",
            "Use only the user question and the governed retrieval summary. Do not use evaluation labels, target behavior fields, privileged evidence, scoring fields, or human review metadata.",
            "State view:",
            stable_json(state_view),
        ]
    )
    _assert_no_leakage(prompt)
    return prompt


def build_query_reformulation_repair_prompt(*, state_view: dict[str, Any], previous_output_digest: str, failure: dict[str, Any]) -> str:
    prompt = "\n".join(
        [
            f"Query reformulation prompt version: {QUERY_REFORMULATION_PROMPT_VERSION}",
            "Repair one Query Plan JSON object. Return JSON only.",
            "Do not wrap the repaired object in query_plan, result, data, choices, or any other outer field.",
            "Do not add answers, citations, tool calls, SQL, file commands, URLs, budget changes, or hidden reasoning.",
            "Previous output digest:",
            previous_output_digest,
            "Failure:",
            stable_json({"code": failure.get("code"), "message": failure.get("message")}),
            "State view:",
            stable_json(state_view),
        ]
    )
    _assert_no_leakage(prompt)
    return prompt


def query_reformulation_prompt_digest(prompt: str) -> str:
    return stable_digest({"prompt_version": QUERY_REFORMULATION_PROMPT_VERSION, "prompt": prompt})


def query_reformulation_prompt_manifest() -> dict[str, Any]:
    empty_view = build_query_reformulation_state_view(question="placeholder question", initial_query="placeholder question", summary={"answerability_status": "insufficient_evidence"})
    prompt = build_query_reformulation_prompt(state_view=empty_view)
    return {
        "schema_version": "opk-rag.query-reformulation-prompt-manifest.v1",
        "prompt_version": QUERY_REFORMULATION_PROMPT_VERSION,
        "prompt_digest": query_reformulation_prompt_digest(prompt),
        "query_plan_contract_version": QUERY_PLAN_CONTRACT_VERSION,
        "gold_leakage_terms_excluded": True,
        "sends_full_evidence_text": False,
    }


def provider_config_redacted(provider: QueryReformulationProvider) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.query-reformulation-provider-config.v1",
        "provider_class": type(provider).__name__,
        "provider_name": getattr(provider, "provider_name", type(provider).__name__),
        "model_name": getattr(provider, "model", None) or getattr(provider, "model_name", None) or "offline",
        "credential_present": bool(getattr(provider, "api_key", None)),
        "base_url_present": bool(getattr(provider, "base_url", None)),
        "base_url_redacted": _redact_base_url(getattr(provider, "base_url", None)),
        "temperature": getattr(provider, "temperature", None),
        "timeout_seconds": getattr(provider, "timeout_seconds", None),
        "thinking_mode": getattr(provider, "thinking_mode", None),
        "json_output": bool(getattr(provider, "json_output", False)),
        "seed_supported": bool(getattr(provider, "seed_supported", False)),
    }


def _redact_base_url(value: str | None) -> str | None:
    if not value:
        return None
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return "<configured>"
    host = parts.hostname or ""
    if len(host) > 10:
        host = f"{host[:4]}...{host[-6:]}"
    netloc = host
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def _sanitize(value: str) -> str:
    sanitized = value
    for term in (*GOLD_LEAKAGE_TERMS, *SECRET_TERMS):
        sanitized = sanitized.replace(term, "[redacted_governance_term]")
        sanitized = sanitized.replace(term.upper(), "[redacted_governance_term]")
    return sanitized


def _assert_no_leakage(value: str) -> None:
    lowered = value.lower()
    leaked = [term for term in (*GOLD_LEAKAGE_TERMS, *SECRET_TERMS) if term in lowered]
    if leaked:
        raise ValueError(f"query reformulation prompt contains prohibited terms: {sorted(set(leaked))}")


def _structural_repair_allowed(exc: AgentRuntimeError) -> bool:
    return exc.failure.code in {"query_plan_malformed_json", "query_plan_contract_failure", "query_plan_empty_query", "query_plan_duplicate_query", "query_plan_original_query_duplicate"}
