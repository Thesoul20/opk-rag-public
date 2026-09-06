from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyProviderResult:
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
    json_output_enabled: bool = True
    thinking_mode: str = "disabled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "request_id_present": self.request_id is not None,
            "error_code": self.error_code,
            "json_output_enabled": self.json_output_enabled,
            "thinking_mode": self.thinking_mode,
            "metadata": self.metadata,
        }


class AgentPolicyProvider(Protocol):
    provider_name: str
    model_name: str
    json_output_enabled: bool
    thinking_mode: str

    def decide(
        self,
        *,
        prompt: str,
        state_view: dict[str, Any],
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> PolicyProviderResult:
        ...


class FakePolicyProvider:
    provider_name = "fake_policy_provider"
    model_name = "offline-scripted-policy"
    json_output_enabled = True
    thinking_mode = "disabled"

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        prompt: str,
        state_view: dict[str, Any],
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> PolicyProviderResult:
        started = time.monotonic()
        self.calls.append({"prompt": prompt, "state_view": state_view, "repair_context": repair_context})
        if self.outputs:
            raw = self.outputs.pop(0)
        else:
            raw = json.dumps(_scripted_decision(state_view), ensure_ascii=False, sort_keys=True)
        return PolicyProviderResult(
            raw_output=raw,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(raw) // 4),
            finish_reason="stop",
            request_id=f"fake-{len(self.calls)}",
        )


class OpenAICompatiblePolicyProvider:
    provider_name = "openai_compatible_policy_provider"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        thinking_mode: str | None = None,
        json_output_enabled: bool | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPK_RAG_AGENT_POLICY_BASE_URL") or "").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPK_RAG_AGENT_POLICY_API_KEY")
        self.model_name = model_name or os.getenv("OPK_RAG_AGENT_POLICY_MODEL") or "unknown-policy-model"
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float(os.getenv("OPK_RAG_AGENT_POLICY_TIMEOUT_SECONDS", "60"))
        self.temperature = temperature if temperature is not None else float(os.getenv("OPK_RAG_AGENT_POLICY_TEMPERATURE", "0"))
        self.json_output_enabled = json_output_enabled if json_output_enabled is not None else _env_bool("OPK_RAG_AGENT_POLICY_JSON_OUTPUT", True)
        self.thinking_mode = thinking_mode or os.getenv("OPK_RAG_AGENT_POLICY_THINKING_MODE", "disabled")
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("OPK_RAG_AGENT_POLICY_MAX_TOKENS", "512"))
        self.seed_supported = False

    def decide(
        self,
        *,
        prompt: str,
        state_view: dict[str, Any],
        output_schema: dict[str, Any],
        repair_context: dict[str, Any] | None = None,
    ) -> PolicyProviderResult:
        started = time.monotonic()
        if not self.base_url or not self.api_key:
            return self._error(started, "agent_policy_provider_failure", "Policy provider base URL or API key is not configured.")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": prompt}],
        }
        if self.thinking_mode != "disabled":
            payload["thinking"] = {"type": self.thinking_mode}
        if self.json_output_enabled:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            return self._error(started, "agent_policy_timeout", "Policy provider request timed out.")
        except urllib.error.HTTPError as exc:
            return self._error(started, "agent_policy_provider_failure", f"Policy provider request failed: HTTP {exc.code}")
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return self._error(started, "agent_policy_provider_failure", f"Policy provider request failed: {type(exc).__name__}")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        return PolicyProviderResult(
            raw_output=message.get("content") or "",
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            request_id=body.get("id"),
            thinking_mode=self.thinking_mode,
            json_output_enabled=self.json_output_enabled,
            metadata={"token_source": "provider_reported" if usage else "unavailable"},
        )

    def _error(self, started: float, code: str, message: str) -> PolicyProviderResult:
        return PolicyProviderResult(
            raw_output="",
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            error_code=code,
            error_message=message,
            thinking_mode=self.thinking_mode,
            json_output_enabled=self.json_output_enabled,
            metadata={"token_source": "unavailable"},
        )

    def redacted_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "base_url_redacted": _redact_url(self.base_url),
            "json_output_enabled": self.json_output_enabled,
            "thinking_mode": self.thinking_mode,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "seed_supported": self.seed_supported,
        }

    def parameter_audit(self) -> dict[str, Any]:
        return {
            "schema_version": "opk-rag.task0066-provider-parameter-audit.v1",
            "provider_type": "openai_compatible",
            "base_url_redacted": _redact_url(self.base_url),
            "model": self.model_name,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "thinking_mode": self.thinking_mode,
            "json_output": self.json_output_enabled,
            "seed_supported": self.seed_supported,
        }


def _scripted_decision(state_view: dict[str, Any]) -> dict[str, Any]:
    action = (state_view.get("available_actions") or ["finish_failure"])[0]
    if action == "search":
        args = {"query": state_view["question"], "top_k": 10}
        reason = "initial_retrieval_required"
        transition = "evidence_available"
    elif action == "evaluate_answerability":
        args = {}
        reason = "evidence_available_requires_assessment"
        transition = "answerability_decided"
    elif action == "expand_evidence":
        args = {"chunk_id": "chunk-1", "max_items": 1}
        reason = "insufficient_evidence_can_expand"
        transition = "evidence_expanded"
    elif action == "generate_grounded_answer":
        args = {}
        reason = "answerable_generation_required"
        transition = "generation_available"
    elif action == "verify_grounding":
        args = {}
        reason = "generated_answer_requires_verification"
        transition = "grounding_verified"
    elif action == "finish_answer":
        args = {}
        reason = "verified_answer_can_finish"
        transition = "terminal_answer"
    elif action == "finish_abstain":
        args = {}
        reason = "unanswerable_must_abstain"
        transition = "terminal_abstain"
    else:
        args = {}
        reason = "invalid_state_must_fail"
        transition = "terminal_failure"
    return {
        "contract_version": "opk-rag.agent-policy-decision.v1",
        "action": action,
        "reason_code": reason,
        "arguments": args,
        "decision_summary": f"Selected governed action {action}.",
        "expected_state_transition": transition,
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "json_object"}


def _redact_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme or None,
        "host_suffix": _host_suffix(parsed.hostname or ""),
        "path_present": bool(parsed.path and parsed.path != "/"),
    }


def _host_suffix(host: str) -> str | None:
    if not host:
        return None
    parts = host.split(".")
    if len(parts) >= 2:
        return "*." + ".".join(parts[-2:])
    return "<redacted>"
