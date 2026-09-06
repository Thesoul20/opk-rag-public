from __future__ import annotations

import json
import hashlib
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.models import AnswerGenerationRequest, RawAnswerGeneration
from opk_rag.answer.prompt import SYSTEM_PROMPT, output_schema_for_version, render_user_prompt


@dataclass(frozen=True)
class ProviderErrorDetail:
    reason_code: str
    http_status: int | None = None
    sdk_error_type: str | None = None
    error_summary: str | None = None
    attempts: tuple[dict[str, object], ...] = ()


class AnswerProviderError(RuntimeError):
    def __init__(self, message: str, *, detail: ProviderErrorDetail | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class RemoteLLMNotAllowedError(AnswerProviderError):
    pass


class OpenAICompatibleLocalChatProvider:
    def __init__(self, config: AnswerGenerationConfig, *, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key
        self._endpoint_type = _endpoint_type(config.base_url)
        if self._endpoint_type == "remote" and not config.allow_remote:
            raise RemoteLLMNotAllowedError("Remote LLM endpoint is disabled by default; set OPK_RAG_LLM_ALLOW_REMOTE=true to allow it.")

    @property
    def provider_id(self) -> str:
        return self.config.provider_id

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def model_revision(self) -> str | None:
        return self.config.model_revision

    @property
    def endpoint_type(self) -> str:
        return self._endpoint_type

    def generate_answer(self, request: AnswerGenerationRequest) -> RawAnswerGeneration:
        payload = build_chat_completion_payload(self.config, request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()
        raw_response, attempts = self._post_with_retries(endpoint, body, headers)
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            envelope = json.loads(raw_response)
            choice = envelope["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AnswerProviderError("LLM endpoint returned an invalid chat completion envelope.") from exc
        if not isinstance(content, str):
            raise AnswerProviderError(
                "LLM endpoint returned a non-text chat completion content.",
                detail=ProviderErrorDetail("unsupported_response_shape", attempts=tuple(attempts)),
            )
        if not content.strip():
            raise AnswerProviderError(
                "LLM endpoint returned empty chat completion content.",
                detail=ProviderErrorDetail("empty_response_content", attempts=tuple(attempts)),
            )
        parsed = _parse_json_object(content)
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        return RawAnswerGeneration(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            endpoint_type=self._endpoint_type,
            raw_text=content,
            parsed=parsed,
            latency_ms=latency_ms,
            prompt_tokens=_usage_int(usage, "prompt_tokens"),
            completion_tokens=_usage_int(usage, "completion_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            request_attempts=tuple(attempts),
            raw_envelope_hash=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        )

    def _post_with_retries(self, endpoint: str, body: bytes, headers: dict[str, str]) -> tuple[str, list[dict[str, object]]]:
        last_error: Exception | None = None
        attempts: list[dict[str, object]] = []
        for attempt_index in range(self.config.max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                with urlopen(Request(endpoint, data=body, headers=headers, method="POST"), timeout=self.config.timeout_seconds) as response:
                    text = response.read().decode("utf-8")
                    attempts.append({"attempt": attempt_index + 1, "status": "success", "duration_ms": int((time.perf_counter() - attempt_started) * 1000)})
                    return text, attempts
            except HTTPError as exc:
                detail = _http_error_detail(exc, tuple(attempts))
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "http_error",
                        "reason_code": detail.reason_code,
                        "http_status": exc.code,
                        "error_summary": detail.error_summary,
                        "duration_ms": int((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                detail = ProviderErrorDetail(detail.reason_code, detail.http_status, detail.sdk_error_type, detail.error_summary, tuple(attempts))
                last_error = AnswerProviderError(f"LLM endpoint returned HTTP {exc.code}: {detail.reason_code}", detail=detail)
                if exc.code == 404 or 400 <= exc.code < 500:
                    break
            except URLError as exc:
                detail = _url_error_detail(exc, tuple(attempts))
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "url_error",
                        "reason_code": detail.reason_code,
                        "sdk_error_type": detail.sdk_error_type,
                        "error_summary": detail.error_summary,
                        "duration_ms": int((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                detail = ProviderErrorDetail(detail.reason_code, detail.http_status, detail.sdk_error_type, detail.error_summary, tuple(attempts))
                last_error = AnswerProviderError(f"LLM endpoint is unavailable: {detail.reason_code}", detail=detail)
            except TimeoutError as exc:
                detail = ProviderErrorDetail("read_timeout", sdk_error_type=type(exc).__name__, error_summary="LLM endpoint timed out.", attempts=tuple(attempts))
                attempts.append({"attempt": attempt_index + 1, "status": "timeout", "reason_code": detail.reason_code, "duration_ms": int((time.perf_counter() - attempt_started) * 1000)})
                detail = ProviderErrorDetail(detail.reason_code, detail.http_status, detail.sdk_error_type, detail.error_summary, tuple(attempts))
                last_error = AnswerProviderError("LLM endpoint timed out.", detail=detail)
        if last_error is None:
            last_error = AnswerProviderError("LLM endpoint failed.", detail=ProviderErrorDetail("unknown_provider_error", attempts=tuple(attempts)))
        raise last_error


def build_chat_completion_payload(config: AnswerGenerationConfig, request: AnswerGenerationRequest) -> dict:
    payload = {
        "model": config.model_id,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_output_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_user_prompt(
                    request.query,
                    request.evidence_bundle,
                    output_schema_version=request.output_schema_version,
                    answerability=request.answerability,
                    scope_execution_contract=request.scope_execution_contract,
                ),
            },
        ],
    }
    if _send_provider_parameter(config, "repetition_penalty"):
        payload["repetition_penalty"] = config.repetition_penalty
    if config.response_format_type == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "opk_rag_answer_response",
                "strict": True,
                "schema": output_schema_for_version(request.output_schema_version),
            },
        }
    elif config.response_format_type == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if config.thinking_mode is not None:
        payload["thinking"] = {"type": config.thinking_mode}
    if config.top_k is not None and _send_provider_parameter(config, "top_k"):
        payload["top_k"] = config.top_k
    if config.seed is not None and _send_provider_parameter(config, "seed"):
        payload["seed"] = config.seed
    if config.stop_sequences:
        payload["stop"] = list(config.stop_sequences)
    return payload


def _send_provider_parameter(config: AnswerGenerationConfig, parameter: str) -> bool:
    if config.provider_parameter_policy == "deepseek_v4":
        return parameter in {"model", "temperature", "top_p", "max_tokens", "response_format", "thinking", "stop"}
    return True


def _parse_json_object(value: str) -> dict | None:
    value = _json_object_text(value)
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_object_text(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        return stripped if _loads_single_generation_object(stripped) else None
    if stripped.startswith("["):
        return None
    fenced = _single_markdown_fence_body(stripped)
    if fenced is not None:
        return _safe_embedded_generation_object_text(fenced.strip())
    return _safe_embedded_generation_object_text(stripped)


def _single_markdown_fence_body(value: str) -> str | None:
    lines = value.splitlines()
    if len(lines) < 3 or not lines[0].strip().startswith("```") or lines[-1].strip() != "```":
        return None
    if any(line.strip().startswith("```") for line in lines[1:-1]):
        return None
    return "\n".join(lines[1:-1])


def _safe_embedded_generation_object_text(value: str) -> str | None:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict]] = []
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(value, index)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _looks_like_generation_object(parsed):
            candidates.append((index, end, parsed))
    if len(candidates) != 1:
        return None
    start, end, _parsed = candidates[0]
    prefix = value[:start].strip()
    suffix = value[end:].strip()
    if not _safe_json_prefix(prefix) or suffix:
        return None
    return value[start:end]


def _loads_single_generation_object(value: str) -> bool:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and _looks_like_generation_object(parsed)


def _looks_like_generation_object(value: dict) -> bool:
    return any(key in value for key in ("decision", "action", "status"))


def _safe_json_prefix(value: str) -> bool:
    if not value:
        return True
    normalized = " ".join(value.casefold().split()).rstrip(":：")
    return normalized in {
        "here is the result",
        "result",
        "json",
        "the result is",
    }


def _usage_int(usage, key: str) -> int | None:
    if not isinstance(usage, dict) or usage.get(key) is None:
        return None
    return int(usage[key])


def _endpoint_type(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127."):
        return "loopback"
    return "remote"


def _http_error_detail(exc: HTTPError, attempts: tuple[dict[str, object], ...]) -> ProviderErrorDetail:
    body = exc.read().decode("utf-8", errors="replace")
    summary = _sanitize_error_summary(body)
    reason = _classify_http_error(exc.code, summary)
    return ProviderErrorDetail(reason, http_status=exc.code, sdk_error_type=type(exc).__name__, error_summary=summary, attempts=attempts)


def _url_error_detail(exc: URLError, attempts: tuple[dict[str, object], ...]) -> ProviderErrorDetail:
    reason_obj = exc.reason
    reason = _classify_url_error(reason_obj)
    return ProviderErrorDetail(reason, sdk_error_type=type(reason_obj).__name__, error_summary=_sanitize_error_summary(str(reason_obj)), attempts=attempts)


def _classify_http_error(status: int, summary: str) -> str:
    lowered = summary.lower()
    if status in {401, 403}:
        return "authentication_failure" if status == 401 else "authorization_failure"
    if status == 404:
        if "model" in lowered:
            return "model_not_found"
        return "endpoint_not_found"
    if status == 408:
        return "provider_timeout"
    if status == 402 or "insufficient balance" in lowered or "insufficient_balance" in lowered:
        return "insufficient_balance"
    if status == 413 or "context" in lowered and "length" in lowered:
        return "context_length_exceeded"
    if status == 429:
        return "rate_limit"
    if status == 400:
        if "response_format" in lowered or "json_schema" in lowered or "schema" in lowered:
            return "unsupported_response_format"
        if "supported api model names" in lowered or "you passed" in lowered or "invalid model" in lowered:
            return "invalid_model"
        if "unsupported" in lowered or "unknown parameter" in lowered or "extra" in lowered:
            return "unsupported_parameter"
        if "model" in lowered and "not found" in lowered:
            return "model_not_found"
        return "unsupported_parameter"
    if 500 <= status < 600:
        return "server_error"
    return "unknown_provider_error"


def _classify_url_error(reason: object) -> str:
    if isinstance(reason, ConnectionRefusedError):
        return "network_failure"
    if isinstance(reason, TimeoutError):
        return "provider_timeout"
    if isinstance(reason, socket.gaierror):
        return "network_failure"
    if isinstance(reason, ssl.SSLError):
        return "network_failure"
    lowered = str(reason).lower()
    if "proxy" in lowered:
        return "network_failure"
    if "timed out" in lowered or "timeout" in lowered:
        return "provider_timeout"
    if "name or service not known" in lowered or "nodename nor servname" in lowered:
        return "network_failure"
    return "provider_error"


def _sanitize_error_summary(value: str) -> str:
    text = value.replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    for marker in ("Bearer ", "api_key=", "access_token="):
        if marker in text:
            head, tail = text.split(marker, 1)
            text = head + marker + "<redacted>" + tail.split(" ", 1)[-1] if " " in tail else head + marker + "<redacted>"
    return text[:500]
