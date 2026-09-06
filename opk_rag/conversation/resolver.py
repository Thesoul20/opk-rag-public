from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Literal, Mapping

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.models import RawAnswerGeneration
from opk_rag.answer.provider import AnswerProviderError, OpenAICompatibleLocalChatProvider
from opk_rag.conversation.models import (
    FOLLOWUP_REWRITE_VERSION,
    STANDALONE_QUERY_SCHEMA_VERSION,
    ConversationContext,
    FollowupResolution,
)

STANDALONE_QUERY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_followup": {"type": "boolean"},
        "standalone_query": {"type": "string"},
        "referenced_turn_numbers": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        "referenced_citation_ids": {"type": "array", "items": {"type": "string", "pattern": "^C[1-9][0-9]*$"}, "uniqueItems": True},
        "reason": {"type": "string"},
    },
    "required": ["is_followup", "standalone_query", "referenced_turn_numbers", "referenced_citation_ids", "reason"],
}

RESOLVER_SYSTEM_PROMPT = """Resolve Chinese knowledge-base follow-up questions into standalone retrieval queries.
Conversation history is context only, not evidence. Do not answer the user question.
Do not add facts that are absent from the current question or conversation context.
Return only JSON matching the requested schema."""

ResolverStructuredOutputMode = Literal["auto", "json_schema", "json_object", "prompt_json"]
ResolvedStructuredOutputMode = Literal["json_schema", "json_object", "prompt_json"]


@dataclass(frozen=True)
class ResolverExecutionTrace:
    requested_mode: ResolverStructuredOutputMode
    resolved_mode: ResolvedStructuredOutputMode
    deterministic_passthrough: bool
    provider_request_count: int
    provider_response_count: int
    provider_transport_attempt_count: int
    initial_structured_valid: bool | None
    structural_repair_invoked: bool
    structural_repair_success: bool
    final_structured_valid: bool
    failure_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "deterministic_passthrough": self.deterministic_passthrough,
            "provider_request_count": self.provider_request_count,
            "provider_response_count": self.provider_response_count,
            "provider_transport_attempt_count": self.provider_transport_attempt_count,
            "initial_structured_valid": self.initial_structured_valid,
            "structural_repair_invoked": self.structural_repair_invoked,
            "structural_repair_success": self.structural_repair_success,
            "final_structured_valid": self.final_structured_valid,
            "failure_code": self.failure_code,
            "raw_provider_output_persisted": False,
            "hidden_reasoning_persisted": False,
        }


class HeuristicFollowupQueryResolver:
    provider_id = "heuristic_followup_resolver"
    model_id = "heuristic-followup-resolver-v1"
    model_revision = None

    def resolve(self, *, current_query: str, conversation_context: ConversationContext) -> FollowupResolution:
        stripped = current_query.strip()
        latest = conversation_context.turns[-1] if conversation_context.turns else None
        is_followup = latest is not None and _looks_like_followup(stripped)
        referenced_turn_numbers = (latest.turn_number,) if is_followup and latest is not None else ()
        referenced_citation_ids = tuple(c.citation_id for c in latest.citations) if is_followup and latest is not None else ()
        if is_followup:
            anchor = latest.standalone_query or latest.user_query
            standalone_query = f"{anchor}；追问：{stripped}"
            reason = "Heuristic follow-up marker matched; previous turn was used only to rewrite the retrieval query."
        else:
            standalone_query = stripped
            reason = "Query appears standalone."
        return FollowupResolution(
            standalone_query=standalone_query,
            is_followup=is_followup,
            referenced_turn_numbers=referenced_turn_numbers,
            referenced_citation_ids=referenced_citation_ids,
            resolution_reason=reason,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=FOLLOWUP_REWRITE_VERSION,
            output_schema_version=STANDALONE_QUERY_SCHEMA_VERSION,
            input_token_count=None,
            output_token_count=None,
            latency_ms=0,
        )


class OpenAICompatibleFollowupQueryResolver:
    def __init__(
        self,
        config: AnswerGenerationConfig,
        *,
        api_key: str | None = None,
        structured_output_mode: ResolverStructuredOutputMode | None = None,
        max_structural_repairs: int = 1,
        env: Mapping[str, str] = os.environ,
    ) -> None:
        if max_structural_repairs not in {0, 1}:
            raise ValueError("max_structural_repairs must be 0 or 1.")
        self.config = config
        self.provider = OpenAICompatibleLocalChatProvider(config, api_key=api_key)
        requested = structured_output_mode or _configured_resolver_mode(env)
        self.requested_structured_output_mode = requested
        self.structured_output_mode = _resolve_structured_output_mode(config, requested)
        self.max_structural_repairs = max_structural_repairs
        self.last_trace: ResolverExecutionTrace | None = None

    def resolve(self, *, current_query: str, conversation_context: ConversationContext) -> FollowupResolution:
        stripped = current_query.strip()
        if not stripped:
            raise AnswerProviderError("Resolver current_query must be a non-empty string.")
        if not conversation_context.turns:
            self.last_trace = ResolverExecutionTrace(
                requested_mode=self.requested_structured_output_mode,
                resolved_mode=self.structured_output_mode,
                deterministic_passthrough=True,
                provider_request_count=0,
                provider_response_count=0,
                provider_transport_attempt_count=0,
                initial_structured_valid=None,
                structural_repair_invoked=False,
                structural_repair_success=False,
                final_structured_valid=True,
            )
            return FollowupResolution(
                standalone_query=stripped,
                is_followup=False,
                referenced_turn_numbers=(),
                referenced_citation_ids=(),
                resolution_reason="No previous conversation turn exists; current query is deterministically standalone.",
                model_id="deterministic-no-context-passthrough-v1",
                model_revision=None,
                prompt_version=FOLLOWUP_REWRITE_VERSION,
                output_schema_version=STANDALONE_QUERY_SCHEMA_VERSION,
                input_token_count=0,
                output_token_count=0,
                latency_ms=0,
            )

        provider_requests = 0
        provider_responses = 0
        transport_attempts = 0
        initial_valid = False
        repair_invoked = False
        try:
            raw, attempts = self._request_resolution(
                current_query=stripped,
                conversation_context=conversation_context,
                repair=False,
            )
            provider_requests += 1
            provider_responses += 1
            transport_attempts += len(attempts)
            try:
                resolution = _resolution_from_raw(stripped, conversation_context, raw)
                initial_valid = True
                self.last_trace = ResolverExecutionTrace(
                    requested_mode=self.requested_structured_output_mode,
                    resolved_mode=self.structured_output_mode,
                    deterministic_passthrough=False,
                    provider_request_count=provider_requests,
                    provider_response_count=provider_responses,
                    provider_transport_attempt_count=transport_attempts,
                    initial_structured_valid=True,
                    structural_repair_invoked=False,
                    structural_repair_success=False,
                    final_structured_valid=True,
                )
                return resolution
            except AnswerProviderError:
                if self.max_structural_repairs == 0:
                    raise

            repair_invoked = True
            raw, attempts = self._request_resolution(
                current_query=stripped,
                conversation_context=conversation_context,
                repair=True,
            )
            provider_requests += 1
            provider_responses += 1
            transport_attempts += len(attempts)
            resolution = _resolution_from_raw(stripped, conversation_context, raw)
            self.last_trace = ResolverExecutionTrace(
                requested_mode=self.requested_structured_output_mode,
                resolved_mode=self.structured_output_mode,
                deterministic_passthrough=False,
                provider_request_count=provider_requests,
                provider_response_count=provider_responses,
                provider_transport_attempt_count=transport_attempts,
                initial_structured_valid=initial_valid,
                structural_repair_invoked=True,
                structural_repair_success=True,
                final_structured_valid=True,
            )
            return resolution
        except AnswerProviderError as exc:
            detail = exc.detail
            if detail is not None and detail.attempts:
                # A Provider transport failure may occur before _request_resolution returns.
                transport_attempts += len(detail.attempts)
            if provider_requests == 0 or (repair_invoked and provider_requests == 1):
                provider_requests += 1
            failure_code = detail.reason_code if detail is not None else "resolver_structured_output_invalid"
            self.last_trace = ResolverExecutionTrace(
                requested_mode=self.requested_structured_output_mode,
                resolved_mode=self.structured_output_mode,
                deterministic_passthrough=False,
                provider_request_count=provider_requests,
                provider_response_count=provider_responses,
                provider_transport_attempt_count=transport_attempts,
                initial_structured_valid=initial_valid,
                structural_repair_invoked=repair_invoked,
                structural_repair_success=False,
                final_structured_valid=False,
                failure_code=failure_code,
            )
            raise

    def _request_resolution(
        self,
        *,
        current_query: str,
        conversation_context: ConversationContext,
        repair: bool,
    ) -> tuple[RawAnswerGeneration, list[dict[str, object]]]:
        payload = self._payload(
            current_query=current_query,
            conversation_context=conversation_context,
            repair=repair,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()
        raw_response, attempts = self.provider._post_with_retries(endpoint, body, headers)
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            envelope = json.loads(raw_response)
        except json.JSONDecodeError:
            envelope = {}
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""
        if not isinstance(content, str):
            content = ""
        # Transport success with malformed/empty structured content is a structural
        # failure, not a transport failure. Return parsed=None so resolve() can use
        # its single bounded structural repair before failing closed.
        parsed = _parse_resolution_json(content)
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        raw = RawAnswerGeneration(
            provider_id=self.provider.provider_id,
            model_id=self.provider.model_id,
            model_revision=self.provider.model_revision,
            endpoint_type=self.provider.endpoint_type,
            raw_text=content,
            parsed=parsed,
            latency_ms=latency_ms,
            prompt_tokens=_usage_int(usage, "prompt_tokens"),
            completion_tokens=_usage_int(usage, "completion_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            request_attempts=tuple(attempts),
        )
        return raw, attempts

    def _payload(
        self,
        *,
        current_query: str,
        conversation_context: ConversationContext,
        repair: bool,
    ) -> dict[str, object]:
        system_prompt = RESOLVER_SYSTEM_PROMPT
        if repair:
            system_prompt += (
                "\nA previous attempt failed local JSON/schema validation. "
                "Return exactly one valid JSON object with every required field and no Markdown fence or extra prose."
            )
        payload: dict[str, object] = {
            "model": self.provider.model_id,
            "temperature": 0,
            "max_tokens": min(self.config.max_output_tokens, 512),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_resolver_prompt(current_query, conversation_context)},
            ],
        }
        if self.structured_output_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "opk_rag_standalone_query",
                    "strict": True,
                    "schema": STANDALONE_QUERY_JSON_SCHEMA,
                },
            }
        elif self.structured_output_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.config.thinking_mode is not None:
            # Reuse the same declared Provider capability/configuration as answer
            # generation. Omitting this can change the endpoint response shape even
            # when transport and response_format are otherwise compatible.
            payload["thinking"] = {"type": self.config.thinking_mode}
        return payload


def _configured_resolver_mode(env: Mapping[str, str]) -> ResolverStructuredOutputMode:
    raw = env.get("OPK_RAG_CONVERSATION_STRUCTURED_OUTPUT_MODE", "auto").strip() or "auto"
    if raw not in {"auto", "json_schema", "json_object", "prompt_json"}:
        raise ValueError("OPK_RAG_CONVERSATION_STRUCTURED_OUTPUT_MODE must be auto, json_schema, json_object, or prompt_json.")
    return raw  # type: ignore[return-value]


def _resolve_structured_output_mode(
    config: AnswerGenerationConfig,
    requested: ResolverStructuredOutputMode,
) -> ResolvedStructuredOutputMode:
    if requested == "auto":
        # The existing answer-provider configuration is the deterministic capability
        # declaration for this OpenAI-compatible endpoint. Transport compatibility
        # does not imply strict JSON Schema support.
        return config.response_format_type
    return requested


def render_resolver_prompt(current_query: str, context: ConversationContext) -> str:
    payload = {
        "schema_version": STANDALONE_QUERY_SCHEMA_VERSION,
        "current_query": current_query,
        "rules": [
            "Rewrite only if the current query depends on previous turns.",
            "Conversation history can clarify references but is not knowledge-base evidence.",
            "Only current-turn retrieval evidence may be cited later.",
            "Use citation IDs only to identify references mentioned by the user; do not treat citation text as evidence.",
            "If the current query mentions a previous citation ID, rewrite with the cited turn topic plus the citation relative_path and heading words.",
            "Do not leave bare phrases like 'previous C1' as the standalone retrieval query.",
            "Preserve the current question's propositional polarity and answer orientation. Do not invert a negative confirmation into a positive alternative question or vice versa.",
            "Preserve explicit negation, contrast, uncertainty, quantitative requests, and yes/no framing when rewriting; add context anchors without changing what a yes/no answer would mean.",
            "For topic-switch-and-return questions, recover the referenced earlier topic while keeping the current turn's predicate and polarity intact.",
            "Prefer the immediately preceding turn when it introduced a new subproblem and the current query uses short references such as '之前', '这个', '那', '又是另一类问题', or asks whether that newly introduced problem relates to a cause. Do not jump back to an older topic merely because it shares the same product name.",
            "When the current query asks for an exact version, number, date, or measured value, preserve that exact-target request verbatim in the standalone query.",
        ],
        "conversation_context": [
            {
                "turn_number": turn.turn_number,
                "user_query": turn.user_query,
                "standalone_query": turn.standalone_query,
                "answer_summary_for_context": turn.answer_text,
                "answer_decision": turn.answer_decision,
                "abstention_reason": turn.abstention_reason,
                "citations": [
                    {
                        "citation_id": citation.citation_id,
                        "relative_path": citation.relative_path,
                        "heading_path": list(citation.heading_path),
                        "start_line": citation.start_line,
                        "end_line": citation.end_line,
                    }
                    for citation in turn.citations
                ],
            }
            for turn in context.turns
        ],
        "required_output": STANDALONE_QUERY_JSON_SCHEMA,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _resolution_from_raw(current_query: str, context: ConversationContext, raw: RawAnswerGeneration) -> FollowupResolution:
    parsed = raw.parsed
    if not isinstance(parsed, dict):
        raise AnswerProviderError("Resolver returned invalid JSON.")
    required_keys = {"is_followup", "standalone_query", "referenced_turn_numbers", "referenced_citation_ids", "reason"}
    if set(parsed) != required_keys:
        raise AnswerProviderError("Resolver JSON fields do not match the required schema.")
    standalone_query = parsed.get("standalone_query")
    is_followup = parsed.get("is_followup")
    referenced_turn_numbers = parsed.get("referenced_turn_numbers")
    referenced_citation_ids = parsed.get("referenced_citation_ids")
    reason = parsed.get("reason")
    if not isinstance(standalone_query, str) or not standalone_query.strip():
        raise AnswerProviderError("Resolver standalone_query must be a non-empty string.")
    if not isinstance(is_followup, bool):
        raise AnswerProviderError("Resolver is_followup must be boolean.")
    valid_turns = {turn.turn_number for turn in context.turns}
    if (
        not isinstance(referenced_turn_numbers, list)
        or len(referenced_turn_numbers) != len(set(referenced_turn_numbers))
        or any(type(value) is not int or value not in valid_turns for value in referenced_turn_numbers)
    ):
        raise AnswerProviderError("Resolver referenced_turn_numbers are invalid for the context.")
    valid_citations = {citation.citation_id for turn in context.turns for citation in turn.citations}
    if (
        not isinstance(referenced_citation_ids, list)
        or len(referenced_citation_ids) != len(set(referenced_citation_ids))
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"C[1-9][0-9]*", value) is None
            or value not in valid_citations
            for value in referenced_citation_ids
        )
    ):
        raise AnswerProviderError("Resolver referenced_citation_ids are invalid for the context.")
    if not isinstance(reason, str) or not reason.strip():
        raise AnswerProviderError("Resolver reason must be a non-empty string.")
    if not is_followup:
        referenced_turn_numbers = []
        referenced_citation_ids = []
    return FollowupResolution(
        standalone_query=standalone_query.strip(),
        is_followup=is_followup,
        referenced_turn_numbers=tuple(sorted(set(referenced_turn_numbers))),
        referenced_citation_ids=tuple(referenced_citation_ids),
        resolution_reason=reason.strip(),
        model_id=raw.model_id,
        model_revision=raw.model_revision,
        prompt_version=FOLLOWUP_REWRITE_VERSION,
        output_schema_version=STANDALONE_QUERY_SCHEMA_VERSION,
        input_token_count=raw.prompt_tokens,
        output_token_count=raw.completion_tokens,
        latency_ms=raw.latency_ms,
    )


def _parse_resolution_json(value: str) -> dict | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _usage_int(usage, key: str) -> int | None:
    if not isinstance(usage, dict) or usage.get(key) is None:
        return None
    return int(usage[key])


def _looks_like_followup(query: str) -> bool:
    normalized = query.strip().casefold()
    if re.search(r"\b(c[1-9][0-9]*)\b", normalized):
        return True
    markers = ("那", "这个", "这些", "它", "它们", "上述", "上面", "前面", "刚才", "继续", "还有", "哪些", "为什么", "怎么处理")
    return any(marker in normalized for marker in markers) and len(normalized) <= 80
