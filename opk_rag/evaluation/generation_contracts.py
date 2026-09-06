from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import re
import time
from typing import Any
from urllib.request import Request, urlopen

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.grounding import GroundingValidator
from opk_rag.answer.prompt import ANSWER_RESPONSE_JSON_SCHEMA, SYSTEM_PROMPT, render_user_prompt
from opk_rag.answer.provider import AnswerProviderError, build_chat_completion_payload
from opk_rag.answer.service import _parse_model_decision
from opk_rag.answerability import AnswerabilityDecision
from opk_rag.evaluation.generation_abstention import classify_abstention
from opk_rag.evaluation.generation_stability import generation_parameters_payload, normalize_output, stable_hash
from opk_rag.search.models import EvidenceBundle


class ContractLevel(str, Enum):
    LEVEL_0 = "level_0"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"


class ResponseMode(str, Enum):
    TEXT = "text"
    PLAIN_JSON = "plain_json"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


SIMPLE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["answer", "abstain"]},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string", "pattern": "^C[1-9][0-9]*$"}},
    },
    "required": ["action", "answer", "citations"],
}


@dataclass(frozen=True)
class GenerationContract:
    level: ContractLevel
    response_mode: ResponseMode
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any] | None
    fingerprint: str


def build_generation_contract(
    *,
    level: ContractLevel,
    response_mode: ResponseMode,
    question: str,
    bundle: EvidenceBundle,
    config: AnswerGenerationConfig,
    answerability: AnswerabilityDecision | None = None,
) -> GenerationContract:
    if level == ContractLevel.LEVEL_3:
        system = SYSTEM_PROMPT
        user = render_user_prompt(question, bundle, output_schema_version=config.output_schema_version, answerability=answerability)
        schema = ANSWER_RESPONSE_JSON_SCHEMA
    else:
        evidence = render_plain_evidence(bundle)
        if level == ContractLevel.LEVEL_0:
            system = "Answer using only the supplied evidence. If evidence is insufficient, reply exactly: ABSTAIN."
            user = f"问题：{question}\n\n证据：\n{evidence}\n\n请直接回答。证据不足时只输出 ABSTAIN。"
            schema = None
        elif level == ContractLevel.LEVEL_1:
            system = "Answer using only the supplied evidence. Cite supported facts with citation IDs like [C1]. If evidence is insufficient, reply exactly: ABSTAIN."
            user = f"问题：{question}\n\n证据：\n{evidence}\n\n请用普通文本回答，并在每个事实句后写 [C1] 这类引用。证据不足时只输出 ABSTAIN。"
            schema = None
        else:
            system = "Return only a JSON object. Use only the supplied evidence. If evidence is insufficient, set action to abstain."
            user = (
                f"问题：{question}\n\n证据：\n{evidence}\n\n"
                '输出 JSON：{"action":"answer|abstain","answer":"string","citations":["C1"]}。'
                "answer 和 abstain 互斥；answer 必须只包含有证据和 citation 支持的内容。"
            )
            schema = SIMPLE_JSON_SCHEMA
    return GenerationContract(
        level=level,
        response_mode=response_mode,
        system_prompt=system,
        user_prompt=user,
        output_schema=schema,
        fingerprint=stable_hash({"level": level.value, "response_mode": response_mode.value, "system": system, "user": user, "schema": schema}),
    )


def build_experiment_payload(config: AnswerGenerationConfig, contract: GenerationContract) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model_id,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "max_tokens": config.max_output_tokens,
        "messages": [
            {"role": "system", "content": contract.system_prompt},
            {"role": "user", "content": contract.user_prompt},
        ],
    }
    if config.top_k is not None:
        payload["top_k"] = config.top_k
    if config.seed is not None:
        payload["seed"] = config.seed
    if config.stop_sequences:
        payload["stop"] = list(config.stop_sequences)
    if contract.response_mode == ResponseMode.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    elif contract.response_mode == ResponseMode.JSON_SCHEMA:
        schema = contract.output_schema or ANSWER_RESPONSE_JSON_SCHEMA
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "opk_rag_generation_contract", "strict": True, "schema": schema}}
    return payload


def run_contract_request(config: AnswerGenerationConfig, payload: dict[str, Any], *, api_key: str | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    try:
        with urlopen(Request(config.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST"), timeout=config.timeout_seconds) as response:
            raw_envelope = response.read().decode("utf-8")
    except Exception as exc:
        return {
            "provider_success": False,
            "provider_error": type(exc).__name__,
            "provider_error_summary": str(exc)[:500],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "raw_output": None,
            "finish_reason": None,
        }
    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        envelope = json.loads(raw_envelope)
        choice = envelope["choices"][0]
        raw_output = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        usage = envelope.get("usage") if isinstance(envelope, dict) else {}
    except Exception as exc:
        return {"provider_success": False, "provider_error": "invalid_chat_completion_envelope", "provider_error_summary": str(exc), "latency_ms": latency_ms, "raw_output": raw_envelope, "finish_reason": None}
    return {
        "provider_success": True,
        "latency_ms": latency_ms,
        "raw_output": raw_output,
        "finish_reason": finish_reason,
        "prompt_tokens": _usage_int(usage, "prompt_tokens"),
        "completion_tokens": _usage_int(usage, "completion_tokens"),
        "total_tokens": _usage_int(usage, "total_tokens"),
    }


def parse_contract_output(level: ContractLevel, raw_output: str | None) -> dict[str, Any]:
    raw = raw_output or ""
    normalized = normalize_output(raw)
    if not raw.strip():
        return {"parser_success": False, "action": "abstain", "parse_failure_reason": "empty_output", "answer": "", "citations": []}
    if level in {ContractLevel.LEVEL_0, ContractLevel.LEVEL_1}:
        explicit_abstain = normalized.strip().casefold() == "abstain" or "证据不足" in raw or "无法回答" in raw
        citations = sorted(set(re.findall(r"\[?(C[1-9][0-9]*)\]?", raw)))
        return {"parser_success": True, "action": "abstain" if explicit_abstain else "answer", "parse_failure_reason": None, "answer": "" if explicit_abstain else raw.strip(), "citations": citations}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"parser_success": False, "action": "abstain", "parse_failure_reason": "malformed_json", "answer": "", "citations": []}
    if not isinstance(parsed, dict):
        return {"parser_success": False, "action": "abstain", "parse_failure_reason": "malformed_json", "answer": "", "citations": []}
    if level == ContractLevel.LEVEL_2:
        missing = [key for key in ("action", "answer", "citations") if key not in parsed]
        if missing:
            return {"parser_success": False, "action": "abstain", "parse_failure_reason": "missing_required_fields", "answer": "", "citations": []}
        action = parsed.get("action")
        if action not in {"answer", "abstain"}:
            return {"parser_success": False, "action": "abstain", "parse_failure_reason": "invalid_decision", "answer": "", "citations": []}
        citations = parsed.get("citations") if isinstance(parsed.get("citations"), list) else []
        return {"parser_success": True, "action": action, "parse_failure_reason": None, "answer": str(parsed.get("answer") or ""), "citations": [str(value) for value in citations]}
    try:
        decision = _parse_model_decision(parsed)
    except ValueError as exc:
        return {"parser_success": False, "action": "abstain", "parse_failure_reason": str(exc), "answer": "", "citations": []}
    return {"parser_success": True, "action": decision["decision"], "parse_failure_reason": None, "answer": decision["answer"], "citations": decision["citations"], "model_decision": decision}


def evaluate_contract_output(*, level: ContractLevel, raw_output: str | None, bundle: EvidenceBundle, finish_reason: Any) -> dict[str, Any]:
    parsed = parse_contract_output(level, raw_output)
    action = parsed["action"]
    citation_present = bool(parsed["citations"])
    grounding_status = "not_evaluated"
    grounding_reason = None
    unsupported_answer = False
    if action == "answer":
        grounding = GroundingValidator().validate(answer_text=parsed["answer"], evidence_bundle=bundle, structured_citations=parsed["citations"])
        grounding_status = "grounded" if grounding.valid else str(grounding.reason_code or grounding.status)
        grounding_reason = grounding.reason
        unsupported_answer = not grounding.valid
    attribution = classify_abstention(
        raw_output=raw_output,
        parsed_output=None,
        model_decision={"decision": "abstain" if action == "abstain" else "answer", "reason": parsed.get("parse_failure_reason")},
        generation_status="abstention" if action == "abstain" else "success",
        generation_reason=str(parsed.get("parse_failure_reason") or ""),
        final_action="abstain" if action == "abstain" else "answer",
        finish_reason=finish_reason,
        output_truncated=finish_reason == "length",
        empty_output=not bool((raw_output or "").strip()),
    )
    return {
        **parsed,
        "citation_presence": citation_present,
        "grounding_status": grounding_status,
        "grounding_reason": grounding_reason,
        "final_action": "answer" if action == "answer" and grounding_status == "grounded" else "abstain",
        "unsupported_answer": unsupported_answer,
        "abstention_category": attribution.category if action == "abstain" else None,
        "explicit_abstain": attribution.explicit_abstention_marker if action == "abstain" else False,
        "parser_inferred_abstain": attribution.parser_inferred_abstention_marker if action == "abstain" else False,
    }


def current_level3_payload(config: AnswerGenerationConfig, question: str, bundle: EvidenceBundle) -> dict[str, Any]:
    from opk_rag.answer.models import AnswerGenerationRequest

    return build_chat_completion_payload(config, AnswerGenerationRequest(question, bundle, config.prompt_version, config.output_schema_version))


def render_plain_evidence(bundle: EvidenceBundle) -> str:
    lines: list[str] = []
    for index, item in enumerate(bundle.items, start=1):
        heading = " > ".join(item.heading_path)
        source = f"{item.relative_path}" + (f" / {heading}" if heading else "")
        lines.append(f"[C{index}] {source}\n{item.content}")
    return "\n\n".join(lines)


def config_for_model(config: AnswerGenerationConfig, *, model_id: str, model_revision: str | None = None, max_retries: int = 0) -> AnswerGenerationConfig:
    return replace(config, model_id=model_id, model_revision=model_revision, max_retries=max_retries)


def request_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_roles": [row.get("role") for row in payload.get("messages", [])],
        "message_character_counts": [len(str(row.get("content") or "")) for row in payload.get("messages", [])],
        "response_format": payload.get("response_format", {"type": "text"}).get("type") if isinstance(payload.get("response_format", {"type": "text"}), dict) else payload.get("response_format"),
        "generation_parameters": {key: payload.get(key) for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed") if key in payload},
    }


def _usage_int(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict) or usage.get(key) is None:
        return None
    return int(usage[key])
