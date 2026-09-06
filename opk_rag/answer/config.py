from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Mapping

from opk_rag.answerability import AnswerabilityConfig
from opk_rag.answer.grounding import GroundingValidationConfig

ANSWER_PROMPT_VERSION = "answer-prompt-v4"
ANSWER_OUTPUT_SCHEMA_VERSION = "answer-response-v3"
SCOPE_UNITS_OUTPUT_SCHEMA_VERSION = "answer-response-v4-scope-units"
DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_LLM_MODEL_ID = "qwen2.5:7b-instruct"
DEFAULT_LOCAL_LLM_MODEL_REVISION = None
DEFAULT_LOCAL_LLM_LICENSE = "Apache-2.0"


class AnswerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AnswerGenerationConfig:
    provider_id: str = "openai_compatible_local_chat"
    base_url: str = DEFAULT_LOCAL_LLM_BASE_URL
    model_id: str = DEFAULT_LOCAL_LLM_MODEL_ID
    model_revision: str | None = DEFAULT_LOCAL_LLM_MODEL_REVISION
    model_license: str = DEFAULT_LOCAL_LLM_LICENSE
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_output_tokens: int = 1024
    response_format_type: Literal["json_schema", "json_object"] = "json_schema"
    thinking_mode: Literal["enabled", "disabled"] | None = None
    seed: int | None = None
    stop_sequences: tuple[str, ...] = ()
    max_retries: int = 1
    allow_remote: bool = False
    prompt_version: str = ANSWER_PROMPT_VERSION
    output_schema_version: str = ANSWER_OUTPUT_SCHEMA_VERSION
    provider_parameter_policy: Literal["compat", "deepseek_v4"] = "compat"
    min_evidence_items: int = 1
    min_context_tokens: int = 1
    answerability: AnswerabilityConfig = AnswerabilityConfig()
    grounding: GroundingValidationConfig = GroundingValidationConfig()

    def __post_init__(self) -> None:
        if self.provider_id != "openai_compatible_local_chat":
            raise AnswerConfigError("provider_id must be openai_compatible_local_chat.")
        if not self.base_url.strip():
            raise AnswerConfigError("base_url must not be empty.")
        if not self.model_id.strip():
            raise AnswerConfigError("model_id must not be empty.")
        if self.timeout_seconds <= 0:
            raise AnswerConfigError("timeout_seconds must be > 0.")
        if not 0.0 <= self.temperature <= 2.0:
            raise AnswerConfigError("temperature must be between 0.0 and 2.0.")
        if not 0.0 < self.top_p <= 1.0:
            raise AnswerConfigError("top_p must be > 0.0 and <= 1.0.")
        if self.top_k is not None and self.top_k < 1:
            raise AnswerConfigError("top_k must be >= 1 when set.")
        if self.repetition_penalty <= 0.0:
            raise AnswerConfigError("repetition_penalty must be > 0.0.")
        if self.max_output_tokens < 1:
            raise AnswerConfigError("max_output_tokens must be >= 1.")
        if self.response_format_type not in {"json_schema", "json_object"}:
            raise AnswerConfigError("response_format_type must be json_schema or json_object.")
        if self.thinking_mode is not None and self.thinking_mode not in {"enabled", "disabled"}:
            raise AnswerConfigError("thinking_mode must be enabled, disabled, or unset.")
        if self.seed is not None and self.seed < 0:
            raise AnswerConfigError("seed must be >= 0 when set.")
        if self.max_retries < 0:
            raise AnswerConfigError("max_retries must be >= 0.")
        if self.provider_parameter_policy not in {"compat", "deepseek_v4"}:
            raise AnswerConfigError("provider_parameter_policy must be compat or deepseek_v4.")
        if self.min_evidence_items < 1:
            raise AnswerConfigError("min_evidence_items must be >= 1.")
        if self.min_context_tokens < 1:
            raise AnswerConfigError("min_context_tokens must be >= 1.")


def load_answer_generation_config(env: Mapping[str, str] = os.environ) -> AnswerGenerationConfig:
    timeout = env.get("OPK_RAG_LLM_TIMEOUT_SECONDS", "60").strip()
    max_output_tokens = env.get("OPK_RAG_LLM_MAX_OUTPUT_TOKENS", "1024").strip()
    min_evidence = int(env.get("OPK_RAG_ANSWERABILITY_MIN_EVIDENCE", env.get("OPK_RAG_ANSWER_MIN_EVIDENCE_ITEMS", "1")).strip())
    min_score_raw = env.get("OPK_RAG_ANSWERABILITY_MIN_SCORE", "").strip()
    min_coverage_raw = env.get("OPK_RAG_GROUNDING_MIN_COVERAGE", "").strip()
    response_format_type = env.get("OPK_RAG_LLM_RESPONSE_FORMAT", "").strip() or "json_schema"
    answerability = AnswerabilityConfig(
        enabled=_bool_env(env.get("OPK_RAG_ANSWERABILITY_ENABLED", "true")),
        min_evidence=min_evidence,
        min_distinct_sources=int(env.get("OPK_RAG_ANSWERABILITY_MIN_DISTINCT_SOURCES", "1").strip()),
        min_score=float(min_score_raw) if min_score_raw else None,
        min_context_tokens=int(env.get("OPK_RAG_ANSWER_MIN_CONTEXT_TOKENS", "1").strip()),
    )
    grounding = GroundingValidationConfig(
        enabled=_bool_env(env.get("OPK_RAG_GROUNDING_VALIDATION_ENABLED", "true")),
        require_citations=_bool_env(env.get("OPK_RAG_GROUNDING_REQUIRE_CITATIONS", "true")),
        min_citations=int(env.get("OPK_RAG_GROUNDING_MIN_CITATIONS", "1").strip()),
        min_coverage=float(min_coverage_raw) if min_coverage_raw else None,
        max_repair_attempts=int(env.get("OPK_RAG_GROUNDING_MAX_REPAIR_ATTEMPTS", "0").strip()),
    )
    model_id = (
        env.get("OPK_RAG_LLM_MODEL", "")
        .strip()
        or env.get("OPK_RAG_LLM_MODEL_ID", DEFAULT_LOCAL_LLM_MODEL_ID).strip()
    )
    return AnswerGenerationConfig(
        provider_id=env.get("OPK_RAG_ANSWER_PROVIDER", "openai_compatible_local_chat").strip(),
        base_url=env.get("OPK_RAG_LLM_BASE_URL", DEFAULT_LOCAL_LLM_BASE_URL).strip(),
        model_id=model_id,
        model_revision=(env.get("OPK_RAG_LLM_MODEL_REVISION") or DEFAULT_LOCAL_LLM_MODEL_REVISION or "").strip() or None,
        model_license=env.get("OPK_RAG_LLM_MODEL_LICENSE", DEFAULT_LOCAL_LLM_LICENSE).strip(),
        timeout_seconds=float(timeout),
        temperature=float(env.get("OPK_RAG_LLM_TEMPERATURE", "0").strip()),
        top_p=float(env.get("OPK_RAG_LLM_TOP_P", "1").strip()),
        top_k=int(env["OPK_RAG_LLM_TOP_K"].strip()) if env.get("OPK_RAG_LLM_TOP_K", "").strip() else None,
        repetition_penalty=float(env.get("OPK_RAG_LLM_REPETITION_PENALTY", "1").strip()),
        max_output_tokens=int(max_output_tokens),
        response_format_type=response_format_type,
        thinking_mode=env.get("OPK_RAG_LLM_THINKING_MODE", "").strip() or None,
        seed=int(env["OPK_RAG_LLM_SEED"].strip()) if env.get("OPK_RAG_LLM_SEED", "").strip() else None,
        stop_sequences=tuple(value for value in env.get("OPK_RAG_LLM_STOP", "").split("\u0000") if value),
        max_retries=int(env.get("OPK_RAG_LLM_MAX_RETRIES", "1").strip()),
        allow_remote=_bool_env(env.get("OPK_RAG_LLM_ALLOW_REMOTE", "false")),
        prompt_version=env.get("OPK_RAG_ANSWER_PROMPT_VERSION", ANSWER_PROMPT_VERSION).strip(),
        output_schema_version=env.get("OPK_RAG_ANSWER_OUTPUT_SCHEMA_VERSION", ANSWER_OUTPUT_SCHEMA_VERSION).strip(),
        provider_parameter_policy=env.get("OPK_RAG_LLM_PROVIDER_PARAMETER_POLICY", "compat").strip(),
        min_evidence_items=min_evidence,
        min_context_tokens=int(env.get("OPK_RAG_ANSWER_MIN_CONTEXT_TOKENS", "1").strip()),
        answerability=answerability,
        grounding=grounding,
    )


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise AnswerConfigError(f"Invalid boolean value: {value}")
