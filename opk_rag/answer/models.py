from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from opk_rag.answerability import AnswerabilityDecision
from opk_rag.answer.grounding import GroundingValidationDecision
from opk_rag.answer.scope_execution import ScopeExecutionContract
from opk_rag.search.models import EvidenceBundle, EvidenceSignals, SearchResponse

RefusalReasonCode = Literal[
    "no_evidence",
    "insufficient_evidence",
    "no_relevant_context",
    "partial_evidence",
    "conflicting_evidence",
    "unsupported_inference",
    "false_premise",
    "below_score_threshold",
    "insufficient_distinct_sources",
    "invalid_evidence_bundle",
    "model_abstained",
    "invalid_citations",
    "unsupported_claims",
    "ungrounded_answer",
    "prompt_injection_detected",
    "answerable_generation_abstained",
    "partial_supported_scope_missing",
    "partial_unsupported_scope_fabricated",
    "partial_disclosure_missing",
    "false_premise_not_corrected",
    "false_premise_followed",
    "citation_not_supporting_claim",
    "generation_overreach",
]

SystemErrorCode = Literal[
    "provider_error",
    "database_error",
    "timeout",
    "invalid_runtime",
    "model_not_found",
    "invalid_model_output",
    "validation_error",
]


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    document_id: str
    relative_path: str
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None
    snippet: str


@dataclass(frozen=True)
class AnswerGenerationRequest:
    query: str
    evidence_bundle: EvidenceBundle
    prompt_version: str
    output_schema_version: str
    answerability: AnswerabilityDecision | None = None
    scope_execution_contract: ScopeExecutionContract | None = None


@dataclass(frozen=True)
class RawAnswerGeneration:
    provider_id: str
    model_id: str
    model_revision: str | None
    endpoint_type: Literal["loopback", "remote"]
    raw_text: str
    parsed: dict | None
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None = None
    request_attempts: tuple[dict[str, object], ...] = ()
    raw_envelope_hash: str | None = None


class AnswerGeneratorProvider(Protocol):
    @property
    def provider_id(self) -> str:
        ...

    @property
    def model_id(self) -> str:
        ...

    @property
    def model_revision(self) -> str | None:
        ...

    def generate_answer(self, request: AnswerGenerationRequest) -> RawAnswerGeneration:
        ...


@dataclass(frozen=True)
class AnswerValidationResult:
    ok: bool
    reason_code: RefusalReasonCode | None
    detail: str


class AnswerSystemError(RuntimeError):
    def __init__(self, code: SystemErrorCode, message: str, *, provider_error_detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.provider_error_detail = provider_error_detail


@dataclass(frozen=True)
class AnswerResponse:
    status: Literal["answered", "refused"]
    answerable: bool
    answer: str
    refusal_reason_code: RefusalReasonCode | None
    answerability: AnswerabilityDecision
    grounding: GroundingValidationDecision
    citations: tuple[Citation, ...]
    unsupported_claims: tuple[str, ...]
    search_response: SearchResponse
    provider_id: str
    model_id: str
    model_revision: str | None
    model_license: str
    runtime_base_url: str
    runtime_endpoint_type: Literal["loopback", "remote"]
    prompt_version: str
    output_schema_version: str
    evidence_serialization: str
    generation_latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model_decision: dict
    raw_generation_text: str | None = None
    finish_reason: str | None = None
    output_truncated: bool = False
    empty_output: bool = False
    request_attempts: tuple[dict[str, object], ...] = ()
    runtime_trace: dict | None = None
    # TASK-0259: preserve the original deterministic Answerability decision for
    # Controller/Shadow governance. answerability may be semantically refined
    # for Scope/Generation, but that refinement must not rewrite Agent state.
    controller_answerability: AnswerabilityDecision | None = None
