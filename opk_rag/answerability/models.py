from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnswerabilityStatus = Literal["answerable", "partially_answerable", "unanswerable"]

AnswerabilityReasonCode = Literal[
    "answerable",
    "disabled",
    "insufficient_evidence",
    "no_relevant_context",
    "partial_evidence",
    "conflicting_evidence",
    "unsupported_inference",
    "false_premise",
    "invalid_evidence_bundle",
    "prompt_injection_detected",
    # Compatibility reason codes retained for existing callers and configs.
    "no_evidence",
    "below_score_threshold",
    "insufficient_distinct_sources",
]


@dataclass(frozen=True, init=False)
class AnswerabilityDecision:
    status: AnswerabilityStatus
    reason_code: AnswerabilityReasonCode
    reason: str
    confidence: float
    evidence_chunk_ids: tuple[str, ...]
    evidence_score: float | None
    evidence_count: int
    considered_evidence_count: int
    diagnostics: dict[str, object]

    def __init__(
        self,
        status: AnswerabilityStatus | bool | None = None,
        reason_code: AnswerabilityReasonCode = "answerable",
        reason: str = "",
        confidence: float | int = 0.0,
        evidence_chunk_ids: tuple[str, ...] | int = (),
        evidence_score: float | dict[str, object] | None = None,
        evidence_count: int | None = None,
        considered_evidence_count: int | None = None,
        diagnostics: dict[str, object] | None = None,
        *,
        answerable: bool | None = None,
    ) -> None:
        if answerable is not None:
            status = answerable
        if status is None:
            status = "answerable"
        if isinstance(status, bool):
            old_evidence_count = int(evidence_count if evidence_count is not None else confidence)
            old_considered_count = int(considered_evidence_count if considered_evidence_count is not None else evidence_chunk_ids)
            old_diagnostics = diagnostics or (evidence_score if isinstance(evidence_score, dict) else {})
            status_value: AnswerabilityStatus = "answerable" if status else "unanswerable"
            confidence_value = 1.0 if status else 0.0
            evidence_ids: tuple[str, ...] = ()
            evidence_score_value = None
            evidence_count_value = old_evidence_count
            considered_count_value = old_considered_count
            diagnostics_value = old_diagnostics
        else:
            status_value = status
            confidence_value = float(confidence)
            evidence_ids = tuple(evidence_chunk_ids) if isinstance(evidence_chunk_ids, tuple) else ()
            evidence_score_value = evidence_score if isinstance(evidence_score, float) or evidence_score is None else None
            evidence_count_value = int(evidence_count or 0)
            considered_count_value = int(considered_evidence_count or 0)
            diagnostics_value = diagnostics or {}
        object.__setattr__(self, "status", status_value)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, confidence_value)))
        object.__setattr__(self, "evidence_chunk_ids", evidence_ids)
        object.__setattr__(self, "evidence_score", evidence_score_value)
        object.__setattr__(self, "evidence_count", evidence_count_value)
        object.__setattr__(self, "considered_evidence_count", considered_count_value)
        object.__setattr__(self, "diagnostics", diagnostics_value)

    @property
    def answerable(self) -> bool:
        return self.status in {"answerable", "partially_answerable"}


@dataclass(frozen=True)
class AnswerabilityConfig:
    enabled: bool = True
    min_evidence: int = 1
    min_distinct_sources: int = 1
    min_score: float | None = None
    min_context_tokens: int = 1
    min_full_coverage: float = 0.6
    min_partial_coverage: float = 0.25
    low_score_partial_margin: float = 0.15

    def __post_init__(self) -> None:
        if self.min_evidence < 1:
            raise ValueError("answerability min_evidence must be >= 1.")
        if self.min_distinct_sources < 1:
            raise ValueError("answerability min_distinct_sources must be >= 1.")
        if self.min_score is not None and not 0.0 <= self.min_score <= 1.0:
            raise ValueError("answerability min_score must be between 0.0 and 1.0 when set.")
        if self.min_context_tokens < 1:
            raise ValueError("answerability min_context_tokens must be >= 1.")
        if not 0.0 <= self.min_partial_coverage <= self.min_full_coverage <= 1.0:
            raise ValueError("answerability coverage thresholds must satisfy 0 <= partial <= full <= 1.")
        if not 0.0 <= self.low_score_partial_margin <= 1.0:
            raise ValueError("answerability low_score_partial_margin must be between 0.0 and 1.0.")
