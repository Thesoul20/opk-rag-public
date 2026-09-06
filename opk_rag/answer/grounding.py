from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from opk_rag.answer.citation_parser import CitationParseResult
from opk_rag.answer.citation_parser import parse_citation_markers
from opk_rag.search.models import EvidenceBundle

GroundingReasonCode = Literal[
    "grounded",
    "disabled",
    "not_applicable",
    "empty_answer",
    "missing_citations",
    "invalid_citation_format",
    "unknown_citation",
    "citation_out_of_range",
    "citation_set_mismatch",
    "insufficient_citation_coverage",
    "historical_citation_reference",
    "invalid_evidence_reference",
    "ungrounded_answer",
    "validation_error",
    "partial_supported_scope_missing",
    "partial_unsupported_scope_fabricated",
    "partial_disclosure_missing",
    "false_premise_not_corrected",
    "false_premise_followed",
    "citation_not_supporting_claim",
    "generation_overreach",
]


@dataclass(frozen=True)
class GroundingValidationConfig:
    enabled: bool = True
    require_citations: bool = True
    min_citations: int = 1
    min_coverage: float | None = None
    max_repair_attempts: int = 0

    def __post_init__(self) -> None:
        if self.min_citations < 0:
            raise ValueError("min_citations must be >= 0.")
        if self.min_coverage is not None and not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0.0 and 1.0.")
        if self.max_repair_attempts not in {0, 1}:
            raise ValueError("max_repair_attempts must be 0 or 1.")


@dataclass(frozen=True)
class GroundingValidationDecision:
    valid: bool
    status: Literal["grounded", "refused", "disabled", "not_applicable", "error"]
    reason_code: GroundingReasonCode
    reason: str
    cited_ids: tuple[str, ...]
    valid_cited_ids: tuple[str, ...]
    invalid_cited_ids: tuple[str, ...]
    available_evidence_ids: tuple[str, ...]
    citation_coverage: float | None
    diagnostics: dict[str, object]


class GroundingValidator:
    def __init__(self, config: GroundingValidationConfig | None = None) -> None:
        self.config = config or GroundingValidationConfig()

    def not_applicable(self, bundle: EvidenceBundle | None = None) -> GroundingValidationDecision:
        available = _available_ids(bundle) if bundle is not None else ()
        return GroundingValidationDecision(
            valid=False,
            status="not_applicable",
            reason_code="not_applicable",
            reason="Grounding validation does not apply because answer generation did not run.",
            cited_ids=(),
            valid_cited_ids=(),
            invalid_cited_ids=(),
            available_evidence_ids=available,
            citation_coverage=None,
            diagnostics={"validation_enabled": self.config.enabled},
        )

    def validate(
        self,
        *,
        answer_text: str,
        evidence_bundle: EvidenceBundle,
        structured_citations: object,
    ) -> GroundingValidationDecision:
        available = _available_ids(evidence_bundle)
        if not self.config.enabled:
            return GroundingValidationDecision(
                valid=True,
                status="disabled",
                reason_code="disabled",
                reason="Grounding validation is disabled by configuration.",
                cited_ids=(),
                valid_cited_ids=(),
                invalid_cited_ids=(),
                available_evidence_ids=available,
                citation_coverage=None,
                diagnostics={"validation_enabled": False},
            )

        answer = answer_text.strip()
        parse = parse_citation_markers(answer)
        coverage, coverage_diagnostics = citation_coverage(answer, set(available))
        diagnostics: dict[str, object] = {
            "validation_enabled": True,
            "evidence_count": len(available),
            "raw_matches": list(parse.raw_matches),
            "invalid_tokens": list(parse.invalid_tokens),
            **coverage_diagnostics,
        }
        if not _has_visible_answer_content(answer):
            return _decision(False, "empty_answer", "Answer text contains no visible answer content.", parse.cited_ids, (), parse.cited_ids, available, coverage, diagnostics)
        if _is_citation_label_only_answer(answer, available):
            return _decision(False, "empty_answer", "Answer text only repeats citation labels and contains no substantive answer content.", parse.cited_ids, (), parse.cited_ids, available, coverage, diagnostics)
        if parse.invalid_tokens:
            return _decision(False, "invalid_citation_format", "Answer contains malformed citation markers.", parse.cited_ids, (), parse.cited_ids, available, coverage, diagnostics)

        structured_result = _parse_structured_citations(structured_citations)
        diagnostics["structured_citation_ids"] = list(structured_result.cited_ids)
        diagnostics["structured_invalid_tokens"] = list(structured_result.invalid_tokens)
        if structured_result.invalid_tokens:
            return _decision(False, "invalid_citation_format", "Structured citations contain malformed citation IDs.", parse.cited_ids, (), structured_result.invalid_tokens, available, coverage, diagnostics)

        cited_set = set(parse.cited_ids)
        structured_set = set(structured_result.cited_ids)
        available_set = set(available)
        valid = tuple(citation_id for citation_id in parse.cited_ids if citation_id in available_set)
        invalid = tuple(citation_id for citation_id in parse.cited_ids if citation_id not in available_set)

        if self.config.require_citations and not parse.cited_ids:
            return _decision(False, "missing_citations", "Answerable output must include at least one current-turn citation.", (), (), (), available, coverage, diagnostics)
        if cited_set != structured_set:
            mismatch = tuple(sorted(cited_set.symmetric_difference(structured_set), key=_citation_sort_key))
            diagnostics["citation_set_mismatch"] = list(mismatch)
            return _decision(False, "citation_set_mismatch", "Inline citations and structured citations differ.", parse.cited_ids, valid, mismatch, available, coverage, diagnostics)
        if invalid:
            reason = _unknown_reason(invalid, available)
            return _decision(False, reason, "Answer cites IDs outside the current evidence bundle.", parse.cited_ids, valid, invalid, available, coverage, diagnostics)
        if len(valid) < self.config.min_citations:
            return _decision(False, "missing_citations", "Answerable output does not meet the minimum citation count.", parse.cited_ids, valid, invalid, available, coverage, diagnostics)
        if self.config.min_coverage is not None and coverage is not None and coverage < self.config.min_coverage:
            return _decision(False, "insufficient_citation_coverage", "Citation coverage is below the configured threshold.", parse.cited_ids, valid, (), available, coverage, diagnostics)
        return _decision(True, "grounded", "Answer citations are grounded in the current evidence bundle.", parse.cited_ids, valid, (), available, coverage, diagnostics)


def citation_coverage(answer_text: str, available_ids: set[str]) -> tuple[float | None, dict[str, object]]:
    paragraphs = [part.strip() for part in answer_text.splitlines() if part.strip()]
    citable = []
    cited = []
    for paragraph in paragraphs:
        if _is_structural_paragraph(paragraph):
            continue
        if not _has_visible_answer_content(paragraph):
            continue
        citable.append(paragraph)
        parsed = parse_citation_markers(paragraph)
        if any(citation_id in available_ids for citation_id in parsed.cited_ids):
            cited.append(paragraph)
    if not citable:
        return None, {"citable_paragraph_count": 0, "cited_paragraph_count": 0}
    return len(cited) / len(citable), {"citable_paragraph_count": len(citable), "cited_paragraph_count": len(cited)}


def _parse_structured_citations(value: object) -> CitationParseResult:
    if not isinstance(value, list):
        return _parse_like_result((), ("<non-list>",))
    cited: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in value:
        citation_id = str(raw).strip()
        if not isinstance(raw, str) or citation_id != raw or not _is_valid_citation_id(citation_id):
            invalid.append(str(raw))
            continue
        if citation_id not in seen:
            cited.append(citation_id)
            seen.add(citation_id)
    return _parse_like_result(tuple(cited), tuple(dict.fromkeys(invalid)))


def _parse_like_result(cited_ids: tuple[str, ...], invalid_tokens: tuple[str, ...]):
    return CitationParseResult(cited_ids=cited_ids, invalid_tokens=invalid_tokens, raw_matches=cited_ids)


def _decision(
    valid: bool,
    reason_code: GroundingReasonCode,
    reason: str,
    cited_ids: tuple[str, ...],
    valid_cited_ids: tuple[str, ...],
    invalid_cited_ids: tuple[str, ...],
    available_evidence_ids: tuple[str, ...],
    citation_coverage_value: float | None,
    diagnostics: dict[str, object],
) -> GroundingValidationDecision:
    return GroundingValidationDecision(
        valid=valid,
        status="grounded" if valid else "refused",
        reason_code=reason_code,
        reason=reason,
        cited_ids=cited_ids,
        valid_cited_ids=valid_cited_ids,
        invalid_cited_ids=invalid_cited_ids,
        available_evidence_ids=available_evidence_ids,
        citation_coverage=citation_coverage_value,
        diagnostics=diagnostics,
    )


def _available_ids(bundle: EvidenceBundle) -> tuple[str, ...]:
    return tuple(f"C{index}" for index, _item in enumerate(bundle.items, start=1))


def _has_visible_answer_content(text: str) -> bool:
    without_citations = parse_citation_markers(text)
    visible = text
    for raw in without_citations.raw_matches:
        visible = visible.replace(raw, "")
    visible = visible.strip(" \t\r\n。！？.!?，,；;：:-_*`#>（）()[]")
    return bool(visible)


def _is_citation_label_only_answer(text: str, available_ids: tuple[str, ...]) -> bool:
    visible = text
    for raw in parse_citation_markers(text).raw_matches:
        visible = visible.replace(raw, " ")
    visible = visible.strip()
    if not visible:
        return False
    tokens = [token for token in re.split(r"[\s,，;；:：。.!！？、()\[\]（）]+", visible) if token]
    if not tokens:
        return False
    available = set(available_ids)
    return all(token in available for token in tokens)


def _is_structural_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if stripped.startswith("#"):
        return True
    normalized = stripped.rstrip("：:")
    return normalized in {"回答", "来源", "参考", "Sources", "References"}


def _unknown_reason(invalid_ids: tuple[str, ...], available: tuple[str, ...]) -> GroundingReasonCode:
    if not available:
        return "invalid_evidence_reference"
    max_available = len(available)
    numeric_ids = []
    for citation_id in invalid_ids:
        try:
            numeric_ids.append(int(citation_id[1:]))
        except ValueError:
            return "unknown_citation"
    if all(value > max_available for value in numeric_ids):
        return "citation_out_of_range" if max_available < 3 else "historical_citation_reference"
    return "unknown_citation"


def _is_valid_citation_id(value: str) -> bool:
    import re

    return re.fullmatch(r"C[1-9][0-9]*", value) is not None


def _citation_sort_key(value: str) -> tuple[int, str]:
    if len(value) > 1 and value[0] == "C" and value[1:].isdigit():
        return (int(value[1:]), value)
    return (10**9, value)
