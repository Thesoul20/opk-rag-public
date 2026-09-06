from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal

from opk_rag.answer.models import AnswerValidationResult, Citation
from opk_rag.search.models import EvidenceBundle

_CITATION_RE = re.compile(r"\[(C[1-9][0-9]*)\]")
_RELATIVE_PATH_RE = re.compile(r"(?<![\w/.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}(?![\w/.-])")
_URL_RE = re.compile(r"https?://[^\s\])}>'\"]+")
_FLAG_RE = re.compile(r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9-]*")
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b")
_KEBAB_RE = re.compile(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b")
_FILENAME_RE = re.compile(r"(?<![\w/.-])[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}(?![\w/.-])")
_VERSION_RE = re.compile(r"\b(?:v)?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9_.-]+)?\b")
_DATE_RE = re.compile(r"\b\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_KEY_COMBO_RE = re.compile(r"\b[A-Za-z](?:[+,+-][A-Za-z0-9;])+\b|[A-Za-z]\+;")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,80})`")
_TECH_TERM_RE = re.compile(r"\b(?=[A-Za-z0-9._-]*[A-Z0-9])[A-Za-z][A-Za-z0-9._-]{1,79}\b")
_CJK_SCOPE_TERMS = (
    "团队协作",
    "直接群发",
    "草稿箱",
    "移动端",
    "桌面端",
    "本地服务",
    "公众号配置",
    "模板配置",
    "草稿链接",
    "三线表",
    "视觉检查",
    "发布流程",
)
_CJK_STATUS_TERMS = ("必须", "默认", "已经", "已完成", "支持", "不支持", "暂缓", "后续", "计划", "可选", "第一阶段", "当前阶段")

SupportStatus = Literal["direct_support", "faithful_paraphrase", "partial_support", "topic_related_only", "contradicted", "not_found"]


@dataclass(frozen=True)
class ClaimGroundingDiagnostic:
    claim_id: str
    claim: str
    citation_ids: tuple[str, ...]
    checkable_values: tuple[str, ...]
    unsupported_values: tuple[str, ...]
    support_status: SupportStatus
    validator_status: Literal["supported", "unsupported"]
    reason: str


def build_citations(raw_citations: object, bundle: EvidenceBundle) -> tuple[Citation, ...]:
    if not isinstance(raw_citations, list):
        raise ValueError("citations must be a list.")
    evidence_by_id = {f"C{index}": item for index, item in enumerate(bundle.items, start=1)}
    citations: list[Citation] = []
    seen: set[str] = set()
    for raw in raw_citations:
        citation_id = str(raw).strip()
        if not re.fullmatch(r"C[1-9][0-9]*", citation_id):
            raise ValueError(f"invalid citation_id: {citation_id}")
        if citation_id in seen:
            continue
        item = evidence_by_id.get(citation_id)
        if item is None:
            raise ValueError(f"citation references unknown evidence: {citation_id}")
        citations.append(
            Citation(
                citation_id=citation_id,
                chunk_id=str(item.chunk_id),
                document_id=str(item.document_id),
                relative_path=item.relative_path,
                heading_path=item.heading_path,
                start_line=item.start_line,
                end_line=item.end_line,
                snippet=_evidence_text(item.content),
            )
        )
        seen.add(citation_id)
    return tuple(citations)


def validate_citation_markers(answer: str, citations: tuple[Citation, ...]) -> AnswerValidationResult:
    declared = {citation.citation_id for citation in citations}
    used = set(_CITATION_RE.findall(answer))
    if used - declared:
        return AnswerValidationResult(False, "invalid_citations", f"answer references undeclared citations: {sorted(used - declared)}")
    if declared - used:
        return AnswerValidationResult(False, "invalid_citations", f"declared citations are not used: {sorted(declared - used)}")
    if not used:
        return AnswerValidationResult(False, "invalid_citations", "answerable output must use at least one citation marker.")
    return AnswerValidationResult(True, None, "")


def unsupported_claims(answer: str, citations: tuple[Citation, ...]) -> tuple[str, ...]:
    claims: list[str] = []
    for diagnostic in claim_grounding_diagnostics(answer, citations):
        if diagnostic.validator_status != "unsupported":
            continue
        claims.extend(diagnostic.unsupported_values or (diagnostic.claim,))
    return tuple(claims)


def claim_grounding_diagnostics(answer: str, citations: tuple[Citation, ...]) -> tuple[ClaimGroundingDiagnostic, ...]:
    evidence_by_citation = {
        citation.citation_id: _normalize(" ".join((citation.relative_path, " ".join(citation.heading_path), citation.snippet)))
        for citation in citations
    }
    diagnostics: list[ClaimGroundingDiagnostic] = []
    for index, sentence in enumerate(split_answer_claims(answer), start=1):
        citation_ids = tuple(_CITATION_RE.findall(sentence))
        checkable_values = _extract_checkable_claims(sentence)
        if not citation_ids:
            if _is_missing_evidence_statement(sentence):
                diagnostics.append(_claim_diagnostic(index, sentence, citation_ids, checkable_values, (), "direct_support", "supported", "missing-evidence statement does not assert a source fact"))
            else:
                diagnostics.append(_claim_diagnostic(index, sentence, citation_ids, checkable_values, checkable_values or (sentence,), "not_found", "unsupported", "claim has no citation"))
            continue
        evidence_text = "\n".join(evidence_by_citation.get(citation_id, "") for citation_id in citation_ids)
        normalized_evidence = _normalize(evidence_text)
        unsupported_values = tuple(token for token in checkable_values if _is_unsupported_value(token, sentence, normalized_evidence))
        if unsupported_values:
            diagnostics.append(_claim_diagnostic(index, sentence, citation_ids, checkable_values, unsupported_values, "partial_support", "unsupported", "one or more checkable values are absent from cited evidence"))
            continue
        support_status: SupportStatus = "direct_support" if _normalize(_CITATION_RE.sub("", sentence)).strip() in normalized_evidence else "faithful_paraphrase"
        diagnostics.append(_claim_diagnostic(index, sentence, citation_ids, checkable_values, (), support_status, "supported", "all extracted checkable values are present in cited evidence"))
    return tuple(diagnostics)


def split_answer_claims(answer: str) -> tuple[str, ...]:
    claims: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)、])\s*", "", stripped)
        citation_suffix = "".join(f"[{citation_id}]" for citation_id in _CITATION_RE.findall(stripped)) if re.search(r"(?:\[C[1-9][0-9]*\])+\s*$", stripped) else ""
        # Keep evidence-backed missing-information disclosures separate from the cited factual clause.
        # Providers sometimes emit "...[C1] 当前知识库没有提供..." without sentence punctuation.
        stripped = re.sub(
            r"((?:\[C[1-9][0-9]*\])+)(?:\s+)(?=(?:当前)?知识库(?:没有|未)|当前知识库没有提供|当前知识库未提供)",
            r"\1\n",
            stripped,
        )
        # Do not split inside key combinations such as C+; . Chinese punctuation is a safe boundary
        # even when the provider omits a following space.
        for part in re.split(r"\n|(?<=[。！？!?])(?!\[C[1-9])\s*|(?<=\.)(?!\[C[1-9])\s+|；\s*|(?<!\+);\s+", stripped):
            normalized = part.strip()
            if normalized:
                if citation_suffix and not _CITATION_RE.findall(normalized):
                    normalized = f"{normalized}{citation_suffix}"
                claims.append(normalized)
    return tuple(claims)


def _extract_checkable_claims(sentence: str) -> tuple[str, ...]:
    text = _CITATION_RE.sub("", sentence)
    patterns = (
        _URL_RE,
        _RELATIVE_PATH_RE,
        _BACKTICK_RE,
        _KEY_COMBO_RE,
        _FLAG_RE,
        _ENV_RE,
        _SNAKE_RE,
        _KEBAB_RE,
        _FILENAME_RE,
        _DATE_RE,
        _TIME_RE,
        _PERCENT_RE,
        _VERSION_RE,
        _NUMBER_RE,
    )
    values: list[str] = []
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if any(not (span[1] <= existing[0] or span[0] >= existing[1]) for existing in spans):
                continue
            values.append((match.group(1) if pattern is _BACKTICK_RE else match.group(0)).strip(".,:，。：）)]}"))
            spans.append(span)
    for match in _TECH_TERM_RE.finditer(text):
        span = match.span()
        value = match.group(0).strip(".,;:，。；：）)]}")
        if re.fullmatch(r"C[1-9][0-9]*", value):
            continue
        if any(not (span[1] <= existing[0] or span[0] >= existing[1]) for existing in spans):
            continue
        values.append(value)
        spans.append(span)
    for term in _CJK_SCOPE_TERMS:
        if term in text:
            values.append(term)
    return tuple(dict.fromkeys(value for value in values if value))


def _claim_diagnostic(
    index: int,
    claim: str,
    citation_ids: tuple[str, ...],
    checkable_values: tuple[str, ...],
    unsupported_values: tuple[str, ...],
    support_status: SupportStatus,
    validator_status: Literal["supported", "unsupported"],
    reason: str,
) -> ClaimGroundingDiagnostic:
    return ClaimGroundingDiagnostic(
        claim_id=f"claim-{index}",
        claim=claim,
        citation_ids=citation_ids,
        checkable_values=checkable_values,
        unsupported_values=unsupported_values,
        support_status=support_status,
        validator_status=validator_status,
        reason=reason,
    )


def _is_missing_evidence_statement(sentence: str) -> bool:
    text = _normalize(_CITATION_RE.sub("", sentence))
    missing_markers = ("未提供", "没有提供", "未说明", "没有说明", "未记录", "没有记录", "无法确定", "没有足够证据", "知识库没有", "证据没有")
    return any(marker in text for marker in missing_markers)


def _is_unsupported_value(token: str, claim: str, normalized_evidence: str) -> bool:
    normalized_token = _normalize(token)
    normalized_claim = _normalize(_CITATION_RE.sub("", claim))
    if normalized_token not in normalized_evidence and not _supported_by_equivalent_cjk_phrase(normalized_token, normalized_claim, normalized_evidence):
        return True
    if _negation_conflicts(normalized_token, normalized_claim, normalized_evidence):
        return True
    if _phase_scope_conflicts(normalized_token, normalized_claim, normalized_evidence):
        return True
    if _status_scope_conflicts(normalized_token, normalized_claim, normalized_evidence):
        return True
    return False


def _supported_by_equivalent_cjk_phrase(token: str, claim: str, evidence: str) -> bool:
    if token == "发布流程":
        has_static_site_scope = "静态站点" not in claim or "静态站点" in evidence
        return has_static_site_scope and "发布" in evidence and "流程" in evidence
    return False


def _claim_negates_token(token: str, claim: str) -> bool:
    if token not in claim:
        return False
    negative_markers = (
        "不支持", "并不支持", "尚未支持", "未支持", "暂不支持", "没有支持", "还没有支持",
        "未实现", "尚未实现", "还没有实现", "没有实现",
        "未完成", "尚未完成", "没有完成", "不包含", "不包括",
    )
    if any(marker in claim for marker in negative_markers):
        return True
    # Token-scoped negation covers ordinary model paraphrases such as
    # “不会直接群发” while remaining bounded to the extracted claim token.
    compact = claim.replace(" ", "")
    token_compact = token.replace(" ", "")
    return any(f"{prefix}{token_compact}" in compact for prefix in ("不", "不会", "未", "尚未", "没有", "并非", "不是"))


def _negation_conflicts(token: str, claim: str, evidence: str) -> bool:
    if _claim_negates_token(token, claim):
        return False
    claim_asserts_support = any(marker in claim for marker in ("支持", "可支持", "包含", "包括"))
    evidence_denies_support = any(marker in evidence for marker in (f"不支持 {token}", f"不支持{token}", f"不直接支持 {token}", f"不直接支持{token}"))
    return claim_asserts_support and evidence_denies_support


def _phase_scope_conflicts(token: str, claim: str, evidence: str) -> bool:
    if _claim_negates_token(token, claim):
        return False
    claim_first_phase = any(marker in claim for marker in ("第一阶段", "首阶段", "当前阶段", "第一版")) and any(marker in claim for marker in ("交付", "支持", "范围", "必须"))
    evidence_later_phase = any(marker in evidence for marker in (f"后续考虑 {token}", f"后续考虑{token}", f"后支持 {token}", f"后支持{token}", f"短期不做 {token}", f"短期不做{token}"))
    return claim_first_phase and evidence_later_phase


def _status_scope_conflicts(token: str, claim: str, evidence: str) -> bool:
    if _is_missing_evidence_statement(claim) or _claim_negates_token(token, claim):
        return False
    if token == "直接群发" and _claim_negates_token(token, claim):
        return False
    if token in {"word", "word 文件", "团队协作", "直接群发", "移动端"}:
        if any(marker in claim for marker in ("必须", "已经", "支持", "第一版", "第一阶段")) and any(marker in evidence for marker in ("暂缓", "后续", "不直接", "先支持 markdown", "第一阶段只做草稿箱")):
            return True
    return False


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("\\n", " ").replace("\\t", " ")
    normalized = normalized.replace("```", "`")
    return " ".join(normalized.split())


def _evidence_text(content: str) -> str:
    return " ".join(content.split())
