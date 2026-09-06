from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from opk_rag.answerability import AnswerabilityDecision
from opk_rag.answer.negative_semantics import classify_answer_semantics
from opk_rag.search.models import EvidenceBundle

SCOPE_EXECUTION_CONTRACT_VERSION = "scope-execution-contract-v1"

AnswerMode = Literal["full", "partial", "abstain"]
ScopeValidationReason = Literal[
    "valid",
    "invalid_contract",
    "mode_mismatch",
    "unknown_scope_item",
    "forbidden_scope_violation",
    "missing_citation",
    "unknown_citation",
    "unit_count_exceeded",
    "empty_unit",
    "partial_notice_missing",
]


@dataclass(frozen=True)
class ScopeItem:
    item_id: str
    text: str


@dataclass(frozen=True)
class ScopeExecutionContract:
    contract_version: str
    answer_mode: AnswerMode
    allowed_scope_items: tuple[ScopeItem, ...]
    forbidden_scope_items: tuple[ScopeItem, ...]
    premise_correction: str | None
    allowed_citation_ids: tuple[str, ...]
    max_answer_units: int
    unsupported_scope_notice_required: bool
    answer_semantic_class: str = "supported_affirmative"
    negative_answer_allowed: bool = False
    negative_support_citation_ids: tuple[str, ...] = ()
    negative_support_type: str = "none"

    def to_provider_payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "answer_mode": self.answer_mode,
            "allowed_scope_items": [{"scope_item_id": item.item_id, "text": item.text} for item in self.allowed_scope_items],
            "forbidden_scope_items": [{"scope_item_id": item.item_id, "text": item.text} for item in self.forbidden_scope_items],
            "premise_correction": self.premise_correction,
            "allowed_citation_ids": list(self.allowed_citation_ids),
            "max_answer_units": self.max_answer_units,
            "unsupported_scope_notice_required": self.unsupported_scope_notice_required,
            "answer_semantic_class": self.answer_semantic_class,
            "negative_answer_allowed": self.negative_answer_allowed,
            "negative_support_citation_ids": list(self.negative_support_citation_ids),
            "negative_support_type": self.negative_support_type,
        }


@dataclass(frozen=True)
class AnswerUnit:
    unit_id: str
    scope_item_id: str
    text: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class ScopeExecutionValidation:
    valid: bool
    reason_code: ScopeValidationReason
    reason: str
    answer_units: tuple[AnswerUnit, ...]
    answer_text: str
    citations: tuple[str, ...]
    diagnostics: dict[str, object]


def build_scope_execution_contract(decision: AnswerabilityDecision, bundle: EvidenceBundle, *, question: str | None = None) -> ScopeExecutionContract:
    citation_ids_by_chunk = {str(item.chunk_id): f"C{index}" for index, item in enumerate(bundle.items, start=1)}
    selected = tuple(citation_ids_by_chunk[chunk_id] for chunk_id in decision.evidence_chunk_ids if chunk_id in citation_ids_by_chunk)
    allowed_citations = selected or tuple(citation_ids_by_chunk.values())
    supported = _scope_values(decision.diagnostics.get("semantic_supported_scope")) or _scope_values(decision.diagnostics.get("covered_query_terms"))
    if not supported and decision.status != "unanswerable":
        supported = [f"Use evidence {citation_id} for supported facts." for citation_id in allowed_citations]
    forbidden = _unsupported_scope(decision)
    mode: AnswerMode
    if not decision.answerable or decision.status == "unanswerable":
        mode = "abstain"
    elif decision.status == "partially_answerable" or decision.reason_code == "false_premise":
        mode = "partial"
    else:
        mode = "full"
    allowed_items = tuple(ScopeItem(f"S{index}", value) for index, value in enumerate(supported, start=1))
    forbidden_items = tuple(ScopeItem(f"U{index}", value) for index, value in enumerate(forbidden, start=1))
    semantic = classify_answer_semantics(question=question or bundle.query, bundle=bundle, answerability=decision)
    return ScopeExecutionContract(
        contract_version=SCOPE_EXECUTION_CONTRACT_VERSION,
        answer_mode=mode,
        allowed_scope_items=allowed_items,
        forbidden_scope_items=forbidden_items,
        premise_correction=_premise_correction(decision),
        allowed_citation_ids=allowed_citations,
        max_answer_units=0 if mode == "abstain" else max(1, len(allowed_items)),
        unsupported_scope_notice_required=mode == "partial",
        answer_semantic_class=semantic.semantic_class,
        negative_answer_allowed=semantic.semantic_class == "supported_negative" and semantic.safe_to_finish,
        negative_support_citation_ids=semantic.supporting_citation_ids,
        negative_support_type=semantic.negative_support_type,
    )


def validate_scope_execution_output(parsed: dict, contract: ScopeExecutionContract) -> ScopeExecutionValidation:
    action = parsed.get("action", parsed.get("decision"))
    expected_action = {"full": "answer", "partial": "partial_answer", "abstain": "abstain"}[contract.answer_mode]
    if action != expected_action:
        return _invalid("mode_mismatch", "Provider action does not match the server answer mode.", contract, parsed, ())
    units_value = parsed.get("answer_units")
    if contract.answer_mode == "abstain":
        if units_value not in (None, []):
            return _invalid("invalid_contract", "Abstain mode must not contain answer units.", contract, parsed, ())
        return ScopeExecutionValidation(True, "valid", "Scope execution output is valid.", (), "", (), _diagnostics(contract, parsed, ()))
    if not isinstance(units_value, list):
        return _invalid("invalid_contract", "answer_units must be a list.", contract, parsed, ())
    if len(units_value) > contract.max_answer_units:
        return _invalid("unit_count_exceeded", "answer_units exceeds the server maximum.", contract, parsed, ())
    if contract.answer_mode == "partial" and not units_value:
        return _invalid("invalid_contract", "Partial mode requires at least one answer unit.", contract, parsed, ())
    notice = parsed.get("unsupported_scope_notice")
    if contract.answer_mode == "partial" and contract.unsupported_scope_notice_required and not _non_empty_string(notice):
        return _invalid("partial_notice_missing", "Partial mode requires unsupported_scope_notice.", contract, parsed, ())
    if contract.answer_mode == "full" and _non_empty_string(notice):
        return _invalid("invalid_contract", "Full mode must not include unsupported_scope_notice.", contract, parsed, ())

    allowed_scope = {item.item_id for item in contract.allowed_scope_items}
    forbidden_scope = {item.item_id for item in contract.forbidden_scope_items}
    allowed_citations = set(contract.allowed_citation_ids)
    units: list[AnswerUnit] = []
    all_citations: list[str] = []
    for index, raw_unit in enumerate(units_value, start=1):
        if not isinstance(raw_unit, dict):
            return _invalid("invalid_contract", "Each answer unit must be an object.", contract, parsed, tuple(units))
        scope_id = str(raw_unit.get("scope_item_id") or "").strip()
        if scope_id in forbidden_scope or scope_id.startswith("U"):
            return _invalid("forbidden_scope_violation", "Answer unit is bound to forbidden scope.", contract, parsed, tuple(units))
        if scope_id not in allowed_scope:
            return _invalid("unknown_scope_item", "Answer unit references unknown scope.", contract, parsed, tuple(units))
        text = str(raw_unit.get("text") or "").strip()
        if not text:
            return _invalid("empty_unit", "Answer unit text is empty.", contract, parsed, tuple(units))
        citations_value = raw_unit.get("citations")
        if not isinstance(citations_value, list) or not citations_value:
            return _invalid("missing_citation", "Answer unit must contain at least one citation.", contract, parsed, tuple(units))
        citations: list[str] = []
        for citation in citations_value:
            citation_id = str(citation).strip()
            if citation_id not in allowed_citations:
                return _invalid("unknown_citation", "Answer unit cites evidence outside the server contract.", contract, parsed, tuple(units))
            if citation_id not in citations:
                citations.append(citation_id)
            if citation_id not in all_citations:
                all_citations.append(citation_id)
        units.append(AnswerUnit(unit_id=str(raw_unit.get("unit_id") or f"A{index}"), scope_item_id=scope_id, text=text, citations=tuple(citations)))
    answer = merge_answer_units(tuple(units), unsupported_scope_notice=str(notice).strip() if _non_empty_string(notice) else None)
    return ScopeExecutionValidation(True, "valid", "Scope execution output is valid.", tuple(units), answer, tuple(all_citations), _diagnostics(contract, parsed, tuple(units)))


def merge_answer_units(units: tuple[AnswerUnit, ...], *, unsupported_scope_notice: str | None = None) -> str:
    lines = [unit.text for unit in units]
    if unsupported_scope_notice:
        lines.append(unsupported_scope_notice)
    return "\n".join(lines).strip()


def _unsupported_scope(decision: AnswerabilityDecision) -> list[str]:
    values: list[str] = []
    semantic = decision.diagnostics.get("semantic_unsupported_scope")
    if isinstance(semantic, list):
        values.extend(str(value).strip() for value in semantic if str(value).strip())
    for key in ("missing_exact_requirements", "conflicting_values"):
        values.extend(_scope_values(decision.diagnostics.get(key)))
    query_terms = _scope_values(decision.diagnostics.get("query_terms"))
    covered = set(_scope_values(decision.diagnostics.get("covered_query_terms")))
    values.extend(f"未由证据覆盖的问题要素：{value}" for value in query_terms if value not in covered)
    if decision.reason_code == "unsupported_inference":
        values.append("问题要求的推断或保证没有被当前证据支持")
    return list(dict.fromkeys(values))


def _scope_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _premise_correction(decision: AnswerabilityDecision) -> str | None:
    if decision.reason_code != "false_premise":
        return None
    return "The question premise conflicts with the retrieved evidence. Correct the premise first and cite only the corrected fact."


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _invalid(reason_code: ScopeValidationReason, reason: str, contract: ScopeExecutionContract, parsed: dict, units: tuple[AnswerUnit, ...]) -> ScopeExecutionValidation:
    return ScopeExecutionValidation(False, reason_code, reason, units, "", (), _diagnostics(contract, parsed, units))


def _diagnostics(contract: ScopeExecutionContract, parsed: dict, units: tuple[AnswerUnit, ...]) -> dict[str, object]:
    return {
        "contract_version": contract.contract_version,
        "answer_mode": contract.answer_mode,
        "allowed_scope_item_count": len(contract.allowed_scope_items),
        "forbidden_scope_item_count": len(contract.forbidden_scope_items),
        "allowed_citation_count": len(contract.allowed_citation_ids),
        "max_answer_units": contract.max_answer_units,
        "answer_unit_count": len(parsed.get("answer_units") or []) if isinstance(parsed.get("answer_units"), list) else 0,
        "valid_unit_count": len(units),
        "contains_scope_text": False,
        "contains_answer_unit_text": False,
    }
