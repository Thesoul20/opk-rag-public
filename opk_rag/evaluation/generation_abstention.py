from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any

from opk_rag.evaluation.generation_stability import normalize_output

ABSTENTION_CATEGORIES = (
    "explicit_model_abstention",
    "model_claimed_insufficient_evidence",
    "empty_output",
    "truncated_output",
    "malformed_structured_output",
    "missing_required_fields",
    "parse_failure_interpreted_as_abstention",
    "unsupported_scope_only",
    "provider_safety_refusal",
    "contract_defaulted_to_abstention",
    "unknown_abstention",
)


@dataclass(frozen=True)
class AbstentionAttribution:
    category: str
    explicit_abstention_marker: bool
    parser_inferred_abstention_marker: bool
    parse_failure_reason: str | None
    normalized_raw_output: str
    citation_present: bool
    output_token_count: int


def classify_abstention_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_output")
    parsed = row.get("parsed_output") if isinstance(row.get("parsed_output"), dict) else None
    model_decision = parsed or (row.get("answer_payload", {}).get("model_decision") if isinstance(row.get("answer_payload"), dict) else None)
    if not isinstance(model_decision, dict):
        model_decision = {}
    attribution = classify_abstention(
        raw_output=raw,
        parsed_output=parsed,
        model_decision=model_decision,
        generation_status=str(row.get("generation_status") or ""),
        generation_reason=str(row.get("generation_reason") or ""),
        final_action=str(row.get("final_action") or ""),
        finish_reason=row.get("finish_reason"),
        output_truncated=bool(row.get("output_truncated")),
        empty_output=bool(row.get("empty_output")),
    )
    return {
        "sample_id": row.get("sample_id") or row.get("id"),
        "run_index": row.get("run_index"),
        "raw_output": raw,
        "normalized_raw_output": attribution.normalized_raw_output,
        "parsed_decision": model_decision.get("decision"),
        "generation_status": row.get("generation_status"),
        "generation_reason": row.get("generation_reason"),
        "abstention_category": attribution.category,
        "explicit_abstention_marker": attribution.explicit_abstention_marker,
        "parser_inferred_abstention_marker": attribution.parser_inferred_abstention_marker,
        "finish_reason": row.get("finish_reason"),
        "output_token_count": attribution.output_token_count,
        "output_truncated": bool(row.get("output_truncated")),
        "parse_failure_reason": attribution.parse_failure_reason,
        "citation_presence": attribution.citation_present,
        "final_action": row.get("final_action"),
    }


def classify_abstention(
    *,
    raw_output: Any,
    parsed_output: dict[str, Any] | None,
    model_decision: dict[str, Any],
    generation_status: str,
    generation_reason: str,
    final_action: str,
    finish_reason: Any,
    output_truncated: bool,
    empty_output: bool,
) -> AbstentionAttribution:
    raw_text = "" if raw_output is None else str(raw_output)
    normalized = normalize_output(raw_text)
    parse_failure = parse_failure_reason(raw_text, parsed_output)
    citation_present = bool(re.search(r"\[C[1-9][0-9]*\]|\"C[1-9][0-9]*\"", raw_text))
    explicit = explicit_abstention_marker(raw_text, model_decision)
    parser_inferred = final_action == "abstain" and not explicit and (parse_failure is not None or generation_status in {"parse_failure", "contract_failure"})
    category = "unknown_abstention"
    reason = str(model_decision.get("reason") or generation_reason or "")
    unsupported_scope = model_decision.get("unsupported_scope")

    if empty_output or not raw_text.strip():
        category = "empty_output"
    elif output_truncated or finish_reason == "length":
        category = "truncated_output"
    elif _provider_refusal(raw_text):
        category = "provider_safety_refusal"
    elif explicit and normalized.strip().casefold() == "abstain":
        category = "explicit_model_abstention"
    elif parse_failure == "malformed_json":
        category = "malformed_structured_output"
    elif parse_failure == "missing_required_fields":
        category = "missing_required_fields"
    elif parser_inferred:
        category = "parse_failure_interpreted_as_abstention"
    elif unsupported_scope and model_decision.get("decision") == "abstain":
        category = "unsupported_scope_only"
    elif explicit and reason in {"insufficient_evidence", "no_evidence"}:
        category = "model_claimed_insufficient_evidence"
    elif explicit:
        category = "explicit_model_abstention"
    elif final_action == "abstain" and generation_status == "not_started":
        category = "contract_defaulted_to_abstention"

    return AbstentionAttribution(
        category=category,
        explicit_abstention_marker=explicit,
        parser_inferred_abstention_marker=parser_inferred,
        parse_failure_reason=parse_failure,
        normalized_raw_output=normalized,
        citation_present=citation_present,
        output_token_count=len(raw_text.split()),
    )


def parse_failure_reason(raw_text: str, parsed_output: dict[str, Any] | None) -> str | None:
    if not raw_text.strip():
        return None
    if parsed_output is None:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return "malformed_json"
    else:
        parsed = parsed_output
    if not isinstance(parsed, dict):
        return "malformed_json"
    required = {"decision", "answer", "citations", "reason"}
    if not required.issubset(parsed):
        return "missing_required_fields"
    if parsed.get("decision") not in {"answer", "abstain"}:
        return "invalid_decision"
    return None


def explicit_abstention_marker(raw_text: str, model_decision: dict[str, Any]) -> bool:
    if model_decision.get("decision") == "abstain":
        return True
    lowered = raw_text.casefold()
    return any(marker in lowered for marker in ("abstain", "insufficient evidence", "no evidence", "证据不足", "无法回答", "不能回答", "没有足够"))


def summarize_abstention_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [classify_abstention_row(row) for row in rows if row.get("final_action") == "abstain" or row.get("generation_status") == "abstention"]
    raw_values = [str(row.get("raw_output") or "") for row in classified]
    normalized_values = [row["normalized_raw_output"] for row in classified]
    top_raw = Counter(raw_values).most_common(1)
    top_reason = Counter(str(row.get("generation_reason") or "") for row in rows).most_common(1)
    lengths = [row["output_token_count"] for row in classified]
    return {
        "classified_count": len(classified),
        "category_counts": dict(Counter(row["abstention_category"] for row in classified)),
        "unique_raw_output_count": len(set(raw_values)),
        "normalized_unique_output_count": len(set(normalized_values)),
        "top_repeated_output": {"output": top_raw[0][0], "count": top_raw[0][1]} if top_raw else None,
        "top_repeated_abstention_reason": {"reason": top_reason[0][0], "count": top_reason[0][1]} if top_reason else None,
        "output_length_distribution": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "avg": (sum(lengths) / len(lengths)) if lengths else None,
        },
        "citation_presence_rate": (sum(1 for row in classified if row["citation_presence"]) / len(classified)) if classified else None,
        "rows": classified,
    }


def _provider_refusal(raw_text: str) -> bool:
    lowered = raw_text.casefold()
    return any(marker in lowered for marker in ("i cannot comply", "i can't comply", "safety", "policy", "安全策略", "无法协助"))
