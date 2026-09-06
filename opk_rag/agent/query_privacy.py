from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

QUERY_PRIVACY_DIAGNOSTICS_VERSION = "opk-rag.agent-query-privacy-diagnostics.v1"

GOLD_LEAKAGE_TERMS = (
    "answerability_label",
    "expected_action",
    "required_claims",
    "forbidden_claims",
    "gold_evidence",
    "gold evidence",
    "gold citation",
    "owner_review",
    "owner review",
    "benchmark_score",
    "benchmark score",
    "baseline_pass_fail",
    "baseline pass/fail",
)

SECRET_TERMS = ("api_key", "authorization", "database_url", "postgres://", "sk-")

FORBIDDEN_QUERY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sql", r"\b(select|insert|update|delete|drop|alter|truncate|create)\b.+\b(from|table|database|schema|where)\b"),
    ("file_command", r"\b(cat|less|more|tail|head|sed|awk|grep|rg)\s+[/~.]"),
    ("shell_command", r"\b(rm|curl|wget|ssh|scp|bash|zsh|sh|python|node)\b\s+[-/A-Za-z0-9_.$~]"),
    ("url_fetch", r"\b(fetch|download|crawl|open)\b.+https?://"),
    ("url_injection", r"https?://\S+"),
    ("prompt_injection", r"\b(ignore|override|bypass|disregard)\b.+\b(system|instruction|policy|developer)\b"),
    ("prompt_injection", r"(系统提示|开发者指令|忽略.*指令|绕过.*治理|不要遵守)"),
    ("gold_leakage", r"\b(answerability_label|expected_action|required_claims|forbidden_claims|gold evidence|benchmark score)\b"),
    ("direct_answer", r"\b(final answer|answer the user)\b|直接回答|最终答案|输出答案"),
    ("budget_override", r"\b(max_steps|max_tool_calls|max_search_rounds|modify budget|修改预算|增加预算)\b"),
    ("system_prompt", r"\b(system prompt|developer message|系统消息|系统提示词)\b"),
)


@dataclass(frozen=True)
class QueryPrivacyDiagnostics:
    query_digest: str
    query_length: int
    query_present: bool
    forbidden: bool
    reason_codes: tuple[str, ...]
    schema_version: str = QUERY_PRIVACY_DIAGNOSTICS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_digest": self.query_digest,
            "query_length": self.query_length,
            "query_present": self.query_present,
            "forbidden": self.forbidden,
            "reason_codes": list(self.reason_codes),
        }


def query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def diagnose_query_privacy(query: str | None) -> QueryPrivacyDiagnostics:
    normalized = normalize_query(query or "")
    reasons = forbidden_query_reason_codes(normalized)
    return QueryPrivacyDiagnostics(
        query_digest=query_digest(normalized),
        query_length=len(normalized),
        query_present=bool(normalized),
        forbidden=bool(reasons),
        reason_codes=tuple(reasons),
    )


def forbidden_query_reason_codes(query: str) -> list[str]:
    lowered = query.lower()
    reasons: list[str] = []
    for term in GOLD_LEAKAGE_TERMS:
        if term in lowered:
            reasons.append("query_plan_forbidden_content")
            break
    for term in SECRET_TERMS:
        if term in lowered:
            reasons.append("query_plan_forbidden_content")
            break
    for reason, pattern in FORBIDDEN_QUERY_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            if reason == "gold_leakage":
                reasons.append("query_plan_forbidden_content")
            elif reason == "direct_answer":
                reasons.append("query_plan_forbidden_content")
            elif reason == "budget_override":
                reasons.append("query_plan_forbidden_content")
            else:
                reasons.append("query_plan_forbidden_content")
    return sorted(set(reasons))


def redact_query_for_artifact(query: str) -> dict[str, Any]:
    normalized = normalize_query(query)
    return {
        "query_digest": query_digest(normalized),
        "query_length": len(normalized),
    }
