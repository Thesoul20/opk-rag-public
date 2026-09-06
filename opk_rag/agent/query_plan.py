from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.query_privacy import diagnose_query_privacy, normalize_query, query_digest

QUERY_PLAN_CONTRACT_VERSION = "opk-rag.agent-query-plan.v1"

ALLOWED_STRATEGIES = frozenset(
    {
        "focused_reformulation",
        "entity_disambiguation",
        "scope_narrowing",
        "terminology_expansion",
        "document_locator_rewrite",
        "safe_no_reformulation",
    }
)

ALLOWED_REASON_CODES = frozenset(
    {
        "initial_evidence_insufficient",
        "initial_query_too_broad",
        "missing_scope_localization",
        "ambiguous_entity",
        "terminology_mismatch",
        "document_found_chunk_missing",
        "reformulation_not_expected_to_help",
        "budget_exhausted",
    }
)

ALLOWED_QUERY_ROLES = frozenset({"primary_rewrite", "secondary_rewrite", "entity_locator", "scope_locator", "terminology_variant"})
ALLOWED_SOURCES = frozenset({"model_reformulation", "deterministic_reformulation", "no_reformulation"})
ALLOWED_EXPECTED_GAINS = frozenset(
    {
        "scope_localization",
        "document_discovery",
        "terminology_alignment",
        "entity_disambiguation",
        "metadata_recovery",
        "no_expected_gain",
    }
)
ALLOWED_FIELDS = frozenset({"contract_version", "strategy", "reason_code", "queries", "source", "expected_information_gain", "stop_after_round"})


@dataclass(frozen=True)
class QueryPlanQuery:
    query: str
    query_role: str

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "query_role": self.query_role}


@dataclass(frozen=True)
class QueryPlan:
    strategy: str
    reason_code: str
    queries: tuple[QueryPlanQuery, ...]
    source: str
    expected_information_gain: str
    stop_after_round: int = 2
    contract_version: str = QUERY_PLAN_CONTRACT_VERSION

    def to_dict(self, *, redact_queries: bool = False) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        for item in self.queries:
            payload = item.to_dict()
            if redact_queries:
                payload = {
                    "query_digest": query_digest(item.query),
                    "query_length": len(item.query),
                    "query_role": item.query_role,
                }
            queries.append(payload)
        return {
            "contract_version": self.contract_version,
            "strategy": self.strategy,
            "reason_code": self.reason_code,
            "queries": queries,
            "source": self.source,
            "expected_information_gain": self.expected_information_gain,
            "stop_after_round": self.stop_after_round,
        }


@dataclass(frozen=True)
class QueryPlanValidationConfig:
    max_queries: int = 2
    max_query_length: int = 256
    max_search_rounds: int = 2


def parse_query_plan_json(raw_output: str) -> dict[str, Any]:
    stripped = raw_output.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AgentRuntimeError("query_plan_malformed_json", "Query plan provider output is not valid JSON.", origin="query_reformulation", detail={"position": exc.pos}) from exc
    if not isinstance(parsed, dict):
        raise AgentRuntimeError("query_plan_contract_failure", "Query plan must be a single JSON object.", origin="query_reformulation")
    if set(parsed) == {"query_plan"} and isinstance(parsed["query_plan"], dict):
        parsed = parsed["query_plan"]
    parsed = _adapt_provider_query_plan(parsed)
    return parsed


def validate_query_plan(
    payload: dict[str, Any],
    *,
    original_query: str,
    config: QueryPlanValidationConfig | None = None,
) -> QueryPlan:
    config = config or QueryPlanValidationConfig()
    unknown = set(payload) - ALLOWED_FIELDS
    if unknown:
        raise AgentRuntimeError("query_plan_contract_failure", "Query plan contains unknown fields.", origin="query_reformulation", detail={"unknown_fields": sorted(unknown)})
    if payload.get("contract_version") != QUERY_PLAN_CONTRACT_VERSION:
        raise AgentRuntimeError("query_plan_contract_failure", "Unknown query plan contract version.", origin="query_reformulation")
    strategy = _require_enum(payload, "strategy", ALLOWED_STRATEGIES)
    reason_code = _require_enum(payload, "reason_code", ALLOWED_REASON_CODES)
    source = _require_enum(payload, "source", ALLOWED_SOURCES)
    expected_gain = _require_enum(payload, "expected_information_gain", ALLOWED_EXPECTED_GAINS)
    stop_after_round = payload.get("stop_after_round", 2)
    if not isinstance(stop_after_round, int) or stop_after_round < 1 or stop_after_round > config.max_search_rounds:
        raise AgentRuntimeError("retrieval_round_budget_exhausted", "Query plan stop_after_round exceeds search round budget.", origin="query_reformulation")
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        raise AgentRuntimeError("query_plan_contract_failure", "Query plan queries must be a list.", origin="query_reformulation")
    if len(raw_queries) > config.max_queries:
        raise AgentRuntimeError("query_plan_contract_failure", "Query plan contains too many queries.", origin="query_reformulation")
    if strategy == "safe_no_reformulation":
        if raw_queries:
            raise AgentRuntimeError("query_plan_contract_failure", "safe_no_reformulation must not include queries.", origin="query_reformulation")
        return QueryPlan(strategy=strategy, reason_code=reason_code, queries=(), source=source, expected_information_gain=expected_gain, stop_after_round=stop_after_round)

    original_normalized = normalize_query(original_query)
    seen: set[str] = set()
    queries: list[QueryPlanQuery] = []
    for index, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            raise AgentRuntimeError("query_plan_contract_failure", "Each query entry must be an object.", origin="query_reformulation")
        if set(item) != {"query", "query_role"}:
            raise AgentRuntimeError("query_plan_contract_failure", "Each query entry must contain exactly query and query_role.", origin="query_reformulation")
        role = item.get("query_role")
        if role not in ALLOWED_QUERY_ROLES:
            raise AgentRuntimeError("query_plan_contract_failure", "Unknown query role.", origin="query_reformulation", detail={"query_index": index})
        query = item.get("query")
        if not isinstance(query, str):
            raise AgentRuntimeError("query_plan_empty_query", "Query must be a string.", origin="query_reformulation")
        normalized = normalize_query(query)
        if not normalized:
            raise AgentRuntimeError("query_plan_empty_query", "Query must be non-empty after trimming whitespace.", origin="query_reformulation")
        if len(normalized) > config.max_query_length:
            raise AgentRuntimeError("query_plan_contract_failure", "Query exceeds maximum length.", origin="query_reformulation", detail={"max_query_length": config.max_query_length})
        normalized_key = normalized.casefold()
        if normalized_key in seen:
            raise AgentRuntimeError("query_plan_duplicate_query", "Query plan contains duplicate queries.", origin="query_reformulation")
        if normalized_key == original_normalized.casefold():
            raise AgentRuntimeError("query_plan_original_query_duplicate", "Query plan repeats the original query.", origin="query_reformulation")
        privacy = diagnose_query_privacy(normalized)
        if privacy.forbidden:
            raise AgentRuntimeError("query_plan_forbidden_content", "Query plan contains forbidden query content.", origin="query_reformulation", detail=privacy.to_dict())
        seen.add(normalized_key)
        queries.append(QueryPlanQuery(query=normalized, query_role=role))
    return QueryPlan(strategy=strategy, reason_code=reason_code, queries=tuple(queries), source=source, expected_information_gain=expected_gain, stop_after_round=stop_after_round)


def query_plan_contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": QUERY_PLAN_CONTRACT_VERSION,
        "allowed_strategies": sorted(ALLOWED_STRATEGIES),
        "allowed_reason_codes": sorted(ALLOWED_REASON_CODES),
        "allowed_query_roles": sorted(ALLOWED_QUERY_ROLES),
        "allowed_sources": sorted(ALLOWED_SOURCES),
        "allowed_expected_information_gains": sorted(ALLOWED_EXPECTED_GAINS),
        "limits": {
            "max_queries": 2,
            "max_query_length": 256,
            "max_search_rounds": 2,
        },
    }


def _require_enum(payload: dict[str, Any], field: str, allowed: frozenset[str]) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or value not in allowed:
        raise AgentRuntimeError("query_plan_contract_failure", f"Invalid query plan field: {field}.", origin="query_reformulation")
    return value


def _strip_markdown_fence(value: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise AgentRuntimeError("query_plan_malformed_json", "Query plan output contains an invalid Markdown fence.", origin="query_reformulation")
    return match.group(1).strip()


def _adapt_provider_query_plan(payload: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(payload)
    if "query_plan" in adapted and isinstance(adapted["query_plan"], dict):
        adapted = dict(adapted["query_plan"])
    elif "search_rounds" in adapted and isinstance(adapted["search_rounds"], list):
        for item in adapted["search_rounds"]:
            if isinstance(item, dict) and isinstance(item.get("query_plan"), dict):
                adapted = dict(item["query_plan"])
                break

    source_aliases = {
        "model": "model_reformulation",
        "model_generated": "model_reformulation",
        "reformulation": "model_reformulation",
        "user_question": "model_reformulation",
        "llm": "model_reformulation",
    }
    gain_aliases = {
        "high": "document_discovery",
        "medium": "scope_localization",
        "low": "terminology_alignment",
        "expected_gain": "document_discovery",
    }
    role_aliases = {
        "primary": "primary_rewrite",
        "secondary": "secondary_rewrite",
        "rewrite": "primary_rewrite",
        "locator": "scope_locator",
        "entity": "entity_locator",
        "scope": "scope_locator",
        "terminology": "terminology_variant",
    }
    source_value = adapted.get("source")
    if isinstance(source_value, str) and source_value in source_aliases:
        adapted["source"] = source_aliases[source_value]
    elif isinstance(source_value, str) and source_value not in ALLOWED_SOURCES:
        adapted["source"] = "model_reformulation"
    gain_value = adapted.get("expected_information_gain")
    if isinstance(gain_value, str) and gain_value in gain_aliases:
        adapted["expected_information_gain"] = gain_aliases[gain_value]
    if adapted.get("stop_after_round") == 1 and adapted.get("queries"):
        adapted["stop_after_round"] = 2

    raw_queries = adapted.get("queries")
    if isinstance(raw_queries, list):
        normalized_queries = []
        for index, item in enumerate(raw_queries):
            if isinstance(item, str):
                normalized_queries.append({"query": item, "query_role": "primary_rewrite" if index == 0 else "secondary_rewrite"})
                continue
            if isinstance(item, dict):
                query_value = item.get("query") or item.get("search_query") or item.get("text")
                role_value = item.get("query_role") or item.get("role") or ("primary_rewrite" if index == 0 else "secondary_rewrite")
                if role_value in role_aliases:
                    role_value = role_aliases[str(role_value)]
                normalized_queries.append({"query": query_value, "query_role": role_value})
                continue
            normalized_queries.append(item)
        adapted["queries"] = normalized_queries
    return adapted
