from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from opk_rag.answer.models import AnswerResponse
    from opk_rag.search.models import SearchResponse


SHADOW_SCHEMA_VERSION = "opk-rag.agentic-v2.selective-shadow-observation.v1"
SHADOW_ENV = "OPK_RAG_SELECTIVE_AGENT_SHADOW_ENABLED"


def _env_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def shadow_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return _env_true(values.get(SHADOW_ENV))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base(query: str, *, execution_scope: str) -> dict[str, Any]:
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "runtime_role": "shadow_observer",
        "execution_scope": execution_scope,
        "query_digest": _digest(query),
        "raw_query_persisted": False,
        "authoritative": False,
        "production_mutation_allowed": False,
        "llm_finish_authority": False,
        "abstain_to_finish_override_allowed": False,
        "max_graph_hop": 1,
    }


_AMBIGUITY_MARKERS = (
    "之前那个", "前面那个", "上面那个", "刚才那个", "这个问题", "那个问题",
    "后来怎么样", "这个呢", "那个呢", "它呢", "这件事", "这个方案", "那个方案",
)
_GRAPH_MARKERS = ("链接", "关联", "显式关系", "跨文档", "关系图", "路线图", "G0-G2")
_STRUCTURE_MARKERS = ("章节", "同一文档", "上下文", "相关测试笔记", "技术路线")
_EXACT_MARKERS = ("版本号", "版本是多少", "最新版本", "端口", "地址", "价格", "订阅价格", "数值", "QPS", "TTL", "路径")
_COMPARISON_MARKERS = ("相比", "比较", "哪个更", "谁更", "更快", "更慢", "更高", "更低", "更好", "更差")
_VERSION_RE = re.compile(r"(?<!\d)[vV]?\d+\.\d+(?:\.\d+)?(?:[-+._a-zA-Z0-9]*)?")
_NUMERIC_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*(?:ms|s|秒|毫秒|%|MB|GB|GiB|MiB|QPS|元|美元|USD|CNY|PHP)?", re.I)


def observe_search_response(search_response: SearchResponse, *, execution_scope: str) -> dict[str, Any]:
    query = search_response.query
    obs = _base(query, execution_scope=execution_scope)
    signals = search_response.evidence_signals
    graph_trace = dict(search_response.graph_trace or {})
    guard_trace = dict(search_response.guard_trace or {})
    top_score = signals.top_reranker_score if signals is not None else None
    evidence_count = len(search_response.evidence_bundle.items) if search_response.evidence_bundle is not None else 0
    ambiguity = any(marker in query for marker in _AMBIGUITY_MARKERS)
    graph_opportunity = any(marker.lower() in query.lower() for marker in _GRAPH_MARKERS) and not bool(graph_trace.get("graph_activated"))
    structure_opportunity = any(marker in query for marker in _STRUCTURE_MARKERS) and not bool(guard_trace.get("structure_lane_invoked"))
    weak = evidence_count <= 0 or top_score is None or top_score < 0.35
    role: str | None = None
    reason: str
    if ambiguity:
        role, reason = "ambiguity_rewrite", "query_ambiguity_detected"
    elif graph_opportunity:
        role, reason = "recovery_selection", "graph_recovery_opportunity"
    elif structure_opportunity:
        role, reason = "recovery_selection", "structure_recovery_opportunity"
    elif weak:
        role, reason = "recovery_selection", "retrieval_evidence_insufficient"
    else:
        reason = "clear_search_no_agent_needed"
    obs.update({
        "llm_invocation_recommended": role is not None,
        "shadow_role": role,
        "reason_code": reason,
        "ambiguity_detected": ambiguity,
        "recovery_opportunity_detected": role == "recovery_selection",
        "evidence_count": evidence_count,
        "top_rerank_score": top_score,
        "structure_lane_invoked": bool(guard_trace.get("structure_lane_invoked")),
        "graph_activated": bool(graph_trace.get("graph_activated")),
        "candidate_identity_digest": _digest("|".join(str(x.chunk_id) for x in search_response.results)),
        "result_count": search_response.result_count,
    })
    return obs


def _evidence_text(search_response: SearchResponse) -> str:
    bundle = search_response.evidence_bundle
    if bundle is None:
        return ""
    return "\n".join(item.content for item in bundle.items[:5])


def observe_answer_response(answer: AnswerResponse) -> dict[str, Any]:
    query = answer.search_response.query
    obs = _base(query, execution_scope="ask")
    text = _evidence_text(answer.search_response)
    qlow = query.lower()
    top_score = answer.search_response.evidence_signals.top_reranker_score if answer.search_response.evidence_signals is not None else None
    reasons: list[str] = []
    exact = any(marker.lower() in qlow for marker in _EXACT_MARKERS)
    if exact:
        if "版本" in query:
            supported = bool(_VERSION_RE.search(text))
        elif "路径" in query:
            supported = bool(re.search(r"(?:^|[\s`])(?:~?/|%[A-Z_]+%\\|[A-Za-z]:\\)", text))
        else:
            supported = bool(_NUMERIC_RE.search(text))
        if not supported:
            reasons.append("exact_fact_support_missing")
    comparison = any(marker.lower() in qlow for marker in _COMPARISON_MARKERS)
    if comparison and not any(marker in text.lower() for marker in ("相比", "比较", "更快", "更慢", "高于", "低于", "分别为", "vs", "versus", "延迟", "耗时")):
        reasons.append("comparison_support_missing")
    controller_answerability = answer.controller_answerability or answer.answerability
    if controller_answerability.status == "partially_answerable" and (top_score is None or top_score < 0.35):
        reasons.append("partial_answerability_conflict")
    if answer.status == "answered" and controller_answerability.status == "answerable" and (top_score is None or top_score < 0.35) and not reasons:
        reasons.append("weak_evidence_conflict")
    invoke = answer.status == "answered" and bool(reasons)
    obs.update({
        "production_terminal": "finished" if answer.status == "answered" else "abstained",
        "llm_invocation_recommended": invoke,
        "shadow_role": "abstention_veto" if invoke else None,
        "reason_code": reasons[0] if reasons else ("terminal_not_finish" if answer.status != "answered" else "clear_finish"),
        "conflict_reason_codes": tuple(reasons),
        "answerability_status": controller_answerability.status,
        "top_rerank_score": top_score,
        "shadow_terminal_authority": "finish_to_abstain_only",
    })
    return obs


def maybe_observe_selective_agent_search_shadow(
    search_response: SearchResponse, *, execution_scope: str, env: Mapping[str, str] | None = None
) -> Mapping[str, Any] | None:
    if not shadow_enabled(env):
        return None
    try:
        return observe_search_response(search_response, execution_scope=execution_scope)
    except Exception as exc:  # Shadow failure must never escape into Production.
        out = _base(search_response.query, execution_scope=execution_scope)
        out.update({
            "shadow_failure": True,
            "failure_code": f"shadow_observer_{type(exc).__name__}",
            "llm_invocation_recommended": False,
            "shadow_role": None,
        })
        return out


def maybe_observe_selective_agent_answer_shadow(
    answer: AnswerResponse, *, env: Mapping[str, str] | None = None
) -> Mapping[str, Any] | None:
    if not shadow_enabled(env):
        return None
    try:
        return observe_answer_response(answer)
    except Exception as exc:  # Shadow failure must never escape into Production.
        out = _base(answer.search_response.query, execution_scope="ask")
        out.update({
            "shadow_failure": True,
            "failure_code": f"shadow_observer_{type(exc).__name__}",
            "llm_invocation_recommended": False,
            "shadow_role": None,
        })
        return out


@dataclass(frozen=True)
class ShadowIsolationSnapshot:
    query: str
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def isolation_snapshot(search_response: SearchResponse) -> ShadowIsolationSnapshot:
    bundle = search_response.evidence_bundle
    return ShadowIsolationSnapshot(
        query=search_response.query,
        candidate_ids=tuple(str(x.chunk_id) for x in search_response.results),
        evidence_ids=tuple(str(x.chunk_id) for x in bundle.items) if bundle is not None else (),
    )
