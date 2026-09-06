from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4


TRACE_SCHEMA_VERSION = "opk-rag.runtime-trace.v1"
TRACE_CONTRACT_VERSION = "opk-rag.runtime-trace-contract.v1"
TRACE_LIFECYCLE_STATES = ("started", "running", "completed", "failed", "refused", "partial")
STAGE_STATES = ("not_started", "active", "completed", "skipped", "not_applicable", "failed", "unavailable")
TIMING_UNIT = "milliseconds"

TOP_LEVEL_SECTIONS = (
    "trace",
    "query",
    "runtime",
    "guard",
    "retrieval",
    "structure_recovery",
    "graph_recovery",
    "rerank",
    "evidence",
    "answerability",
    "generation",
    "grounding",
    "citation",
    "timings",
    "outcome",
)

FORBIDDEN_TRACE_KEYS = {
    "chain_of_thought",
    "internal_reasoning",
    "hidden_reasoning",
    "scratchpad",
    "planner_thoughts",
    "api_key",
    "password",
    "authorization",
    "access_token",
    "refresh_token",
    "database_url",
    "dsn",
}

SENSITIVE_VALUE_MARKERS = (
    "authorization: bearer ",
    "postgresql://",
    "postgres://",
    "sk-proj-",
    "sk-ant-",
)

SOURCE_KIND_INITIAL = "initial"
SOURCE_KIND_STRUCTURE = "structure_expansion"
SOURCE_KIND_GRAPH = "graph_recovery"


def utc_iso8601_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class RuntimeTraceContext:
    query_text: str
    execution_scope: str
    query_id: str | None = None
    scenario_id: str | None = None
    enabled: bool = True
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=utc_iso8601_now)
    completed_at: str | None = None
    _started_perf: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def complete(self) -> None:
        if self.enabled and self.completed_at is None:
            self.completed_at = utc_iso8601_now()

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started_perf) * 1000.0, 3)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _copy_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _stage_state(*, completed: bool = False, skipped: bool = False, applicable: bool = True, available: bool = True, failed: bool = False) -> str:
    if failed:
        return "failed"
    if not applicable:
        return "not_applicable"
    if not available:
        return "unavailable"
    if skipped:
        return "skipped"
    return "completed" if completed else "not_started"


def _scenario_lifecycle(scenario: Mapping[str, Any]) -> str:
    if str(scenario.get("execution_status") or "") not in {"passed", "showcase_drift"}:
        return "failed"
    answer = _copy_dict(scenario.get("answer"))
    guard = _copy_dict(scenario.get("guard"))
    if answer.get("status") == "refused" or guard.get("fail_closed") is True:
        return "refused"
    return "completed"


def _execution_scope(scenario: Mapping[str, Any]) -> str:
    command = str(scenario.get("command") or "")
    return "ask" if command == "ask" else "search" if command == "search" else "demo"


def _candidate_provenance(sources: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if source in {"vector", "bm25"}:
            rows.append({"source_kind": SOURCE_KIND_INITIAL, "source_lane": source})
        elif source == "structure":
            rows.append({"source_kind": SOURCE_KIND_STRUCTURE, "source_lane": source})
        elif source == "graph":
            rows.append({"source_kind": SOURCE_KIND_GRAPH, "source_lane": source})
    return rows


def _candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    sources = [str(value) for value in _copy_list(row.get("retrieval_sources"))]
    chunk_id = str(row.get("chunk_id") or "")
    return {
        "candidate_id": chunk_id or None,
        "chunk_id": chunk_id or None,
        "document_id": str(row.get("document_id") or "") or None,
        "document_path": row.get("relative_path"),
        "section_path": None,
        "text_preview": None,
        "retrieval_sources": sources,
        "provenance": _candidate_provenance(sources),
        "retrieval_score": None,
        "initial_rank": None,
        "final_rank": int(row.get("rank")) if row.get("rank") is not None else None,
        "rerank_score": None,
        "structure_expanded": "structure" in sources,
        "graph_recovered": "graph" in sources,
        "selected_for_context": row.get("selected_for_context") is True,
    }


def _live_candidate(result: Any) -> dict[str, Any]:
    sources = [str(value) for value in getattr(result, "retrieval_sources", ()) or ()]
    chunk_id = str(getattr(result, "chunk_id", "") or "")
    return {
        "candidate_id": chunk_id or None,
        "chunk_id": chunk_id or None,
        "document_id": str(getattr(result, "document_id", "") or "") or None,
        "document_path": getattr(result, "relative_path", None),
        "section_path": list(getattr(result, "heading_path", ()) or ()),
        "text_preview": None,
        "retrieval_sources": sources,
        "provenance": _candidate_provenance(sources),
        "retrieval_score": getattr(result, "similarity", None),
        "initial_rank": getattr(result, "original_rank", None),
        "final_rank": getattr(result, "rank", None),
        "rerank_score": getattr(result, "rerank_score", None),
        "structure_expanded": "structure" in sources,
        "graph_recovered": "graph" in sources,
        "selected_for_context": getattr(result, "selected_for_context", False) is True,
    }


def _evidence(row: Mapping[str, Any], position: int) -> dict[str, Any]:
    sources = [str(value) for value in _copy_list(row.get("retrieval_sources"))]
    chunk_id = str(row.get("chunk_id") or "")
    return {
        "evidence_id": f"evidence:{chunk_id}" if chunk_id else f"evidence:position:{position}",
        "source_candidate_id": chunk_id or None,
        "chunk_id": chunk_id or None,
        "document_id": str(row.get("document_id") or "") or None,
        "document_path": row.get("relative_path"),
        "section_path": None,
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
        "evidence_position": position,
        "evidence_score": None,
        "selection_reason_code": "selected_for_context",
        "source_provenance": _candidate_provenance(sources),
        "retrieval_sources": sources,
    }


def _live_evidence(item: Any, position: int) -> dict[str, Any]:
    sources = [str(value) for value in getattr(item, "retrieval_sources", ()) or ()]
    chunk_id = str(getattr(item, "chunk_id", "") or "")
    citation_id = f"C{position}"
    return {
        "evidence_id": citation_id,
        "source_candidate_id": chunk_id or None,
        "chunk_id": chunk_id or None,
        "document_id": str(getattr(item, "document_id", "") or "") or None,
        "document_path": getattr(item, "relative_path", None),
        "section_path": list(getattr(item, "heading_path", ()) or ()),
        "start_line": getattr(item, "start_line", None),
        "end_line": getattr(item, "end_line", None),
        "evidence_position": position,
        "evidence_score": getattr(item, "rerank_score", None),
        "selection_reason_code": "selected_for_context",
        "source_provenance": _candidate_provenance(sources),
        "retrieval_sources": sources,
    }


def _graph_edges(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _copy_list(graph.get("relations")):
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "edge_id": raw.get("edge_id"),
                "source_node_id": raw.get("source_node_id"),
                "target_node_id": raw.get("target_node_id"),
                "relation_type": raw.get("relation_type"),
                "source_document_id": raw.get("source_document_id"),
                "target_document_id": raw.get("target_document_id"),
                "hop_depth": raw.get("hop_depth"),
                "edge_observation": "traversed_query_edge",
            }
        )
    return rows


def _recovery_reason(guard: Mapping[str, Any], graph: Mapping[str, Any]) -> str | None:
    if graph.get("graph_activated") is True:
        return str(graph.get("graph_activation_reason") or "") or None
    if guard.get("structure_lane_invoked") is True:
        return str(guard.get("guard_reason") or "") or None
    return None


def _recovery_action(guard: Mapping[str, Any], graph: Mapping[str, Any]) -> str:
    if graph.get("graph_activated") is True:
        return "graph_recovery"
    if guard.get("structure_lane_invoked") is True:
        return "structure_recovery"
    return "none"


def _retrievers_used(scenario: Mapping[str, Any]) -> list[str]:
    retrieval = _copy_dict(scenario.get("retrieval"))
    result: list[str] = []
    mode = str(retrieval.get("retrieval_mode") or "")
    if mode == "vector":
        result.append("vector")
    elif mode == "bm25":
        result.append("lexical")
    elif mode == "hybrid":
        result.extend(("vector", "lexical", "hybrid"))
    for row in _copy_list(scenario.get("top_candidates")):
        if not isinstance(row, Mapping):
            continue
        for source in _copy_list(row.get("retrieval_sources")):
            normalized = "lexical" if source == "bm25" else str(source)
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def _retrievers_used_from_response(response: Any) -> list[str]:
    result: list[str] = []
    mode = str(getattr(response, "retrieval_mode", "") or "")
    if mode == "vector":
        result.append("vector")
    elif mode == "bm25":
        result.append("lexical")
    elif mode == "hybrid":
        result.extend(("vector", "lexical", "hybrid"))
    for row in getattr(response, "results", ()) or ():
        for source in getattr(row, "retrieval_sources", ()) or ():
            normalized = "lexical" if source == "bm25" else str(source)
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def _reranker_precision(reranker_config: Any, execution_scope: str) -> str | None:
    if reranker_config is None:
        return None
    return getattr(reranker_config, "ask_precision" if execution_scope == "ask" else "search_precision", None)


def _timing_ms(snapshot: Mapping[str, Any], *stages: str) -> float | None:
    """Return a non-overlapping timing aggregate from RetrievalTimingObserver.

    A single requested stage uses its inclusive latency so the stage includes its
    instrumented children. Multi-stage aggregates use exclusive latency to avoid
    double-counting nested observer scopes.
    """
    stage_rows = _copy_dict(snapshot.get("stages"))
    executed = [_copy_dict(stage_rows.get(stage)) for stage in stages]
    executed = [row for row in executed if row.get("stage_executed") is True]
    if not executed:
        return None
    if len(stages) == 1:
        row = executed[0]
        return round(float(row.get("inclusive_latency_ms") or row.get("latency_ms") or 0.0), 6)
    return round(sum(float(row.get("latency_ms") or 0.0) for row in executed), 6)


def _answerability_payload(answerability: Any | None, *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {
            "stage_state": "not_applicable",
            "answerability_state": None,
            "answerable": None,
            "reason_code": None,
            "evidence_count": None,
            "required_support_state": None,
            "note": None,
        }
    if answerability is None:
        return {
            "stage_state": "unavailable",
            "answerability_state": None,
            "answerable": None,
            "reason_code": None,
            "evidence_count": None,
            "required_support_state": None,
            "note": "AnswerabilityDecision was not available to the trace context.",
        }
    return {
        "stage_state": "completed",
        "answerability_state": getattr(answerability, "status", None),
        "answerable": getattr(answerability, "answerable", None),
        "reason_code": getattr(answerability, "reason_code", None),
        "evidence_count": getattr(answerability, "evidence_count", None),
        "required_support_state": getattr(answerability, "status", None),
        "confidence": getattr(answerability, "confidence", None),
        "considered_evidence_count": getattr(answerability, "considered_evidence_count", None),
        "evidence_chunk_ids": list(getattr(answerability, "evidence_chunk_ids", ()) or ()),
        "note": None,
    }


def _grounding_payload(answer: Any | None, *, applicable: bool) -> dict[str, Any]:
    grounding = getattr(answer, "grounding", None) if answer is not None else None
    if not applicable:
        return {
            "stage_state": "not_applicable",
            "grounding_checked": None,
            "grounding_passed": None,
            "supported_claim_count": None,
            "unsupported_claim_count": None,
            "failure_reason_code": None,
        }
    if grounding is None:
        return {
            "stage_state": "unavailable",
            "grounding_checked": None,
            "grounding_passed": None,
            "supported_claim_count": None,
            "unsupported_claim_count": None,
            "failure_reason_code": None,
        }
    return {
        "stage_state": "completed",
        "grounding_checked": True,
        "grounding_passed": bool(getattr(grounding, "valid", False)),
        "supported_claim_count": None,
        "unsupported_claim_count": len(getattr(answer, "unsupported_claims", ()) or ()),
        "failure_reason_code": None if getattr(grounding, "valid", False) else getattr(grounding, "reason_code", None),
        "status": getattr(grounding, "status", None),
        "reason_code": getattr(grounding, "reason_code", None),
        "cited_ids": list(getattr(grounding, "cited_ids", ()) or ()),
        "valid_cited_ids": list(getattr(grounding, "valid_cited_ids", ()) or ()),
        "invalid_cited_ids": list(getattr(grounding, "invalid_cited_ids", ()) or ()),
        "available_evidence_ids": list(getattr(grounding, "available_evidence_ids", ()) or ()),
        "citation_coverage": getattr(grounding, "citation_coverage", None),
    }


def _citation_payload(answer: Any | None, evidence_items: Sequence[Mapping[str, Any]], *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {
            "stage_state": "not_applicable",
            "citation_checked": None,
            "citation_valid": None,
            "citation_count": None,
            "invalid_citation_count": None,
            "citations": [],
            "note": None,
        }
    if answer is None:
        return {
            "stage_state": "unavailable",
            "citation_checked": None,
            "citation_valid": None,
            "citation_count": None,
            "invalid_citation_count": None,
            "citations": [],
            "note": "AnswerResponse was not available to the trace context.",
        }
    evidence_by_citation = {str(row.get("evidence_id")): row for row in evidence_items}
    citations = []
    for citation in getattr(answer, "citations", ()) or ():
        evidence = evidence_by_citation.get(str(getattr(citation, "citation_id", "")))
        citations.append(
            {
                "citation_id": getattr(citation, "citation_id", None),
                "evidence_id": getattr(citation, "citation_id", None),
                "source_candidate_id": evidence.get("source_candidate_id") if evidence else getattr(citation, "chunk_id", None),
                "chunk_id": getattr(citation, "chunk_id", None),
                "document_id": getattr(citation, "document_id", None),
                "document_path": getattr(citation, "relative_path", None),
                "section_path": list(getattr(citation, "heading_path", ()) or ()),
                "start_line": getattr(citation, "start_line", None),
                "end_line": getattr(citation, "end_line", None),
            }
        )
    invalid_count = len(getattr(getattr(answer, "grounding", None), "invalid_cited_ids", ()) or ())
    return {
        "stage_state": "completed",
        "citation_checked": True,
        "citation_valid": invalid_count == 0,
        "citation_count": len(citations),
        "invalid_citation_count": invalid_count,
        "citations": citations,
        "note": None,
    }


def build_runtime_trace_from_live_execution(
    context: RuntimeTraceContext,
    *,
    search_response: Any,
    timing_snapshot: Mapping[str, Any] | None = None,
    embedding_config: Any | None = None,
    reranker_config: Any | None = None,
    qdrant_config: Any | None = None,
    vector_backend: str | None = None,
    reranker_device: str | None = None,
    answer: Any | None = None,
    answer_config: Any | None = None,
    status: str | None = None,
    trace_complete: bool = True,
) -> dict[str, Any]:
    """Build Runtime Trace V1 from objects produced by one live Search/Ask execution."""
    if not context.enabled:
        return {}
    if answer is not None or context.execution_scope != "ask" or status in {"failed", "refused", "completed", "partial"}:
        context.complete()
    timing_snapshot = dict(timing_snapshot or {})
    execution_scope = context.execution_scope
    is_ask = execution_scope == "ask"
    graph = _copy_dict(getattr(search_response, "graph_trace", None))
    guard = _copy_dict(getattr(search_response, "guard_trace", None))
    candidates = [_live_candidate(row) for row in getattr(search_response, "results", ()) or ()]
    evidence_bundle = getattr(search_response, "evidence_bundle", None)
    evidence_items = [_live_evidence(item, index) for index, item in enumerate(getattr(evidence_bundle, "items", ()) or (), start=1)]
    graph_activated = graph.get("graph_activated") is True
    graph_evaluated = graph.get("graph_activation_evaluated") is True
    structure_triggered = guard.get("structure_lane_invoked") is True
    guard_evaluated = guard.get("guard_evaluated") is True
    lifecycle = status or ("refused" if answer is not None and getattr(answer, "status", None) == "refused" else "running" if is_ask and answer is None else "completed")
    prior_runtime = _copy_dict(_copy_dict(getattr(search_response, "runtime_trace", None)).get("runtime"))
    reranker_precision = _reranker_precision(reranker_config, execution_scope) or prior_runtime.get("reranker_precision")
    recovery_after_count = guard.get("candidate_pool_count_after_recovery") or guard.get("candidate_pool_count_after_graph") or guard.get("candidate_pool_count_after_structure")
    structure_ids = [row["candidate_id"] for row in candidates if row["structure_expanded"]]
    graph_ids = [str(value) for value in _copy_list(graph.get("expanded_chunk_ids"))]
    generation_attempted = answer is not None and getattr(answer, "generation_latency_ms", None) is not None
    trace = {
        "trace": {
            "trace_id": context.trace_id,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_contract_version": TRACE_CONTRACT_VERSION,
            "query_execution_identity_origin": "runtime_uuid_v4",
            "source_authority": "live_runtime_execution",
            "source_authority_sha256": None,
            "started_at": context.started_at,
            "completed_at": context.completed_at,
            "status": lifecycle,
            "captured_from_frozen_authority": False,
            "trace_complete": trace_complete,
        },
        "query": {
            "stage_state": "completed",
            "query_id": context.query_id or getattr(search_response, "query_input_hash", None),
            "query_text": getattr(search_response, "query", context.query_text),
            "normalized_query": getattr(search_response, "normalized_query", None),
            "scenario_id": context.scenario_id,
            "execution_scope": execution_scope,
            "requested_top_k": getattr(search_response, "requested_top_k", None),
            "candidate_k": getattr(search_response, "candidate_k", None),
        },
        "runtime": {
            "stage_state": "completed",
            "vector_backend": vector_backend or ("qdrant" if qdrant_config is not None else None) or prior_runtime.get("vector_backend"),
            "vector_collection": getattr(qdrant_config, "collection", None) or prior_runtime.get("vector_collection"),
            "knowledge_base_id": str(getattr(search_response, "knowledge_base_id", "") or "") or None,
            "embedding_model": getattr(embedding_config, "model_name", None) or prior_runtime.get("embedding_model") or getattr(search_response, "model_id", None),
            "embedding_revision": getattr(embedding_config, "model_revision", None) or prior_runtime.get("embedding_revision"),
            "embedding_dimension": getattr(embedding_config, "dimension", None) or prior_runtime.get("embedding_dimension"),
            "reranker_model": getattr(reranker_config, "model_name", None) or prior_runtime.get("reranker_model") or getattr(search_response, "reranker_model_id", None),
            "reranker_revision": getattr(search_response, "reranker_model_revision", None) or getattr(reranker_config, "model_revision", None) or prior_runtime.get("reranker_revision"),
            "reranker_device": reranker_device or prior_runtime.get("reranker_device"),
            "reranker_precision": reranker_precision,
            "initial_retrieval_policy": guard.get("initial_retrieval_policy"),
            "graph_runtime_enabled": graph.get("graph_enabled"),
            "graph_hop_depth": graph.get("hop_depth"),
            "maximum_recovery_attempt_count": guard.get("maximum_recovery_attempt_count"),
            "generation_provider": getattr(answer, "provider_id", None) if answer is not None else getattr(answer_config, "provider_id", None),
            "generation_model": getattr(answer, "model_id", None) if answer is not None else getattr(answer_config, "model_id", None),
            "generation_endpoint_type": getattr(answer, "runtime_endpoint_type", None),
        },
        "guard": {
            "stage_state": _stage_state(completed=guard_evaluated, available=bool(guard)),
            "agent_type": guard.get("agent_type"),
            "selected_route": guard.get("initial_retrieval_policy") or guard.get("selected_retrieval_lane"),
            "initial_decision": guard.get("recovery_decision"),
            "final_decision": "answer" if answer is not None and getattr(answer, "status", None) == "answered" else guard.get("final_decision"),
            "guard_triggered": guard.get("guard_triggered"),
            "guard_reason_code": guard.get("guard_reason"),
            "recovery_required": guard.get("recovery_activated"),
            "recovery_reason_code": _recovery_reason(guard, graph),
            "recovery_action": _recovery_action(guard, graph),
            "recovery_attempt_count": guard.get("recovery_attempt_count"),
            "maximum_recovery_attempt_count": guard.get("maximum_recovery_attempt_count"),
            "fail_closed": answer is not None and getattr(answer, "status", None) == "refused",
            "refusal_reason_code": getattr(answer, "refusal_reason_code", None),
            "hidden_chain_of_thought_exposed": False,
        },
        "retrieval": {
            "stage_state": "completed",
            "requested_policy": guard.get("initial_retrieval_policy"),
            "executed_policy": guard.get("initial_retrieval_policy"),
            "retrieval_mode": getattr(search_response, "retrieval_mode", None),
            "retrievers_used": _retrievers_used_from_response(search_response),
            "initial_candidate_count": getattr(search_response, "candidate_count", None),
            "vector_candidate_count": getattr(search_response, "vector_candidate_count", None),
            "lexical_candidate_count": getattr(search_response, "bm25_candidate_count", None),
            "post_structure_candidate_count": guard.get("candidate_pool_count_after_structure"),
            "post_graph_unified_candidate_count": guard.get("candidate_pool_count_after_graph") or recovery_after_count,
            "rerank_input_count": getattr(search_response, "rerank_candidate_count", None),
            "final_result_count": getattr(search_response, "result_count", None),
            "candidate_pool_count_after_recovery": recovery_after_count,
            "observed_candidate_count": len(candidates),
            "observed_candidates_complete": len(candidates) == int(getattr(search_response, "result_count", 0) or 0),
            "candidates": candidates,
        },
        "structure_recovery": {
            "stage_state": _stage_state(completed=structure_triggered, skipped=guard_evaluated and not structure_triggered, available=guard_evaluated),
            "triggered": structure_triggered,
            "reason_code": guard.get("guard_reason") if structure_triggered else None,
            "seed_candidate_ids": _copy_list(guard.get("structure_seed_candidate_ids")),
            "expanded_candidate_ids": _copy_list(guard.get("structure_expanded_candidate_ids")) or structure_ids,
            "expanded_candidate_count": int(guard.get("structure_candidate_count") or len(structure_ids)),
            "expanded_candidate_ids_complete": True,
            "source_document_ids": _copy_list(guard.get("structure_expanded_document_ids")) or sorted({str(row.get("document_id")) for row in candidates if row["structure_expanded"] and row.get("document_id")}),
            "candidate_pool_count_before": guard.get("candidate_pool_count_before_structure"),
            "candidate_pool_count_after": guard.get("candidate_pool_count_after_structure"),
        },
        "graph_recovery": {
            "stage_state": _stage_state(completed=graph_activated, skipped=graph_evaluated and not graph_activated, available=graph_evaluated),
            "graph_activated": graph_activated,
            "activation_reason_code": graph.get("graph_activation_reason"),
            "activation_policy": graph.get("graph_activation_policy"),
            "hop_depth": graph.get("hop_depth"),
            "seed_node_ids": _copy_list(graph.get("seed_document_ids")),
            "seed_candidate_ids": _copy_list(graph.get("seed_chunk_ids")),
            "traversed_edges": _graph_edges(graph),
            "recovered_node_ids": _copy_list(graph.get("expanded_document_ids")),
            "recovered_candidate_ids": graph_ids,
            "recovered_candidate_count": int(graph.get("expanded_candidate_count") or 0),
            "candidate_pool_count_before": graph.get("candidate_pool_count_before"),
            "candidate_pool_count_after": graph.get("candidate_pool_count_after"),
            "candidate_provenance": _copy_list(graph.get("candidate_provenance")),
            "runtime_gold_metadata_usage": graph.get("runtime_gold_metadata_usage"),
        },
        "rerank": {
            "stage_state": _stage_state(completed=getattr(search_response, "reranker_enabled", None) is True, skipped=getattr(search_response, "reranker_enabled", None) is False, available=getattr(search_response, "reranker_enabled", None) is not None),
            "reranker_model": getattr(search_response, "reranker_model_id", None),
            "reranker_revision": getattr(search_response, "reranker_model_revision", None),
            "reranker_device": reranker_device or prior_runtime.get("reranker_device"),
            "execution_scope": execution_scope,
            "precision": reranker_precision,
            "input_candidate_count": getattr(search_response, "rerank_candidate_count", None),
            "output_candidate_count": getattr(search_response, "result_count", None),
            "ranked_candidates": [
                {"candidate_id": row["candidate_id"], "rerank_score": row["rerank_score"], "rerank_position": row["final_rank"]}
                for row in candidates
            ],
        },
        "evidence": {
            "stage_state": _stage_state(completed=evidence_bundle is not None, available=evidence_bundle is not None),
            "evidence_count": len(evidence_items),
            "evidence_items": evidence_items,
            "candidate_evidence_semantic_separation": True,
            "provenance_available": bool(evidence_items),
        },
        "answerability": _answerability_payload(getattr(answer, "answerability", None) if answer is not None else None, applicable=is_ask),
        "generation": {
            "stage_state": _stage_state(completed=generation_attempted, skipped=is_ask and not generation_attempted, applicable=is_ask, available=answer is not None if is_ask else True),
            "generation_attempted": generation_attempted if is_ask else None,
            "generation_provider": getattr(answer, "provider_id", None) if answer is not None else getattr(answer_config, "provider_id", None),
            "generation_model": getattr(answer, "model_id", None) if answer is not None else getattr(answer_config, "model_id", None),
            "generation_revision": getattr(answer, "model_revision", None) if answer is not None else getattr(answer_config, "model_revision", None),
            "generation_endpoint_type": getattr(answer, "runtime_endpoint_type", None),
            "generation_latency_ms": getattr(answer, "generation_latency_ms", None),
            "prompt_tokens": getattr(answer, "prompt_tokens", None),
            "completion_tokens": getattr(answer, "completion_tokens", None),
            "total_tokens": getattr(answer, "total_tokens", None),
            "finish_reason": getattr(answer, "finish_reason", None),
            "generation_completed": generation_attempted,
            "generation_abstained": getattr(answer, "refusal_reason_code", None) in {"model_abstained", "answerable_generation_abstained"} if answer is not None else None,
            "prompt_digest": None,
            "prompt_length": None,
            "evidence_count": len(evidence_items),
        },
        "grounding": _grounding_payload(answer, applicable=is_ask),
        "citation": _citation_payload(answer, evidence_items, applicable=is_ask),
        "timings": {
            "stage_state": "completed" if timing_snapshot.get("enabled") is True else "unavailable",
            "timing_unit": TIMING_UNIT,
            "timing_source": "RetrievalTimingObserver",
            "total_ms": context.elapsed_ms,
            "query_preparation_ms": _timing_ms(timing_snapshot, "query_preprocessing"),
            "embedding_ms": _timing_ms(timing_snapshot, "query_embedding"),
            "vector_search_ms": _timing_ms(timing_snapshot, "vector_search"),
            "lexical_search_ms": _timing_ms(timing_snapshot, "lexical_search"),
            "structure_expansion_ms": _timing_ms(timing_snapshot, "structure_activation_decision", "structure_expansion"),
            "graph_recovery_ms": _timing_ms(timing_snapshot, "document_relation_lookup", "graph_lookup", "graph_neighbor_lookup", "graph_candidate_merge"),
            "reranking_ms": _timing_ms(timing_snapshot, "reranking_total"),
            "evidence_composition_ms": _timing_ms(timing_snapshot, "evidence_composition"),
            "generation_ms": getattr(answer, "generation_latency_ms", None),
            "validation_ms": None,
            "stage_timings_available": timing_snapshot.get("enabled") is True,
            "retrieval_timing_snapshot": timing_snapshot if timing_snapshot.get("enabled") is True else None,
            "instrumentation_overhead_ms": timing_snapshot.get("instrumentation_overhead_ms"),
            "historical_benchmark_values_mixed_into_live_trace": False,
        },
        "outcome": {
            "status": lifecycle,
            "answer_status": getattr(answer, "status", None) if answer is not None else None,
            "refused": lifecycle == "refused",
            "failure_stage": None,
            "final_candidate_count": getattr(search_response, "result_count", None),
            "final_evidence_count": len(evidence_items),
            "grounding_passed": getattr(getattr(answer, "grounding", None), "valid", None) if answer is not None else None,
            "citation_valid": None,
            "final_decision": "answer" if answer is not None and getattr(answer, "status", None) == "answered" else guard.get("final_decision"),
            "refusal_reason_code": getattr(answer, "refusal_reason_code", None),
        },
    }
    trace["trace"]["unavailable_fields"] = _unavailable_fields(trace)
    if not trace_complete:
        trace["trace"]["unavailable_fields"].append("trace.trace_complete")
    trace["trace"]["trace_semantic_digest"] = trace_semantic_digest(trace)
    return trace


def _unavailable_fields(trace: Mapping[str, Any]) -> list[str]:
    candidate_fields = [
        ("trace", "started_at"),
        ("trace", "completed_at"),
        ("query", "requested_top_k"),
        ("runtime", "vector_collection"),
        ("runtime", "embedding_revision"),
        ("runtime", "reranker_revision"),
        ("runtime", "reranker_device"),
        ("runtime", "reranker_precision"),
        ("runtime", "generation_provider"),
        ("runtime", "generation_model"),
        ("runtime", "generation_endpoint_type"),
        ("retrieval", "candidate_pool_count_after_recovery"),
        ("rerank", "input_candidate_count"),
        ("timings", "query_preparation_ms"),
        ("timings", "embedding_ms"),
        ("timings", "vector_search_ms"),
        ("timings", "lexical_search_ms"),
        ("timings", "structure_expansion_ms"),
        ("timings", "graph_recovery_ms"),
        ("timings", "reranking_ms"),
        ("timings", "evidence_composition_ms"),
        ("timings", "generation_ms"),
        ("timings", "validation_ms"),
    ]
    fields = [f"{section}.{name}" for section, name in candidate_fields if _copy_dict(trace.get(section)).get(name) is None]
    if trace["query"]["execution_scope"] != "ask":
        fields.extend(("answerability.answerability_state", "generation.generation_attempted"))
    return fields


def build_runtime_trace_from_showcase(
    scenario: Mapping[str, Any],
    *,
    source_authority: str,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Normalize an already-captured Showcase execution into Runtime Trace V1.

    This function is presentation/observability-only. It does not execute retrieval,
    Guard policy, Graph traversal, reranking, Evidence selection, or generation.
    Unknown values remain null/unavailable rather than being reconstructed from
    benchmark averages or UI assumptions.
    """

    retrieval = _copy_dict(scenario.get("retrieval"))
    guard = _copy_dict(scenario.get("guard"))
    graph = _copy_dict(scenario.get("graph"))
    evidence = _copy_dict(scenario.get("evidence"))
    runtime = _copy_dict(scenario.get("runtime"))
    answer = _copy_dict(scenario.get("answer"))
    candidates = [_candidate(row) for row in _copy_list(scenario.get("top_candidates")) if isinstance(row, Mapping)]
    evidence_items = [_evidence(row, index) for index, row in enumerate(_copy_list(evidence.get("selected_evidence")), start=1) if isinstance(row, Mapping)]

    source_digest = source_sha256 or semantic_digest(dict(scenario))
    trace_id = f"trace-sha256:{source_digest}"
    execution_scope = _execution_scope(scenario)
    lifecycle = _scenario_lifecycle(scenario)
    graph_activated = graph.get("graph_activated") is True
    graph_evaluated = graph.get("graph_activation_evaluated") is True
    structure_triggered = guard.get("structure_lane_invoked") is True
    guard_evaluated = guard.get("guard_evaluated") is True
    structure_ids = [row["candidate_id"] for row in candidates if row["structure_expanded"]]
    graph_ids = [str(value) for value in _copy_list(graph.get("expanded_chunk_ids"))]

    answer_is_present = bool(answer)
    generation_abstained = answer.get("refusal_reason_code") == "answerable_generation_abstained"
    trace: dict[str, Any] = {
        "trace": {
            "trace_id": trace_id,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_contract_version": TRACE_CONTRACT_VERSION,
            "query_execution_identity_origin": "frozen_authority_content_digest",
            "source_authority": source_authority,
            "source_authority_sha256": source_digest,
            "started_at": None,
            "completed_at": None,
            "status": lifecycle,
            "captured_from_frozen_authority": True,
        },
        "query": {
            "stage_state": "completed",
            "query_id": f"scenario:{scenario.get('scenario_id')}" if scenario.get("scenario_id") else None,
            "query_text": scenario.get("query"),
            "scenario_id": scenario.get("scenario_id"),
            "execution_scope": execution_scope,
            "requested_top_k": None,
        },
        "runtime": {
            "stage_state": "completed",
            "vector_backend": runtime.get("vector_backend"),
            "vector_collection": None,
            "knowledge_base_id": runtime.get("knowledge_base_id"),
            "embedding_model": runtime.get("embedding_model"),
            "embedding_revision": None,
            "embedding_dimension": None,
            "reranker_model": runtime.get("reranker"),
            "reranker_revision": None,
            "reranker_device": None,
            "reranker_precision": None,
            "initial_retrieval_policy": runtime.get("retrieval_policy") or guard.get("initial_retrieval_policy"),
            "graph_runtime_enabled": graph.get("graph_enabled"),
            "graph_hop_depth": runtime.get("graph_hop_depth") if runtime.get("graph_hop_depth") is not None else graph.get("hop_depth"),
            "maximum_recovery_attempt_count": guard.get("maximum_recovery_attempt_count"),
            "generation_provider": None,
            "generation_model": None,
            "generation_endpoint_type": None,
        },
        "guard": {
            "stage_state": _stage_state(completed=guard_evaluated, available=bool(guard)),
            "agent_type": guard.get("agent_type"),
            "selected_route": guard.get("initial_retrieval_policy") or runtime.get("retrieval_policy"),
            "initial_decision": guard.get("recovery_decision"),
            "final_decision": guard.get("final_decision"),
            "guard_triggered": guard.get("guard_triggered"),
            "guard_reason_code": guard.get("guard_reason"),
            "recovery_required": guard.get("recovery_activated"),
            "recovery_reason_code": _recovery_reason(guard, graph),
            "recovery_action": _recovery_action(guard, graph),
            "recovery_attempt_count": guard.get("recovery_attempt_count"),
            "maximum_recovery_attempt_count": guard.get("maximum_recovery_attempt_count"),
            "fail_closed": guard.get("fail_closed") is True,
            "refusal_reason_code": guard.get("refusal_reason_code") or answer.get("refusal_reason_code"),
            "hidden_chain_of_thought_exposed": False,
        },
        "retrieval": {
            "stage_state": "completed" if scenario.get("execution_status") == "passed" else "failed",
            "requested_policy": runtime.get("retrieval_policy") or guard.get("initial_retrieval_policy"),
            "executed_policy": guard.get("initial_retrieval_policy") or runtime.get("retrieval_policy"),
            "retrieval_mode": retrieval.get("retrieval_mode"),
            "retrievers_used": _retrievers_used(scenario),
            "initial_candidate_count": retrieval.get("candidate_count"),
            "vector_candidate_count": retrieval.get("vector_candidate_count"),
            "lexical_candidate_count": retrieval.get("bm25_candidate_count"),
            "final_result_count": retrieval.get("result_count"),
            "candidate_pool_count_after_recovery": None,
            "observed_candidate_count": len(candidates),
            "observed_candidates_complete": len(candidates) == int(retrieval.get("result_count") or 0),
            "candidates": candidates,
        },
        "structure_recovery": {
            "stage_state": _stage_state(completed=structure_triggered, skipped=guard_evaluated and not structure_triggered, available=guard_evaluated),
            "triggered": structure_triggered,
            "reason_code": guard.get("guard_reason") if structure_triggered else None,
            "seed_candidate_ids": None,
            "expanded_candidate_ids": structure_ids,
            "expanded_candidate_count": int(guard.get("structure_candidate_count") or 0),
            "expanded_candidate_ids_complete": len(structure_ids) == int(guard.get("structure_candidate_count") or 0),
            "source_document_ids": sorted({str(row.get("document_id")) for row in candidates if row["structure_expanded"] and row.get("document_id")}),
            "candidate_pool_count_before": retrieval.get("candidate_count"),
            "candidate_pool_count_after": None,
        },
        "graph_recovery": {
            "stage_state": _stage_state(completed=graph_activated, skipped=graph_evaluated and not graph_activated, available=graph_evaluated),
            "graph_activated": graph_activated,
            "activation_reason_code": graph.get("graph_activation_reason"),
            "activation_policy": graph.get("graph_activation_policy"),
            "hop_depth": graph.get("hop_depth"),
            "seed_node_ids": _copy_list(graph.get("seed_document_ids")),
            "seed_candidate_ids": _copy_list(graph.get("seed_chunk_ids")),
            "traversed_edges": _graph_edges(graph),
            "recovered_node_ids": _copy_list(graph.get("expanded_document_ids")),
            "recovered_candidate_ids": graph_ids,
            "recovered_candidate_count": int(graph.get("expanded_candidate_count") or 0),
            "candidate_pool_count_before": None,
            "candidate_pool_count_after": None,
            "candidate_provenance": _copy_list(graph.get("candidate_provenance")),
            "runtime_gold_metadata_usage": graph.get("runtime_gold_metadata_usage"),
        },
        "rerank": {
            "stage_state": _stage_state(completed=retrieval.get("reranker_active") is True, skipped=retrieval.get("reranker_active") is False, available=retrieval.get("reranker_active") is not None),
            "reranker_model": retrieval.get("reranker_model_id") or runtime.get("reranker"),
            "execution_scope": execution_scope,
            "precision": None,
            "input_candidate_count": None,
            "output_candidate_count": retrieval.get("result_count"),
            "ranked_candidates": [
                {"candidate_id": row["candidate_id"], "rerank_score": row["rerank_score"], "rerank_position": row["final_rank"]}
                for row in candidates
            ],
        },
        "evidence": {
            "stage_state": _stage_state(completed=isinstance(scenario.get("evidence"), Mapping), available=isinstance(scenario.get("evidence"), Mapping)),
            "evidence_count": int(evidence.get("selected_evidence_count") or len(evidence_items)),
            "evidence_items": evidence_items,
            "candidate_evidence_semantic_separation": True,
            "provenance_available": evidence.get("provenance_available"),
        },
        "answerability": {
            "stage_state": _stage_state(applicable=execution_scope == "ask", available=False if execution_scope == "ask" else True),
            "answerability_state": None,
            "answerable": None,
            "reason_code": None,
            "evidence_count": int(evidence.get("selected_evidence_count") or len(evidence_items)),
            "required_support_state": None,
            "note": "Current frozen Showcase envelope does not preserve the internal AnswerabilityDecision separately from final AnswerResponse.",
        },
        "generation": {
            "stage_state": "completed" if generation_abstained else _stage_state(applicable=execution_scope == "ask", available=False if execution_scope == "ask" else True),
            "generation_attempted": True if generation_abstained else None,
            "generation_provider": None,
            "generation_model": None,
            "generation_completed": True if generation_abstained else None,
            "generation_abstained": True if generation_abstained else None,
            "prompt_digest": None,
            "prompt_length": None,
            "evidence_count": int(evidence.get("selected_evidence_count") or len(evidence_items)),
        },
        "grounding": {
            "stage_state": _stage_state(completed=answer_is_present and answer.get("grounding_valid") is not None, applicable=execution_scope == "ask", available=(answer.get("grounding_valid") is not None) if execution_scope == "ask" else True),
            "grounding_checked": answer.get("grounding_valid") is not None if answer_is_present else None,
            "grounding_passed": answer.get("grounding_valid") if answer_is_present else None,
            "supported_claim_count": None,
            "unsupported_claim_count": None,
            "failure_reason_code": answer.get("grounding_reason_code") if answer_is_present else None,
        },
        "citation": {
            "stage_state": _stage_state(completed=answer_is_present and answer.get("citation_count") is not None, applicable=execution_scope == "ask", available=(answer.get("citation_count") is not None) if execution_scope == "ask" else True),
            "citation_checked": answer.get("citation_count") is not None if answer_is_present else None,
            "citation_valid": (int(answer.get("citation_count") or 0) > 0) if answer_is_present and answer.get("status") == "answered" else (True if answer_is_present and answer.get("status") == "refused" and int(answer.get("citation_count") or 0) == 0 else None),
            "citation_count": answer.get("citation_count") if answer_is_present else None,
            "invalid_citation_count": None,
            "citations": [],
            "note": "Current frozen Showcase envelope preserves citation count but not parsed Citation objects.",
        },
        "timings": {
            "stage_state": "partial" if False else "completed",
            "timing_unit": TIMING_UNIT,
            "timing_source": "showcase_scenario_wall_clock",
            "total_ms": scenario.get("latency_ms"),
            "query_preparation_ms": None,
            "embedding_ms": None,
            "vector_search_ms": None,
            "lexical_search_ms": None,
            "structure_expansion_ms": None,
            "graph_recovery_ms": None,
            "reranking_ms": None,
            "evidence_composition_ms": None,
            "generation_ms": None,
            "validation_ms": None,
            "stage_timings_available": False,
            "historical_benchmark_values_mixed_into_live_trace": False,
        },
        "outcome": {
            "status": lifecycle,
            "answer_status": answer.get("status") if answer_is_present else None,
            "refused": lifecycle == "refused",
            "failure_stage": None,
            "final_candidate_count": retrieval.get("result_count"),
            "final_evidence_count": int(evidence.get("selected_evidence_count") or len(evidence_items)),
            "grounding_passed": answer.get("grounding_valid") if answer_is_present else None,
            "citation_valid": None,
            "final_decision": guard.get("final_decision"),
            "refusal_reason_code": answer.get("refusal_reason_code") or guard.get("refusal_reason_code"),
        },
    }
    trace["trace"]["unavailable_fields"] = _unavailable_fields(trace)
    trace["trace"]["trace_semantic_digest"] = trace_semantic_digest(trace)
    return trace


def trace_semantic_digest(trace: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(trace, ensure_ascii=False))
    trace_meta = payload.get("trace") or {}
    trace_meta.pop("trace_semantic_digest", None)
    return semantic_digest(payload)


def validate_runtime_trace(trace: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["top_level_sections_complete"] = all(section in trace for section in TOP_LEVEL_SECTIONS)
    meta = _copy_dict(trace.get("trace"))
    checks["schema_version_valid"] = meta.get("trace_schema_version") == TRACE_SCHEMA_VERSION
    checks["trace_identity_present"] = bool(meta.get("trace_id"))
    checks["lifecycle_state_valid"] = meta.get("status") in TRACE_LIFECYCLE_STATES
    stage_sections = [section for section in TOP_LEVEL_SECTIONS if section not in {"trace", "outcome"}]
    checks["stage_states_valid"] = all(_copy_dict(trace.get(section)).get("stage_state") in STAGE_STATES for section in stage_sections)
    checks["guard_hidden_chain_of_thought_exposed_false"] = _copy_dict(trace.get("guard")).get("hidden_chain_of_thought_exposed") is False
    candidates = _copy_list(_copy_dict(trace.get("retrieval")).get("candidates"))
    evidence_items = _copy_list(_copy_dict(trace.get("evidence")).get("evidence_items"))
    candidate_ids = {str(row.get("candidate_id")) for row in candidates if isinstance(row, Mapping) and row.get("candidate_id")}
    evidence_links = [str(row.get("source_candidate_id")) for row in evidence_items if isinstance(row, Mapping) and row.get("source_candidate_id")]
    checks["candidate_identity_stable"] = all(isinstance(row, Mapping) and row.get("candidate_id") == row.get("chunk_id") for row in candidates)
    checks["candidate_evidence_semantic_separation"] = _copy_dict(trace.get("evidence")).get("candidate_evidence_semantic_separation") is True
    checks["evidence_candidate_identity_linkage_valid"] = all(link in candidate_ids for link in evidence_links)
    graph = _copy_dict(trace.get("graph_recovery"))
    checks["graph_hop_depth_bounded"] = graph.get("graph_activated") is not True or int(graph.get("hop_depth") or 0) == 1
    checks["graph_traversed_edges_explicit"] = graph.get("graph_activated") is not True or bool(graph.get("traversed_edges"))
    guard = _copy_dict(trace.get("guard"))
    checks["recovery_budget_bounded"] = int(guard.get("recovery_attempt_count") or 0) <= int(guard.get("maximum_recovery_attempt_count") or 1) == 1
    timings = _copy_dict(trace.get("timings"))
    checks["timing_unit_milliseconds"] = timings.get("timing_unit") == TIMING_UNIT
    checks["no_benchmark_timing_mixing"] = timings.get("historical_benchmark_values_mixed_into_live_trace") is False
    checks["semantic_digest_valid"] = meta.get("trace_semantic_digest") == trace_semantic_digest(trace)
    checks["forbidden_trace_keys_absent"] = not scan_forbidden_keys(trace)
    checks["sensitive_values_absent"] = not scan_sensitive_values(trace)
    return checks


def scan_forbidden_keys(payload: object, *, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key).lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if key_text in FORBIDDEN_TRACE_KEYS:
                findings.append(path)
            findings.extend(scan_forbidden_keys(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            findings.extend(scan_forbidden_keys(value, prefix=f"{prefix}[{index}]"))
    return findings


def scan_sensitive_values(payload: object, *, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(scan_sensitive_values(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            findings.extend(scan_sensitive_values(value, prefix=f"{prefix}[{index}]"))
    elif isinstance(payload, str):
        lowered = payload.lower()
        if any(marker in lowered for marker in SENSITIVE_VALUE_MARKERS):
            findings.append(prefix or "<root>")
    return findings


def runtime_trace_contract_document() -> dict[str, Any]:
    return {
        "contract_version": TRACE_CONTRACT_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "runtime_authority": "opk_rag_core",
        "ui_decision_authority": False,
        "real_runtime_data_required": True,
        "fake_runtime_state_allowed": False,
        "hidden_chain_of_thought_exposed": False,
        "trace_lifecycle_states": list(TRACE_LIFECYCLE_STATES),
        "stage_states": list(STAGE_STATES),
        "timing_unit": TIMING_UNIT,
        "top_level_sections": list(TOP_LEVEL_SECTIONS),
        "identity": {
            "trace_id": "one concrete execution identity",
            "query_id": "logical query/scenario identity; distinct from trace_id",
            "frozen_example_identity_origin": "content digest of the already-captured execution authority",
        },
        "null_missing_zero_semantics": {
            "null": "authoritative value unavailable or not applicable according to field definition",
            "missing": "schema violation unless the field is explicitly optional in a future compatible extension",
            "zero": "real observed or measured zero",
        },
        "candidate_evidence_rule": "Candidate != Evidence; Evidence links to Candidate by source_candidate_id.",
        "candidate_provenance_kinds": [SOURCE_KIND_INITIAL, SOURCE_KIND_STRUCTURE, SOURCE_KIND_GRAPH],
        "timing_semantics": {
            "live_trace_only": True,
            "historical_benchmark_values_must_not_be_mixed": True,
            "unavailable_stage_timing": None,
            "zero_only_when_measured_zero": True,
            "retrieval_observer_wall_clock": "time.perf_counter_ns with CUDA synchronization when observer is enabled",
            "showcase_example_total_ms": "ShowcaseRunner wall-clock around one real scenario execution",
        },
        "error_contract": {
            "allowed_fields": ["error_code", "error_stage", "error_type", "safe_message"],
            "raw_traceback_allowed": False,
            "secret_bearing_provider_response_allowed": False,
        },
        "sensitive_data_policy": {
            "forbidden": ["API keys", "passwords", "tokens", "Authorization headers", "private DSNs", "raw .env contents"],
            "document_preview_policy": "bounded/public-safe only; frozen V1 examples omit text_preview when absent from source authority",
        },
        "compatibility": {
            "add_optional_field": "backward_compatible",
            "add_enum_value": "consumer_safe_handling_required",
            "change_field_meaning": "breaking",
            "rename_required_field": "breaking",
            "change_candidate_or_evidence_identity_semantics": "breaking",
            "change_timing_unit": "breaking",
        },
        "breaking_changes_require_schema_version_change": True,
        "runtime_instrumentation_complete": False,
        "showcase_api_implemented": False,
        "showcase_ui_implemented": False,
    }
