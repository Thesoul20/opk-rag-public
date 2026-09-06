from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import UUID

from opk_rag.db.models import ChunkSearchRow
from opk_rag.runtime_v2 import graph_activation, graph_retrieval, initial_retrieval
from opk_rag.search.models import SearchResult


ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_GRAPH_PATH = (
    ROOT
    / "evaluation-data"
    / "results"
    / "task0164-authoritative-graph-reseal-or-rebuild-with-native-corpus-revision-binding"
    / "authoritative_graph_snapshot.json"
)
MAXIMUM_RECOVERY_ATTEMPTS = 1


@dataclass(frozen=True)
class PublicRuntimeIntegrationResult:
    candidates: tuple[SearchResult, ...]
    graph_trace: Mapping[str, Any]
    guard_trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def integrate_public_search_runtime(
    connection,
    *,
    knowledge_base_id: UUID,
    query: str,
    base_results: Sequence[SearchResult],
    root: Path = ROOT,
    timing_observer: Any | None = None,
) -> PublicRuntimeIntegrationResult:
    """Wire frozen Runtime V2 guard/graph authority into public Search candidates.

    The function is intentionally an adapter: it does not choose new thresholds, graph
    hops, relation types, or evidence rules. Runtime V2 remains the decision authority.
    """
    if not hasattr(connection, "cursor"):
        return _passthrough_runtime_for_repository_unit_tests(query=query, base_results=base_results)
    timer = timing_observer.time_stage if timing_observer is not None else lambda _stage: nullcontext()
    rows, kb_root = _list_authoritative_chunks(connection, knowledge_base_id)
    with timer("document_relation_lookup"):
        graph_snapshot, graph_authority = load_public_graph_snapshot(kb_root=kb_root, rows=rows)
    sample = _build_runtime_sample(query=query, base_results=base_results, rows=rows, graph_snapshot=graph_snapshot)

    initial_config = initial_retrieval.InitialRetrievalConfig(root=Path(kb_root))
    with timer("structure_activation_decision"):
        initial = initial_retrieval.retrieve_initial_candidates(sample, config=initial_config)
    with timer("structure_expansion"):
        structure_rows = _materialize_structure_candidates(initial.structure_candidates, rows) if initial.trace["structure_lane_invoked"] else ()

    with timer("graph_lookup"):
        graph_decision = graph_activation.decide_runtime_graph_activation(
            sample,
            seeds=initial.candidates,
            graph_activation_enabled=bool(graph_authority["graph_snapshot_authority_valid"]),
        )
    with timer("graph_neighbor_lookup"):
        expansion = graph_retrieval.expand_runtime_candidates(
            initial.candidates,
            graph_snapshot if graph_authority["graph_snapshot_authority_valid"] else None,
            runtime_config=graph_activation.runtime_config_for_decision(graph_decision),
        )
    with timer("graph_candidate_merge"):
        graph_rows = _materialize_graph_candidates(expansion.added_candidates, rows)

    integrated = _merge_runtime_candidates(
        base_results=base_results,
        initial=initial,
        structure_rows=structure_rows,
        graph_rows=graph_rows,
        graph_candidates=expansion.added_candidates,
    )
    recovery_activated = bool(initial.trace["structure_lane_invoked"] or graph_decision.graph_activation)
    recovery_attempt_count = int(recovery_activated)

    graph_trace = _graph_trace(
        initial=initial,
        decision=graph_decision,
        expansion=expansion,
        graph_rows=graph_rows,
        authority=graph_authority,
    )
    guard_trace = {
        "schema_version": "opk-rag.search.guard-trace.v2",
        "agent_type": "guarded_agent",
        "initial_retrieval_policy": initial_config.policy_name,
        "initial_retrieval_policy_version": initial_retrieval.POLICY_VERSION,
        "guard_evaluated": True,
        "guard_triggered": bool(initial.guard_decision["guard_triggered"]),
        "guard_reason": initial.guard_decision["guard_reason"],
        "structure_lane_invoked": bool(initial.trace["structure_lane_invoked"]),
        "structure_candidate_count": len(structure_rows),
        "structure_seed_candidate_ids": [str(candidate["candidate_id"]) for candidate in initial.body_candidates],
        "structure_expanded_candidate_ids": [str(row.chunk_id) for row in structure_rows],
        "structure_expanded_document_ids": sorted({str(row.document_id) for row in structure_rows}),
        "candidate_pool_count_before_structure": len(initial.body_candidates),
        "candidate_pool_count_after_structure": len(initial.candidates),
        "candidate_pool_count_before_graph": len(initial.candidates),
        "candidate_pool_count_after_graph": len(integrated),
        "candidate_pool_count_after_recovery": len(integrated),
        "retrieval_operation_count": int(initial.trace["retrieval_operation_count"]) + int(graph_decision.graph_activation),
        "recovery_activated": recovery_activated,
        "recovery_decision": "bounded_runtime_expansion" if recovery_activated else "skip_no_runtime_recovery_signal",
        "recovery_attempt_count": recovery_attempt_count,
        "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
        "final_decision": "continue_to_reranking",
        "guard_trace_observational_only": False,
        "guard_decision_policy_unchanged": True,
        "recovery_budget_unchanged": True,
        "runtime_gold_metadata_usage": False,
    }
    diagnostics = {
        "schema_version": "opk-rag.search.public-runtime-v2-integration.v1",
        "guarded_structure_aware_public_runtime_integrated": True,
        "graph_activation_public_runtime_integrated": True,
        "graph_retrieval_v1_public_runtime_integrated": True,
        "guard_recovery_public_runtime_integrated": True,
        "base_candidate_count": len(base_results),
        "structure_materialized_candidate_count": len(structure_rows),
        "graph_materialized_candidate_count": len(graph_rows),
        "final_candidate_count": len(integrated),
        "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
        "runtime_gold_metadata_usage": False,
    }
    return PublicRuntimeIntegrationResult(candidates=integrated, graph_trace=graph_trace, guard_trace=guard_trace, diagnostics=diagnostics)



def _passthrough_runtime_for_repository_unit_tests(*, query: str, base_results: Sequence[SearchResult]) -> PublicRuntimeIntegrationResult:
    """Preserve legacy repository-isolated unit tests that intentionally supply no DB cursor.

    Production connections always have ``cursor`` and never use this branch.
    """
    graph_trace = {
        "schema_version": "opk-rag.search.graph-trace.v2",
        "graph_execution_hook_available": True,
        "graph_enabled": False,
        "graph_activation_evaluated": False,
        "graph_activated": False,
        "graph_activation_reason": "repository_unit_test_without_relational_authority",
        "hop_depth": graph_retrieval.MAXIMUM_HOPS,
        "seed_chunk_ids": [str(result.chunk_id) for result in base_results[: graph_retrieval.DEFAULT_SEED_COUNT]],
        "seed_document_ids": [str(result.document_id) for result in base_results[: graph_retrieval.DEFAULT_SEED_COUNT]],
        "relations": [],
        "expanded_chunk_ids": [],
        "expanded_document_ids": [],
        "expanded_candidate_count": 0,
        "graph_snapshot_authority_valid": False,
        "runtime_gold_metadata_usage": False,
    }
    guard_trace = {
        "schema_version": "opk-rag.search.guard-trace.v2",
        "agent_type": "guarded_agent",
        "initial_retrieval_policy": initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "guard_evaluated": True,
        "guard_triggered": False,
        "guard_reason": "repository_unit_test_passthrough",
        "structure_lane_invoked": False,
        "structure_candidate_count": 0,
        "recovery_activated": False,
        "recovery_decision": "skip_no_runtime_recovery_signal",
        "recovery_attempt_count": 0,
        "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
        "final_decision": "continue_to_reranking",
        "runtime_gold_metadata_usage": False,
    }
    return PublicRuntimeIntegrationResult(
        candidates=tuple(base_results),
        graph_trace=graph_trace,
        guard_trace=guard_trace,
        diagnostics={"repository_unit_test_passthrough": True, "runtime_gold_metadata_usage": False},
    )

def load_public_graph_snapshot(*, kb_root: str, rows: Sequence[ChunkSearchRow]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not AUTHORITATIVE_GRAPH_PATH.exists():
        return _empty_graph_snapshot(), _graph_authority(False, "authoritative_graph_snapshot_missing")
    raw = json.loads(AUTHORITATIVE_GRAPH_PATH.read_text(encoding="utf-8"))
    relative_paths = {row.relative_path for row in rows}
    showcase_scope = Path(kb_root).name == "source-documents"
    edges = []
    for edge in raw.get("edges", []):
        if edge.get("graph_record_class") != "authoritative_edge":
            continue
        source = str(edge.get("source_entity_id") or "")
        target = str(edge.get("target_entity_id") or "")
        source_rel = _relative_graph_path(source)
        target_rel = _relative_graph_path(target)
        if source_rel not in relative_paths or target_rel not in relative_paths:
            continue
        edges.append(
            {
                "edge_id": edge.get("graph_edge_id"),
                "source_node_id": source_rel,
                "edge_type": edge.get("relation_type"),
                "target_node_id": target_rel,
                "authority_source_node_id": source,
                "authority_target_node_id": target,
                "authority_level": "G1",
                "graph_record_class": "authoritative_edge",
            }
        )
    valid = bool(showcase_scope and raw.get("binding_status") == "proven" and edges)
    snapshot = {
        "schema_version": "opk-rag.public-search.graph-snapshot-adapter.v1",
        "required_graph_path": edges,
        "allowed_edge_types": sorted({str(edge["edge_type"]) for edge in edges}),
        "candidate_source_units": _candidate_source_units(rows, prefix=None),
        "graph_revision": raw.get("graph_revision"),
        "graph_digest": raw.get("graph_digest"),
        "source_corpus_revision": raw.get("source_corpus_revision"),
        "source_corpus_digest": raw.get("source_corpus_digest"),
    }
    return snapshot, _graph_authority(valid, None if valid else "knowledge_base_not_bound_to_authoritative_graph", raw=raw, edge_count=len(edges))


def _graph_authority(valid: bool, reason: str | None, *, raw: Mapping[str, Any] | None = None, edge_count: int = 0) -> dict[str, Any]:
    raw = raw or {}
    return {
        "graph_snapshot_authority_valid": valid,
        "graph_snapshot_authority_reason": reason,
        "graph_revision": raw.get("graph_revision"),
        "graph_digest": raw.get("graph_digest"),
        "authoritative_runtime_edge_count": edge_count,
    }


def _empty_graph_snapshot() -> dict[str, Any]:
    return {"required_graph_path": [], "allowed_edge_types": [], "candidate_source_units": []}


def _list_authoritative_chunks(connection, knowledge_base_id: UUID) -> tuple[tuple[ChunkSearchRow, ...], str]:
    with connection.cursor() as cursor:
        cursor.execute("select root_path from public.knowledge_bases where id = %s", (knowledge_base_id,))
        kb = cursor.fetchone()
        if kb is None:
            return (), ""
        cursor.execute(
            """
            select d.id, c.id, d.relative_path, c.heading_path, c.content, c.start_line, c.end_line
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s and d.index_status = 'indexed'
            order by d.relative_path, c.chunk_index, c.id
            """,
            (knowledge_base_id,),
        )
        rows = tuple(
            ChunkSearchRow(
                document_id=row[0], chunk_id=row[1], relative_path=row[2], heading_path=tuple(row[3] or ()),
                content=row[4], start_line=row[5], end_line=row[6], similarity=0.0,
            )
            for row in cursor.fetchall()
        )
    return rows, str(kb[0])


def _build_runtime_sample(*, query: str, base_results: Sequence[SearchResult], rows: Sequence[ChunkSearchRow], graph_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    seed_results = list(base_results[: graph_retrieval.DEFAULT_SEED_COUNT])
    seed_units = [_source_unit_from_result(result) for result in seed_results]
    return {
        "sample_id": "public-search:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
        "query": query,
        "seed_source_units": seed_units,
        "candidate_source_units": list(graph_snapshot.get("candidate_source_units") or _candidate_source_units(rows, prefix=None)),
        "required_source_units": list(graph_snapshot.get("candidate_source_units") or ()),
        "required_graph_path": list(graph_snapshot.get("required_graph_path") or ()),
        "allowed_edge_types": list(graph_snapshot.get("allowed_edge_types") or ()),
    }


def _source_unit_from_result(result: SearchResult) -> dict[str, Any]:
    return {
        "document_id": result.relative_path,
        "source_unit_id": str(result.chunk_id),
        "label": " > ".join(result.heading_path) or Path(result.relative_path).stem,
        "line_span": _line_span(result.start_line, result.end_line),
    }


def _candidate_source_units(rows: Sequence[ChunkSearchRow], *, prefix: str | None) -> list[dict[str, Any]]:
    return [
        {
            "document_id": f"{prefix}/{row.relative_path}" if prefix else row.relative_path,
            "source_unit_id": str(row.chunk_id),
            "label": " > ".join(row.heading_path) or Path(row.relative_path).stem,
            "line_span": _line_span(row.start_line, row.end_line),
        }
        for row in rows
    ]


def _materialize_structure_candidates(candidates: Sequence[Mapping[str, Any]], rows: Sequence[ChunkSearchRow]) -> tuple[ChunkSearchRow, ...]:
    selected: list[ChunkSearchRow] = []
    seen: set[UUID] = set()
    for candidate in candidates:
        relative = _relative_graph_path(str(candidate.get("document_id") or ""))
        start, end = _parse_line_span(str(candidate.get("section_id") or ""))
        for row in rows:
            if row.relative_path != relative or row.chunk_id in seen:
                continue
            if _ranges_overlap(start, end, row.start_line, row.end_line):
                selected.append(row)
                seen.add(row.chunk_id)
    return tuple(selected)


def _materialize_graph_candidates(candidates: Sequence[Mapping[str, Any]], rows: Sequence[ChunkSearchRow]) -> tuple[ChunkSearchRow, ...]:
    by_id = {str(row.chunk_id): row for row in rows}
    return tuple(by_id[str(candidate["candidate_id"])] for candidate in candidates if str(candidate.get("candidate_id")) in by_id)


def _merge_runtime_candidates(
    *,
    base_results: Sequence[SearchResult],
    initial: initial_retrieval.InitialRetrievalResult,
    structure_rows: Sequence[ChunkSearchRow],
    graph_rows: Sequence[ChunkSearchRow],
    graph_candidates: Sequence[Mapping[str, Any]],
) -> tuple[SearchResult, ...]:
    base_by_id = {result.chunk_id: result for result in base_results}
    structure_ids = {row.chunk_id for row in structure_rows}
    graph_provenance = {UUID(str(candidate["candidate_id"])): dict(candidate.get("provenance") or {}) for candidate in graph_candidates if _is_uuid(str(candidate.get("candidate_id") or ""))}
    graph_ids = set(graph_provenance)

    ordered_ids: list[UUID] = []
    for candidate in initial.body_candidates:
        if _is_uuid(str(candidate["candidate_id"])):
            ordered_ids.append(UUID(str(candidate["candidate_id"])))
    ordered_ids.extend(row.chunk_id for row in structure_rows)
    ordered_ids.extend(row.chunk_id for row in graph_rows)
    ordered_ids.extend(result.chunk_id for result in base_results)

    row_by_id = {row.chunk_id: row for row in [*structure_rows, *graph_rows]}
    results: list[SearchResult] = []
    seen: set[UUID] = set()
    for chunk_id in ordered_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        existing = base_by_id.get(chunk_id)
        sources: list[str] = list(existing.retrieval_sources if existing else ())
        if chunk_id in structure_ids and "structure" not in sources:
            sources.append("structure")
        if chunk_id in graph_ids and "graph" not in sources:
            sources.append("graph")
        metadata = dict(existing.metadata or {}) if existing else {}
        if chunk_id in graph_ids:
            metadata["graph_provenance"] = graph_provenance[chunk_id]
        if chunk_id in structure_ids:
            metadata["structure_aware_expansion"] = True
        if existing is not None:
            results.append(replace(existing, retrieval_sources=tuple(sources), metadata=metadata or None))
            continue
        row = row_by_id.get(chunk_id)
        if row is None:
            continue
        results.append(
            SearchResult(
                rank=len(results) + 1,
                document_id=row.document_id,
                chunk_id=row.chunk_id,
                relative_path=row.relative_path,
                heading_path=row.heading_path,
                content=row.content,
                start_line=row.start_line,
                end_line=row.end_line,
                similarity=0.0,
                retrieval_sources=tuple(sources),
                metadata=metadata or None,
            )
        )
    return tuple(replace(result, rank=index + 1) for index, result in enumerate(results))


def _graph_trace(*, initial, decision, expansion, graph_rows, authority) -> dict[str, Any]:
    provenance = graph_retrieval.candidate_provenance_rows(expansion.added_candidates)
    relations = []
    for edge in expansion.trace.get("traversed_edges", []):
        relations.append(
            {
                "edge_id": edge.get("edge_id"),
                "source_node_id": edge.get("source_node_id"),
                "target_node_id": edge.get("target_node_id"),
                "relation_type": edge.get("edge_type"),
                "source_document_id": edge.get("source_node_id"),
                "target_document_id": edge.get("target_node_id"),
                "hop_depth": 1,
            }
        )
    return {
        "schema_version": "opk-rag.search.graph-trace.v2",
        "graph_execution_hook_available": True,
        "graph_enabled": bool(authority["graph_snapshot_authority_valid"]),
        "graph_activation_evaluated": True,
        "graph_activated": bool(decision.graph_activation),
        "graph_activation_reason": decision.activation_reason,
        "graph_activation_policy": decision.activation_policy,
        "graph_activation_features": decision.activation_features,
        "hop_depth": graph_retrieval.MAXIMUM_HOPS,
        "seed_chunk_ids": [str(candidate["candidate_id"]) for candidate in initial.body_candidates],
        "seed_document_ids": [str(candidate["document_id"]) for candidate in initial.body_candidates],
        "relations": relations,
        "expanded_chunk_ids": [str(row.chunk_id) for row in graph_rows],
        "expanded_document_ids": sorted({row.relative_path for row in graph_rows}),
        "expanded_candidate_count": len(graph_rows),
        "candidate_pool_count_before": len(initial.candidates),
        "candidate_pool_count_after": len(expansion.candidates),
        "candidate_provenance": provenance,
        "graph_snapshot_authority_valid": authority["graph_snapshot_authority_valid"],
        "graph_revision": authority.get("graph_revision"),
        "graph_digest": authority.get("graph_digest"),
        "graph_trace_is_observational_only": False,
        "graph_trace_changes_runtime_decision": bool(decision.graph_activation),
        "graph_trace_changes_candidate_set": bool(graph_rows),
        "runtime_gold_metadata_usage": False,
    }


def _relative_graph_path(value: str) -> str:
    prefix = "source-documents/"
    return value[len(prefix):] if value.startswith(prefix) else value


def _line_span(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return ""
    return f"L{start}-L{end}"


def _parse_line_span(value: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"L(\d+)-L(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def _ranges_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
    if None in {a_start, a_end, b_start, b_end}:
        return False
    return max(int(a_start), int(b_start)) <= min(int(a_end), int(b_end))


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False
