from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "opk-rag.showcase.graph-visualization.v1"
AUTHORITATIVE_SCENARIO_ID = "S03"
HISTORICAL_GRAPH_QUERY = "academic-docx-polisher 的技术路线、最小测试和 Skill 化路线之间是什么关系？"


class GraphVisualizationError(RuntimeError):
    pass


def _copy_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _sources(row: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in row.get("retrieval_sources") or []]


def _candidate_key(row: Mapping[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("canonical_chunk_id") or row.get("candidate_id") or "")


def _dedupe_rows(rows: Sequence[Mapping[str, Any]], *, key_fields: Sequence[str]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def validate_s03_trace(scenario: Mapping[str, Any]) -> dict[str, bool]:
    graph = _copy_dict(scenario.get("graph"))
    runtime = _copy_dict(scenario.get("runtime"))
    validation = _copy_dict(scenario.get("showcase_validation"))
    checks = {
        "scenario_id_valid": scenario.get("scenario_id") == AUTHORITATIVE_SCENARIO_ID,
        "execution_status_passed": scenario.get("execution_status") == "passed",
        "showcase_validation_valid": validation.get("valid") is True,
        "graph_activation_evaluated": graph.get("graph_activation_evaluated") is True,
        "graph_activated": graph.get("graph_activated") is True,
        "graph_hop_depth_one": int(graph.get("hop_depth") or runtime.get("graph_hop_depth") or 0) == 1,
        "expanded_candidate_present": int(graph.get("expanded_candidate_count") or 0) >= 1,
        "graph_candidate_provenance_available": bool(graph.get("candidate_provenance")),
        "runtime_gold_metadata_usage_false": graph.get("runtime_gold_metadata_usage") is False,
        "historical_graph_query_not_promoted": str(scenario.get("query") or "") != HISTORICAL_GRAPH_QUERY,
    }
    return checks


def require_valid_s03_trace(scenario: Mapping[str, Any]) -> None:
    checks = validate_s03_trace(scenario)
    failures = [key for key, ok in checks.items() if not ok]
    if failures:
        raise GraphVisualizationError("invalid_s03_trace:" + ",".join(failures))


def _actual_traversed_relations(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provenance in graph.get("candidate_provenance") or []:
        for edge in provenance.get("graph_path") or []:
            rows.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "relation_type": edge.get("edge_type"),
                    "source_document_id": edge.get("source_node_id"),
                    "target_document_id": edge.get("target_node_id"),
                    "authority_level": edge.get("authority_level"),
                    "graph_record_class": edge.get("graph_record_class"),
                    "hop_depth": int(provenance.get("graph_hop_count") or 1),
                }
            )
    return _dedupe_rows(rows, key_fields=("edge_id", "source_document_id", "target_document_id"))


def _seed_candidates(graph: Mapping[str, Any], top_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_chunk = {_candidate_key(row): row for row in top_candidates if _candidate_key(row)}
    chunk_ids = [str(item) for item in graph.get("seed_chunk_ids") or []]
    document_ids = [str(item) for item in graph.get("seed_document_ids") or []]
    rows: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(chunk_ids):
        source = by_chunk.get(chunk_id, {})
        rows.append(
            {
                "seed_index": index + 1,
                "chunk_id": chunk_id,
                "document_id": document_ids[index] if index < len(document_ids) else source.get("relative_path"),
                "final_rank": source.get("rank"),
                "retrieval_sources": _sources(source),
            }
        )
    return rows


def _expanded_candidates(
    graph: Mapping[str, Any],
    top_candidates: Sequence[Mapping[str, Any]],
    selected_evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    top_by_chunk = {_candidate_key(row): row for row in top_candidates if _candidate_key(row)}
    evidence_by_chunk = {_candidate_key(row): row for row in selected_evidence if _candidate_key(row)}
    rows: list[dict[str, Any]] = []
    for provenance in graph.get("candidate_provenance") or []:
        chunk_id = _candidate_key(provenance)
        top = top_by_chunk.get(chunk_id, {})
        evidence = evidence_by_chunk.get(chunk_id, {})
        rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": provenance.get("document_id") or top.get("relative_path"),
                "section_id": provenance.get("section_id"),
                "source_seed_candidate_id": provenance.get("source_seed_candidate_id"),
                "graph_added": provenance.get("graph_added") is True,
                "graph_hop_count": int(provenance.get("graph_hop_count") or 0),
                "graph_edge_types": list(provenance.get("graph_edge_types") or provenance.get("edge_types") or []),
                "graph_expansion_reason": provenance.get("graph_expansion_reason"),
                "final_rank": top.get("rank"),
                "retrieval_sources": _sources(top),
                "selected_for_context": top.get("selected_for_context") is True,
                "selected_as_evidence": bool(evidence),
                "evidence_lines": None if not evidence else [evidence.get("start_line"), evidence.get("end_line")],
            }
        )
    return rows


def normalize_s03_trace(
    scenario: Mapping[str, Any],
    *,
    source_trace: str,
) -> dict[str, Any]:
    require_valid_s03_trace(scenario)
    runtime = _copy_dict(scenario.get("runtime"))
    retrieval = _copy_dict(scenario.get("retrieval"))
    guard = _copy_dict(scenario.get("guard"))
    graph = _copy_dict(scenario.get("graph"))
    evidence = _copy_dict(scenario.get("evidence"))
    top_candidates = [dict(row) for row in scenario.get("top_candidates") or []]
    selected_evidence = [dict(row) for row in evidence.get("selected_evidence") or []]

    initial_candidates = [
        {
            "rank": row.get("rank"),
            "chunk_id": row.get("chunk_id"),
            "relative_path": row.get("relative_path"),
            "retrieval_sources": _sources(row),
            "selected_for_context": row.get("selected_for_context") is True,
        }
        for row in top_candidates
        if "graph" not in _sources(row)
    ]
    traversed_relations = _actual_traversed_relations(graph)
    expanded = _expanded_candidates(graph, top_candidates, selected_evidence)

    model: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": AUTHORITATIVE_SCENARIO_ID,
        "scenario_name": scenario.get("scenario_name"),
        "query_set_digest": scenario.get("query_set_digest"),
        "query": scenario.get("query"),
        "source_trace": source_trace,
        "runtime": {
            "vector_backend": runtime.get("vector_backend"),
            "retrieval_policy": runtime.get("retrieval_policy"),
            "embedding_model": runtime.get("embedding_model"),
            "reranker": runtime.get("reranker"),
            "graph_hop_depth": int(graph.get("hop_depth") or runtime.get("graph_hop_depth") or 0),
        },
        "activation": {
            "graph_activation_evaluated": graph.get("graph_activation_evaluated") is True,
            "graph_activated": graph.get("graph_activated") is True,
            "graph_activation_policy": graph.get("graph_activation_policy"),
            "graph_activation_reason": graph.get("graph_activation_reason"),
            "relation_intent": (graph.get("graph_activation_features") or {}).get("relation_intent"),
            "relation_terms": list((graph.get("graph_activation_features") or {}).get("relation_terms") or []),
            "recovery_activated": guard.get("recovery_activated") is True,
            "recovery_attempt_count": int(guard.get("recovery_attempt_count") or 0),
            "maximum_recovery_attempt_count": int(guard.get("maximum_recovery_attempt_count") or 0),
            "recovery_decision": guard.get("recovery_decision"),
        },
        "retrieval": {
            "mode": retrieval.get("retrieval_mode"),
            "candidate_count": int(retrieval.get("candidate_count") or 0),
            "result_count": int(retrieval.get("result_count") or 0),
            "reranker_active": retrieval.get("reranker_active") is True,
            "reranker_model_id": retrieval.get("reranker_model_id"),
        },
        "initial_candidates": initial_candidates,
        "seed_candidates": _seed_candidates(graph, top_candidates),
        "authoritative_relations": traversed_relations,
        "expanded_candidates": expanded,
        "reranked_candidates": [
            {
                "rank": row.get("rank"),
                "chunk_id": row.get("chunk_id"),
                "relative_path": row.get("relative_path"),
                "retrieval_sources": _sources(row),
                "selected_for_context": row.get("selected_for_context") is True,
            }
            for row in top_candidates
        ],
        "selected_evidence": [
            {
                "chunk_id": row.get("chunk_id"),
                "relative_path": row.get("relative_path"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "retrieval_sources": _sources(row),
                "graph_enabled_evidence": "graph" in _sources(row),
            }
            for row in selected_evidence
        ],
        "graph_authority": {
            "graph_digest": graph.get("graph_digest"),
            "graph_revision": graph.get("graph_revision"),
            "graph_snapshot_authority_valid": graph.get("graph_snapshot_authority_valid") is True,
        },
        "runtime_gold_metadata_usage": graph.get("runtime_gold_metadata_usage") is True,
        "fabricated_graph_relation_count": 0,
        "fabricated_candidate_count": 0,
        "presentation": {
            "executive_story": [
                "Find an initially relevant document",
                "Conditionally activate the frozen one-hop Graph path",
                "Follow an authoritative relation to recover a related document",
                "Merge the Graph-added candidate into the shared candidate pool",
                "Apply the existing reranker and evidence selection",
            ],
            "technical_story": [
                "initial retrieval",
                "retrieval-aware graph activation",
                "authoritative one-hop relation",
                "graph candidate provenance",
                "shared BGE reranking",
                "final evidence provenance",
            ],
        },
    }
    model["graph_showcase_visualization_v1_digest"] = visualization_digest(model)
    return model


def canonical_visualization_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(model)
    payload.pop("graph_showcase_visualization_v1_digest", None)
    return payload


def visualization_digest(model: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_visualization_payload(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_visualization_digest(model: Mapping[str, Any]) -> bool:
    return str(model.get("graph_showcase_visualization_v1_digest") or "") == visualization_digest(model)


def _label(path: Any) -> str:
    value = str(path or "unknown")
    name = Path(value).name
    return name[:-3] if name.endswith(".md") else name


def _escape(value: Any) -> str:
    return str(value or "").replace('"', "'").replace("\n", " ")


def render_mermaid(model: Mapping[str, Any]) -> str:
    relations = list(model.get("authoritative_relations") or [])
    expanded = list(model.get("expanded_candidates") or [])
    if not relations or not expanded:
        raise GraphVisualizationError("visualization_missing_relation_or_expanded_candidate")
    relation = relations[0]
    graph_candidate = expanded[0]
    initial = list(model.get("initial_candidates") or [])
    evidence = list(model.get("selected_evidence") or [])
    initial_labels = "<br/>".join(f"#{row.get('rank')} {_escape(_label(row.get('relative_path')))}" for row in initial[:3]) or "initial candidates"
    evidence_labels = "<br/>".join(_escape(_label(row.get("relative_path"))) for row in evidence[:4]) or "final evidence"
    query = _escape(model.get("query"))
    source = _escape(_label(relation.get("source_document_id")))
    target = _escape(_label(relation.get("target_document_id")))
    relation_type = _escape(relation.get("relation_type"))
    authority = _escape(relation.get("authority_level"))
    graph_target = _escape(_label(graph_candidate.get("document_id")))
    reranker = _escape((model.get("runtime") or {}).get("reranker"))
    recovery = model.get("activation") or {}
    return "\n".join(
        [
            "flowchart LR",
            f'  Q["S03 Query<br/>{query}"]',
            f'  I["Initial Retrieval<br/>{initial_labels}"]',
            f'  S["Seed Document<br/>{source}"]',
            f'  A["Graph Activation<br/>{_escape(recovery.get("graph_activation_reason"))}<br/>Recovery {recovery.get("recovery_attempt_count")}/{recovery.get("maximum_recovery_attempt_count")}"]',
            f'  R["Authoritative Edge<br/>{relation_type} · {authority}<br/>{source} → {target}"]',
            f'  G["Graph-added Candidate<br/>{graph_target}<br/>hop=1 · provenance verified"]',
            '  P["Shared Candidate Pool<br/>initial + graph-added"]',
            f'  RR["Reranking<br/>{reranker}"]',
            f'  E["Final Evidence<br/>{evidence_labels}"]',
            "  Q --> I",
            "  I --> S",
            "  S --> A",
            "  A -->|activated| R",
            "  R --> G",
            "  I --> P",
            "  G --> P",
            "  P --> RR",
            "  RR --> E",
            "  classDef graphAdded stroke-width:3px;",
            "  classDef authority stroke-dasharray: 5 3;",
            "  class G graphAdded;",
            "  class R authority;",
        ]
    ) + "\n"


def render_executive_mermaid(model: Mapping[str, Any]) -> str:
    relations = list(model.get("authoritative_relations") or [])
    expanded = list(model.get("expanded_candidates") or [])
    if not relations or not expanded:
        raise GraphVisualizationError("visualization_missing_relation_or_expanded_candidate")
    relation = relations[0]
    candidate = expanded[0]
    return "\n".join(
        [
            "flowchart LR",
            f'  Q["Question<br/>{_escape(model.get("query"))}"]',
            f'  F["Find initial document<br/>{_escape(_label(relation.get("source_document_id")))}"]',
            f'  L["Follow known relationship<br/>{_escape(relation.get("relation_type"))} · one hop"]',
            f'  R["Recover related document<br/>{_escape(_label(candidate.get("document_id")))}"]',
            '  U["Use shared ranking + evidence pipeline"]',
            "  Q --> F --> L --> R --> U",
        ]
    ) + "\n"


def render_documentation(model: Mapping[str, Any]) -> str:
    relations = list(model.get("authoritative_relations") or [])
    expanded = list(model.get("expanded_candidates") or [])
    relation = relations[0] if relations else {}
    candidate = expanded[0] if expanded else {}
    activation = model.get("activation") or {}
    evidence = model.get("selected_evidence") or []
    mermaid = render_mermaid(model)
    return f"""# Showcase Graph Retrieval Visualization — S03

## What this view proves

S03 asks:

> {model.get('query')}

The live production trace activates the frozen one-hop Graph Retrieval V1 path. The visualization is generated from that trace rather than from a manually authored knowledge-graph story.

## Executive view

The system first finds relevant Tauri material through normal retrieval. Because the query asks about a relationship and the retrieved seed has authoritative graph connectivity, Graph Retrieval activates once. It follows an authoritative `{relation.get('relation_type')}` edge from `{relation.get('source_document_id')}` to `{relation.get('target_document_id')}` and recovers `{candidate.get('document_id')}` as a Graph-added candidate.

That candidate does **not** bypass ranking. It joins the shared candidate pool, is processed by `{(model.get('runtime') or {}).get('reranker')}`, and is selected into final evidence.

## Technical view

- Retrieval policy: `{(model.get('runtime') or {}).get('retrieval_policy')}`
- Vector backend: `{(model.get('runtime') or {}).get('vector_backend')}`
- Graph activation: `{activation.get('graph_activated')}`
- Activation reason: `{activation.get('graph_activation_reason')}`
- Recovery budget used: `{activation.get('recovery_attempt_count')} / {activation.get('maximum_recovery_attempt_count')}`
- Graph hop depth: `{(model.get('runtime') or {}).get('graph_hop_depth')}`
- Traversed authoritative relation count: `{len(relations)}`
- Graph-added candidate count: `{len(expanded)}`
- Selected evidence count: `{len(evidence)}`
- Runtime gold metadata usage: `{str(model.get('runtime_gold_metadata_usage')).lower()}`

The key distinction is **initial candidate vs Graph-added candidate**. Initial candidates come from the normal retrieval path. The Graph-added candidate is identified only where live candidate provenance says `graph_added=true` and supplies the actual authoritative edge path.

## Canonical Mermaid

```mermaid
{mermaid.rstrip()}
```

## What this does not claim

This is not unbounded Graph traversal, multi-hop GraphRAG, or an open-ended Planner. The frozen runtime uses retrieval-aware activation, a one-hop graph boundary, and a maximum recovery budget of one. The visualization also does not imply that following an edge automatically makes a chunk Evidence: the recovered candidate still passes through the shared reranking/evidence pipeline.

## Authority

- Scenario: `S03`
- Query-set digest: `{model.get('query_set_digest')}`
- Visualization digest: `{model.get('graph_showcase_visualization_v1_digest')}`
- Graph revision: `{(model.get('graph_authority') or {}).get('graph_revision')}`
- Source trace: `{model.get('source_trace')}`
"""


def render_drawio_handoff(model: Mapping[str, Any]) -> str:
    relation = (model.get("authoritative_relations") or [{}])[0]
    candidate = (model.get("expanded_candidates") or [{}])[0]
    return f"""# Draw.io Handoff — S03 Graph Retrieval Showcase

Create a **16:9 horizontal technical presentation diagram** from the canonical TASK-0220 visualization authority. This is a rendering task only; do not infer or invent nodes, relations, candidates, metrics, or runtime behavior.

## Required flow

`Query → Initial Retrieval → Seed → Graph Activation → Authoritative Edge → Graph-added Candidate → Shared Candidate Pool → BGE Reranker → Final Evidence`

## Authoritative facts

- Query: `{model.get('query')}`
- Graph hop depth: `{(model.get('runtime') or {}).get('graph_hop_depth')}`
- Recovery: `{(model.get('activation') or {}).get('recovery_attempt_count')} / {(model.get('activation') or {}).get('maximum_recovery_attempt_count')}`
- Source seed: `{relation.get('source_document_id')}`
- Relation: `{relation.get('relation_type')}`
- Relation authority: `{relation.get('authority_level')}` / `{relation.get('graph_record_class')}`
- Target: `{relation.get('target_document_id')}`
- Graph-added candidate: `{candidate.get('document_id')}`
- Candidate final rank: `{candidate.get('final_rank')}`
- Candidate selected as evidence: `{candidate.get('selected_as_evidence')}`
- Reranker: `{(model.get('runtime') or {}).get('reranker')}`

## Visual hierarchy

1. Put the query at far left.
2. Show initial candidates as a compact grouped block.
3. Highlight the actual seed document.
4. Show Graph activation as a decision/gate, not as an always-on step.
5. Render the authoritative edge as a clearly labeled single-hop relation.
6. Visually distinguish the Graph-added candidate from initial candidates.
7. Merge both lanes into one shared candidate pool.
8. Place the normal BGE reranker after the merge.
9. End with final evidence.

Use a dark black/ink-green presentation background with restrained mint-green emphasis if styling is needed. Avoid people, cartoons, generic spider-web graphs, or decorative graph nodes unrelated to the live trace.

## Truthfulness constraints

- Do not show multi-hop traversal.
- Do not add additional Graph edges.
- Do not show the historical academic-docx-polisher query.
- Do not imply Graph candidates bypass reranking.
- Do not display secrets or private absolute filesystem paths.
- The Mermaid source is `docs/diagrams/showcase_graph_retrieval_s03.mmd`.
- Canonical visualization digest: `{model.get('graph_showcase_visualization_v1_digest')}`.
"""
