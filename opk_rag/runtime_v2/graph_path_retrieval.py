from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any

from opk_rag.runtime_v2 import graph_retrieval


PATH_RETRIEVAL_POLICY = "bounded_multi_hop_path_retrieval"
POLICY_VERSION = "opk-rag.graph-path-retrieval.bounded-multi-hop.v1"
SOURCE_TASK = "TASK-0141"
DEFAULT_MAXIMUM_HOPS = 2
DEFAULT_MAX_NEIGHBORS_PER_NODE = 3
DEFAULT_MAX_PATHS_PER_QUERY = 8
DEFAULT_MAXIMUM_ADDED_CANDIDATES = 5
DEFAULT_MAXIMUM_EXPANDED_NODES = 20


class GraphPathRetrievalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GraphPathRetrievalPolicy:
    policy_name: str = PATH_RETRIEVAL_POLICY
    policy_version: str = POLICY_VERSION
    source_task: str = SOURCE_TASK
    maximum_hops: int = DEFAULT_MAXIMUM_HOPS
    max_neighbors_per_node: int = DEFAULT_MAX_NEIGHBORS_PER_NODE
    max_paths_per_query: int = DEFAULT_MAX_PATHS_PER_QUERY
    maximum_expanded_nodes: int = DEFAULT_MAXIMUM_EXPANDED_NODES
    maximum_added_candidates: int = DEFAULT_MAXIMUM_ADDED_CANDIDATES
    allowed_authority_levels: tuple[str, ...] = graph_retrieval.ALLOWED_AUTHORITY_LEVELS
    supported_edge_types: tuple[str, ...] = graph_retrieval.SUPPORTED_EDGE_TYPES
    relation_aware: bool = True

    def __post_init__(self) -> None:
        if self.policy_name != PATH_RETRIEVAL_POLICY:
            raise GraphPathRetrievalConfigError(f"unsupported graph path retrieval policy: {self.policy_name}")
        if self.maximum_hops < 1:
            raise GraphPathRetrievalConfigError("maximum_hops must be positive")
        if self.max_neighbors_per_node < 1:
            raise GraphPathRetrievalConfigError("max_neighbors_per_node must be positive")
        if self.max_paths_per_query < 1:
            raise GraphPathRetrievalConfigError("max_paths_per_query must be positive")
        if self.maximum_expanded_nodes < 1:
            raise GraphPathRetrievalConfigError("maximum_expanded_nodes must be positive")
        if self.maximum_added_candidates < 0:
            raise GraphPathRetrievalConfigError("maximum_added_candidates must be non-negative")

    @property
    def policy_digest(self) -> str:
        return stable_digest(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "source_task": self.source_task,
            "maximum_hops": self.maximum_hops,
            "max_neighbors_per_node": self.max_neighbors_per_node,
            "max_paths_per_query": self.max_paths_per_query,
            "maximum_expanded_nodes": self.maximum_expanded_nodes,
            "maximum_added_candidates": self.maximum_added_candidates,
            "allowed_authority_levels": list(self.allowed_authority_levels),
            "supported_edge_types": list(self.supported_edge_types),
            "relation_aware": self.relation_aware,
            "gold_signal_used_by_graph_policy": False,
            "benchmark_specific_graph_rule": False,
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_digest"] = self.policy_digest
        return payload


@dataclass(frozen=True)
class GraphPathRetrievalResult:
    candidates: tuple[dict[str, Any], ...]
    added_candidates: tuple[dict[str, Any], ...]
    paths: tuple[dict[str, Any], ...]
    trace: dict[str, Any]
    diagnostics: dict[str, Any]


def retrieve_path_candidates(
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    graph_snapshot: dict[str, Any],
    policy: GraphPathRetrievalPolicy | None = None,
) -> GraphPathRetrievalResult:
    resolved = policy or GraphPathRetrievalPolicy()
    started = perf_counter()
    graph_retrieval.validate_graph_snapshot(graph_snapshot)
    seed_candidates = list(seeds)
    seed_docs = sorted({candidate["document_id"] for candidate in seed_candidates})
    seed_candidate_ids = [candidate["candidate_id"] for candidate in seed_candidates]
    allowed_edge_types = _allowed_edge_types(graph_snapshot, resolved)
    graph = graph_retrieval.graph_edges(graph_snapshot, allowed_authority_levels=set(resolved.allowed_authority_levels))
    units_by_doc = _candidate_units_by_document(graph_snapshot)

    seen_candidate_ids = set(seed_candidate_ids)
    seen_nodes = set(seed_docs)
    expanded_nodes: set[str] = set()
    added: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    traversed_edges: list[dict[str, Any]] = []
    queue = deque((node, tuple(), frozenset({node})) for node in seed_docs)
    stop_reason = "completed"

    while queue:
        node, path_edges, visited = queue.popleft()
        if len(path_edges) >= resolved.maximum_hops:
            continue
        if len(expanded_nodes) >= resolved.maximum_expanded_nodes:
            stop_reason = "max_expanded_nodes"
            break
        expanded_nodes.add(node)
        neighbors = [edge for edge in graph.get(node, []) if edge["edge_type"] in allowed_edge_types]
        for edge in neighbors[: resolved.max_neighbors_per_node]:
            target = edge["target_node_id"]
            if target in visited:
                continue
            next_edges = (*path_edges, edge)
            path_row = _path_row(seed_docs=seed_docs, edges=list(next_edges), target_node_id=target)
            paths.append(path_row)
            traversed_edges.append(edge)
            if len(paths) >= resolved.max_paths_per_query:
                stop_reason = "max_paths_per_query"
            if target not in seen_nodes:
                seen_nodes.add(target)
                queue.append((target, next_edges, frozenset((*visited, target))))
            for unit in units_by_doc.get(target, []):
                if unit["source_unit_id"] in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(unit["source_unit_id"])
                candidate = graph_retrieval._candidate_from_unit(unit, origin="graph_expansion", seed_id=edge["source_node_id"], path=list(next_edges))
                candidate["origin"] = "graph_path_retrieval"
                candidate["provenance"] = {
                    **candidate["provenance"],
                    "graph_added": True,
                    "graph_path_retrieval_policy": resolved.policy_name,
                    "graph_path_identity": path_row["path_id"],
                    "path_score": path_row["path_score"],
                }
                added.append(candidate)
                if len(added) >= resolved.maximum_added_candidates:
                    stop_reason = "max_added_candidates"
                    break
            if stop_reason in {"max_paths_per_query", "max_added_candidates"}:
                break
        if stop_reason in {"max_paths_per_query", "max_added_candidates"}:
            break

    candidates = tuple(graph_retrieval.merge_candidates(seed_candidates, added))
    trace = {
        "schema_version": "opk-rag.runtime-v2.graph-path-retrieval-trace.v1",
        "source_task": SOURCE_TASK,
        "graph_path_retrieval_policy_name": resolved.policy_name,
        "graph_path_retrieval_policy_version": resolved.policy_version,
        "graph_path_retrieval_policy_digest": resolved.policy_digest,
        "seed_candidate_ids": seed_candidate_ids,
        "allowed_edge_types": allowed_edge_types,
        "maximum_hops": resolved.maximum_hops,
        "max_neighbors_per_node": resolved.max_neighbors_per_node,
        "max_paths_per_query": resolved.max_paths_per_query,
        "maximum_expanded_nodes": resolved.maximum_expanded_nodes,
        "maximum_added_candidates": resolved.maximum_added_candidates,
        "traversed_edges": traversed_edges,
        "path_ids": [row["path_id"] for row in paths],
        "graph_added_candidate_ids": [candidate["candidate_id"] for candidate in added],
        "graph_path_retrieval_digest": stable_digest({"seed": seed_candidate_ids, "paths": paths, "added": [candidate["candidate_id"] for candidate in added]}),
        "stop_reason": stop_reason,
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
    }
    diagnostics = {
        "graph_activation_state": "graph_path_candidates_added" if added else "graph_path_activated_no_candidates",
        "graph_traversal_latency_ms": (perf_counter() - started) * 1000,
        "baseline_candidate_count": len(seed_candidates),
        "graph_added_candidate_count": len(added),
        "final_candidate_count": len(candidates),
        "path_count": len(paths),
        "expanded_node_count": len(expanded_nodes),
        "additional_vector_retrieval_calls": 0,
        "additional_lexical_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "additional_model_calls": 0,
    }
    return GraphPathRetrievalResult(candidates=candidates, added_candidates=tuple(added), paths=tuple(paths), trace=trace, diagnostics=diagnostics)


def stable_path_identity(edges: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    payload = [(edge.get("edge_id"), edge.get("source_node_id"), edge.get("edge_type"), edge.get("target_node_id")) for edge in edges]
    return "path:" + stable_digest(payload)[:24]


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _allowed_edge_types(graph_snapshot: dict[str, Any], policy: GraphPathRetrievalPolicy) -> list[str]:
    if policy.relation_aware:
        requested = set(graph_snapshot.get("allowed_edge_types") or [])
    else:
        requested = set(policy.supported_edge_types)
    return sorted(requested & set(policy.supported_edge_types))


def _candidate_units_by_document(graph_snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    units = graph_snapshot.get("candidate_source_units") or graph_snapshot.get("required_source_units", [])
    out: dict[str, list[dict[str, Any]]] = {}
    for unit in sorted(units, key=lambda row: (row["document_id"], row["source_unit_id"])):
        out.setdefault(unit["document_id"], []).append(unit)
    return out


def _path_row(*, seed_docs: list[str], edges: list[dict[str, Any]], target_node_id: str) -> dict[str, Any]:
    path_id = stable_path_identity(edges)
    return {
        "schema_version": "opk-rag.runtime-v2.graph-path.v1",
        "path_id": path_id,
        "source_seed_node_ids": seed_docs,
        "target_node_id": target_node_id,
        "hop_count": len(edges),
        "edge_ids": [edge["edge_id"] for edge in edges],
        "edge_types": [edge["edge_type"] for edge in edges],
        "node_sequence": [edges[0]["source_node_id"], *[edge["target_node_id"] for edge in edges]],
        "path_score": deterministic_path_score(edges),
    }


def deterministic_path_score(edges: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    authority_weight = {"G0": 1.0, "G1": 0.9, "G2": 0.8}
    if not edges:
        return 1.0
    base = sum(authority_weight.get(edge.get("authority_level", "G1"), 0.5) for edge in edges) / len(edges)
    return round(base / len(edges), 6)
