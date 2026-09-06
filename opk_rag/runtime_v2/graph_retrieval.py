from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any


GRAPH_RETRIEVAL_DISABLED = "disabled"
ONE_HOP_GRAPH_RETRIEVAL_POLICY = "one_hop_relation_expansion"
POLICY_VERSION = "opk-rag.graph-retrieval.one-hop-relation-expansion.v1"
SOURCE_TASK = "TASK-0137"
MAXIMUM_HOPS = 1
DEFAULT_SEED_COUNT = 3
DEFAULT_MAXIMUM_EXPANDED_NODES = 20
DEFAULT_MAXIMUM_ADDED_CANDIDATES = 5
ALLOWED_AUTHORITY_LEVELS = ("G0", "G1", "G2")
SUPPORTED_EDGE_TYPES = ("BELONGS_TO", "CONTAINS", "EMBEDS", "LINKS_TO", "NEXT_SECTION", "PARENT_OF", "TAGGED_WITH")


class GraphRetrievalConfigError(ValueError):
    pass


class GraphSnapshotCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class GraphRetrievalPolicy:
    policy_name: str = ONE_HOP_GRAPH_RETRIEVAL_POLICY
    policy_version: str = POLICY_VERSION
    source_task: str = SOURCE_TASK
    maximum_hops: int = MAXIMUM_HOPS
    maximum_expanded_nodes: int = DEFAULT_MAXIMUM_EXPANDED_NODES
    maximum_added_candidates: int = DEFAULT_MAXIMUM_ADDED_CANDIDATES
    allowed_authority_levels: tuple[str, ...] = ALLOWED_AUTHORITY_LEVELS
    supported_edge_types: tuple[str, ...] = SUPPORTED_EDGE_TYPES

    def __post_init__(self) -> None:
        if self.policy_name != ONE_HOP_GRAPH_RETRIEVAL_POLICY:
            raise GraphRetrievalConfigError(f"unsupported graph retrieval policy: {self.policy_name}")
        if self.maximum_hops != MAXIMUM_HOPS:
            raise GraphRetrievalConfigError("promoted graph retrieval policy is frozen to maximum_hops=1")
        if self.maximum_expanded_nodes < 1:
            raise GraphRetrievalConfigError("maximum_expanded_nodes must be positive")
        if self.maximum_added_candidates < 0:
            raise GraphRetrievalConfigError("maximum_added_candidates must be non-negative")

    @property
    def policy_digest(self) -> str:
        return stable_digest(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "source_task": self.source_task,
            "maximum_hops": self.maximum_hops,
            "maximum_expanded_nodes": self.maximum_expanded_nodes,
            "maximum_added_candidates": self.maximum_added_candidates,
            "allowed_authority_levels": list(self.allowed_authority_levels),
            "supported_edge_types": list(self.supported_edge_types),
            "seed_policy": f"top_{DEFAULT_SEED_COUNT}_authoritative_seed_source_units",
            "dedup_behavior": "preserve_seed_order_then_first_graph_added_candidate",
            "candidate_provenance_required": True,
            "gold_signal_used_by_runtime_graph_policy": False,
            "benchmark_specific_graph_rule": False,
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_digest"] = self.policy_digest
        return payload


@dataclass(frozen=True)
class GraphRetrievalRuntimeConfig:
    policy: str = GRAPH_RETRIEVAL_DISABLED
    fallback_on_failure: bool = True

    def __post_init__(self) -> None:
        if self.policy not in {GRAPH_RETRIEVAL_DISABLED, ONE_HOP_GRAPH_RETRIEVAL_POLICY, "one_hop"}:
            raise GraphRetrievalConfigError(f"unsupported graph retrieval activation policy: {self.policy}")

    @property
    def graph_enabled(self) -> bool:
        return self.policy in {ONE_HOP_GRAPH_RETRIEVAL_POLICY, "one_hop"}


@dataclass(frozen=True)
class GraphExpansionResult:
    candidates: tuple[dict[str, Any], ...]
    added_candidates: tuple[dict[str, Any], ...]
    trace: dict[str, Any]
    diagnostics: dict[str, Any]


def default_graph_retrieval_policy() -> GraphRetrievalPolicy:
    return GraphRetrievalPolicy()


def select_seed_candidates(sample: dict[str, Any], *, seed_count: int = DEFAULT_SEED_COUNT) -> list[dict[str, Any]]:
    candidates = [_candidate_from_unit(unit, origin="canonical_seed") for unit in sample.get("seed_source_units", [])]
    return candidates[:seed_count]


def expand_graph_candidates(
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    graph_snapshot: dict[str, Any],
    policy: GraphRetrievalPolicy | None = None,
) -> GraphExpansionResult:
    resolved = policy or default_graph_retrieval_policy()
    started = perf_counter()
    validate_graph_snapshot(graph_snapshot)
    seed_candidates = list(seeds)
    allowed_edge_types = sorted(set(graph_snapshot.get("allowed_edge_types") or []) & set(resolved.supported_edge_types))
    graph = graph_edges(graph_snapshot, allowed_authority_levels=set(resolved.allowed_authority_levels))
    seed_ids = {candidate["document_id"] for candidate in seed_candidates}
    seen_nodes = set(seed_ids)
    added: list[dict[str, Any]] = []
    traversed: list[dict[str, Any]] = []
    queue = deque((node, 0, []) for node in sorted(seed_ids))
    stop_reason = "completed"
    while queue:
        node, hop, path = queue.popleft()
        if hop >= resolved.maximum_hops:
            continue
        if len(seen_nodes) > resolved.maximum_expanded_nodes:
            stop_reason = "max_nodes"
            break
        for edge in graph.get(node, []):
            if edge["edge_type"] not in allowed_edge_types:
                continue
            traversed.append(edge)
            target = edge["target_node_id"]
            next_path = [*path, edge]
            if target not in seen_nodes:
                seen_nodes.add(target)
                queue.append((target, hop + 1, next_path))
            existing_ids = {candidate["candidate_id"] for candidate in [*seed_candidates, *added]}
            for unit in graph_snapshot.get("candidate_source_units", graph_snapshot.get("required_source_units", [])):
                if unit["document_id"] == target and unit["source_unit_id"] not in existing_ids:
                    added.append(_candidate_from_unit(unit, origin="graph_expansion", seed_id=edge["source_node_id"], path=next_path))
                    if len(added) >= resolved.maximum_added_candidates:
                        stop_reason = "max_added_candidates"
                        break
            if stop_reason == "max_added_candidates":
                break
        if stop_reason == "max_added_candidates":
            break
    trace = {
        "schema_version": "opk-rag.runtime-v2.graph-expansion-trace.v1",
        "source_task": SOURCE_TASK,
        "graph_retrieval_policy_name": resolved.policy_name,
        "graph_retrieval_policy_version": resolved.policy_version,
        "graph_retrieval_policy_digest": resolved.policy_digest,
        "seed_candidate_ids": [candidate["candidate_id"] for candidate in seed_candidates],
        "allowed_edge_types": allowed_edge_types,
        "maximum_hops": resolved.maximum_hops,
        "maximum_expanded_nodes": resolved.maximum_expanded_nodes,
        "maximum_added_candidates": resolved.maximum_added_candidates,
        "traversed_edges": traversed,
        "graph_added_candidate_ids": [candidate["candidate_id"] for candidate in added],
        "graph_expansion_digest": stable_digest({"seed": sorted(seed_ids), "edges": traversed, "added": [candidate["candidate_id"] for candidate in added]}),
        "stop_reason": stop_reason,
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
    }
    candidates = tuple(merge_candidates(seed_candidates, added))
    diagnostics = {
        "graph_activation_state": "graph_candidates_added" if added else "graph_activated_no_neighbors",
        "graph_traversal_latency_ms": (perf_counter() - started) * 1000,
        "baseline_candidate_count": len(seed_candidates),
        "graph_added_candidate_count": len(added),
        "final_candidate_count": len(candidates),
        "additional_vector_retrieval_calls": 0,
        "additional_lexical_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "additional_model_calls": 0,
    }
    return GraphExpansionResult(candidates=candidates, added_candidates=tuple(added), trace=trace, diagnostics=diagnostics)


def expand_runtime_candidates(
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    graph_snapshot: dict[str, Any] | None,
    *,
    runtime_config: GraphRetrievalRuntimeConfig | None = None,
    policy: GraphRetrievalPolicy | None = None,
) -> GraphExpansionResult:
    cfg = runtime_config or GraphRetrievalRuntimeConfig()
    seed_tuple = tuple(seeds)
    disabled_trace = {
        "schema_version": "opk-rag.runtime-v2.graph-expansion-trace.v1",
        "source_task": SOURCE_TASK,
        "graph_retrieval_policy_name": GRAPH_RETRIEVAL_DISABLED,
        "graph_retrieval_policy_version": None,
        "graph_retrieval_policy_digest": None,
        "seed_candidate_ids": [candidate["candidate_id"] for candidate in seed_tuple],
        "allowed_edge_types": [],
        "maximum_hops": 0,
        "maximum_expanded_nodes": 0,
        "maximum_added_candidates": 0,
        "traversed_edges": [],
        "graph_added_candidate_ids": [],
        "graph_expansion_digest": stable_digest({"seed": [candidate["candidate_id"] for candidate in seed_tuple], "edges": [], "added": []}),
        "stop_reason": "graph_disabled",
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
    }
    if not cfg.graph_enabled:
        return GraphExpansionResult(
            candidates=seed_tuple,
            added_candidates=(),
            trace=disabled_trace,
            diagnostics={
                "graph_activation_state": "graph_not_activated",
                "fallback_used": False,
                "fallback_reason": None,
                "additional_model_calls": 0,
            },
        )
    try:
        if graph_snapshot is None:
            raise GraphSnapshotCompatibilityError("graph snapshot is unavailable")
        return expand_graph_candidates(seed_tuple, graph_snapshot, policy or default_graph_retrieval_policy())
    except Exception as exc:
        if not cfg.fallback_on_failure:
            raise
        return GraphExpansionResult(
            candidates=seed_tuple,
            added_candidates=(),
            trace={**disabled_trace, "stop_reason": "graph_failure_fallback"},
            diagnostics={
                "graph_activation_state": "graph_capability_unavailable_fallback_to_canonical",
                "fallback_used": True,
                "fallback_reason": exc.__class__.__name__,
                "additional_model_calls": 0,
            },
        )


def validate_graph_snapshot(graph_snapshot: dict[str, Any]) -> None:
    if not isinstance(graph_snapshot, dict):
        raise GraphSnapshotCompatibilityError("graph snapshot must be a mapping")
    if "required_graph_path" not in graph_snapshot:
        raise GraphSnapshotCompatibilityError("graph snapshot missing required_graph_path")
    if "allowed_edge_types" not in graph_snapshot:
        raise GraphSnapshotCompatibilityError("graph snapshot missing allowed_edge_types")


def graph_edges(graph_snapshot: dict[str, Any], *, allowed_authority_levels: set[str]) -> dict[str, list[dict[str, Any]]]:
    edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graph_snapshot.get("required_graph_path", []):
        if "edge_type" not in item:
            continue
        if item.get("authority_level", "G1") not in allowed_authority_levels:
            continue
        edges[item["source_node_id"]].append(item)
    return {source: sorted(rows, key=lambda row: (row["edge_type"], row["target_node_id"])) for source, rows in edges.items()}


def merge_candidates(seed: list[dict[str, Any]] | tuple[dict[str, Any], ...], added: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [*seed, *added]:
        if candidate["candidate_id"] in seen:
            continue
        seen.add(candidate["candidate_id"])
        out.append(candidate)
    return out


def candidate_provenance_rows(added_candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...], *, sample_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for candidate in added_candidates:
        provenance = dict(candidate.get("provenance") or {})
        rows.append(
            {
                "schema_version": "opk-rag.runtime-v2.graph-candidate-provenance.v1",
                "sample_id": sample_id,
                "candidate_id": candidate["candidate_id"],
                "canonical_chunk_id": candidate["canonical_chunk_id"],
                "document_id": candidate.get("document_id"),
                "section_id": candidate.get("section_id"),
                "graph_added": True,
                **provenance,
            }
        )
    return rows


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _candidate_from_unit(unit: dict[str, Any], *, origin: str, seed_id: str | None = None, path: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    candidate_id = unit["source_unit_id"]
    document_id = unit["document_id"]
    edge_types = [edge["edge_type"] for edge in path or []]
    return {
        "candidate_id": candidate_id,
        "canonical_chunk_id": candidate_id,
        "document_id": document_id,
        "section_id": unit.get("line_span"),
        "origin": origin,
        "provenance": {
            "candidate_id": candidate_id,
            "canonical_chunk_id": candidate_id,
            "document_id": document_id,
            "section_id": unit.get("line_span"),
            "graph_added": origin == "graph_expansion",
            "seed_candidate_id": seed_id,
            "source_seed_candidate_id": seed_id,
            "graph_path": path or [],
            "graph_hop_count": len(path or []),
            "graph_edge_types": edge_types,
            "edge_types": edge_types,
            "graph_expansion_reason": "explicit_authoritative_relation_path",
        },
    }
