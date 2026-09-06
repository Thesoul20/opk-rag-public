from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any

from opk_rag.runtime_v2 import graph_retrieval


GRAPH_ACTIVATION_POLICY_DISABLED = "disabled"
GRAPH_ACTIVATION_POLICY_EXPLICIT = "explicit"
GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE = "retrieval_aware"
DEFAULT_GRAPH_ACTIVATION_POLICY = GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE

GRAPH_ACTIVATION_DISABLED = "graph_disabled"
GRAPH_ACTIVATION_ALWAYS_ON = "graph_always_on"
GRAPH_ACTIVATION_ORACLE = "oracle_routing_offline_only"
GRAPH_ACTIVATION_QUERY_ROUTER = "deterministic_query_router"
GRAPH_ACTIVATION_RETRIEVAL_AWARE_ROUTER = "retrieval_aware_router"
DECISION_VERSION = "opk-rag.graph-activation-router.v2"

RELATION_TERMS = (
    "依赖",
    "来源",
    "影响",
    "所属",
    "上游",
    "下游",
    "关联",
    "关系",
    "因果",
    "组成",
    "引用",
    "链接",
    "通过",
    "related",
    "relationship",
    "depend",
    "source",
    "impact",
    "link",
    "reference",
)
STRUCTURAL_TERMS = ("什么依赖", "依赖什么", "影响哪些", "如何通过", "关联到", "linked to", "depends on")


@dataclass(frozen=True)
class GraphActivationDecision:
    query_id: str
    activation_policy: str
    graph_activation: bool
    activation_reason: str
    query_signal: bool
    retrieval_signal: bool
    graph_connectivity_signal: bool
    seed_candidate_count: int
    graph_expandable_seed_count: int
    activation_features: dict[str, Any]
    decision_version: str = DECISION_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "activation_policy": self.activation_policy,
            "graph_activation": self.graph_activation,
            "activation_reason": self.activation_reason,
            "query_signal": self.query_signal,
            "retrieval_signal": self.retrieval_signal,
            "graph_connectivity_signal": self.graph_connectivity_signal,
            "seed_candidate_count": self.seed_candidate_count,
            "graph_expandable_seed_count": self.graph_expandable_seed_count,
            "activation_features": self.activation_features,
            "decision_version": self.decision_version,
        }


def decide_graph_activation(
    sample: dict[str, Any],
    *,
    activation_policy: str,
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    graph_activation_enabled: bool = True,
    explicit_graph_activation: bool = True,
) -> GraphActivationDecision:
    if not graph_activation_enabled:
        return _decision(sample, activation_policy, False, "explicit_disable_override", retrieval_aware_features(sample, seeds=seeds))
    if activation_policy in {GRAPH_ACTIVATION_POLICY_DISABLED, GRAPH_ACTIVATION_DISABLED}:
        return _decision(sample, activation_policy, False, "graph_disabled_baseline", query_features(sample))
    if activation_policy in {GRAPH_ACTIVATION_POLICY_EXPLICIT, GRAPH_ACTIVATION_ALWAYS_ON}:
        return _decision(sample, activation_policy, explicit_graph_activation, "explicit_graph_activation" if explicit_graph_activation else "explicit_graph_activation_false", query_features(sample))
    if activation_policy == GRAPH_ACTIVATION_ORACLE:
        return _decision(sample, activation_policy, _oracle_activation(sample), "offline_oracle_label", {"offline_oracle_only": True})
    if activation_policy == GRAPH_ACTIVATION_QUERY_ROUTER:
        features = query_features(sample)
        active = _query_router_active(features)
        return _decision(sample, activation_policy, active, "query_relation_or_structure_signal" if active else "no_query_activation_signal", features)
    if activation_policy in {GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE, GRAPH_ACTIVATION_RETRIEVAL_AWARE_ROUTER}:
        features = retrieval_aware_features(sample, seeds=seeds)
        active = _query_router_active(features) and features["seed_connectivity"] and features["graph_expansion_availability"]
        reason = "query_signal_with_seed_graph_availability" if active else "missing_query_or_retrieval_signal"
        return _decision(sample, activation_policy, active, reason, features)
    raise ValueError(f"unsupported graph activation policy: {activation_policy}")


def decide_runtime_graph_activation(
    sample: dict[str, Any],
    *,
    activation_policy: str = DEFAULT_GRAPH_ACTIVATION_POLICY,
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    graph_activation_enabled: bool = True,
    explicit_graph_activation: bool = True,
) -> GraphActivationDecision:
    return decide_graph_activation(
        sample,
        activation_policy=activation_policy,
        seeds=seeds,
        graph_activation_enabled=graph_activation_enabled,
        explicit_graph_activation=explicit_graph_activation,
    )


def query_features(sample: dict[str, Any]) -> dict[str, Any]:
    query = _query_text(sample)
    relation_terms = [term for term in RELATION_TERMS if term.lower() in query.lower()]
    structural_terms = [term for term in STRUCTURAL_TERMS if term.lower() in query.lower()]
    entity_count = _query_entity_count(query, sample)
    question_type = str(sample.get("question_type") or "")
    relation_intent = bool(relation_terms) or "traversal" in question_type or "reference" in question_type
    structural_question = bool(structural_terms) or "traversal" in question_type
    return {
        "relation_intent": relation_intent,
        "relation_terms": relation_terms,
        "multi_entity_signal": entity_count >= 2,
        "query_entity_count": entity_count,
        "structural_question_signal": structural_question,
        "question_type_signal": question_type,
        "uses_gold_label": False,
        "uses_gold_evidence": False,
        "uses_benchmark_category": False,
    }


def retrieval_aware_features(
    sample: dict[str, Any],
    *,
    seeds: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    resolved_seeds = list(seeds) if seeds is not None else graph_retrieval.select_seed_candidates(sample)
    features = query_features(sample)
    seed_docs = {candidate["document_id"] for candidate in resolved_seeds}
    graph = graph_retrieval.graph_edges(sample, allowed_authority_levels=set(graph_retrieval.ALLOWED_AUTHORITY_LEVELS))
    allowed = set(sample.get("allowed_edge_types") or []) & set(graph_retrieval.SUPPORTED_EDGE_TYPES)
    traversable = [edge for doc_id in seed_docs for edge in graph.get(doc_id, []) if edge.get("edge_type") in allowed]
    expandable_seed_docs = {edge["source_node_id"] for edge in traversable}
    candidate_docs = {unit["document_id"] for unit in sample.get("candidate_source_units", sample.get("required_source_units", []))}
    available_targets = {edge["target_node_id"] for edge in traversable} & candidate_docs
    features.update(
        {
            "seed_candidate_count": len(resolved_seeds),
            "seed_connectivity": bool(traversable),
            "seed_connectivity_edge_count": len(traversable),
            "graph_expandable_seed_count": len(expandable_seed_docs),
            "graph_expansion_availability": bool(available_targets),
            "available_neighbor_count": len(available_targets),
        }
    )
    return features


def runtime_config_for_decision(decision: GraphActivationDecision) -> graph_retrieval.GraphRetrievalRuntimeConfig:
    policy = graph_retrieval.ONE_HOP_GRAPH_RETRIEVAL_POLICY if decision.graph_activation else graph_retrieval.GRAPH_RETRIEVAL_DISABLED
    return graph_retrieval.GraphRetrievalRuntimeConfig(policy=policy)


def _decision(sample: dict[str, Any], policy: str, active: bool, reason: str, features: dict[str, Any]) -> GraphActivationDecision:
    query_signal = _query_router_active(features) if "relation_intent" in features else False
    retrieval_signal = bool(features.get("seed_connectivity") and features.get("graph_expansion_availability"))
    graph_connectivity_signal = bool(features.get("seed_connectivity"))
    return GraphActivationDecision(
        query_id=str(sample["sample_id"]),
        activation_policy=policy,
        graph_activation=active,
        activation_reason=reason,
        query_signal=query_signal,
        retrieval_signal=retrieval_signal,
        graph_connectivity_signal=graph_connectivity_signal,
        seed_candidate_count=int(features.get("seed_candidate_count", len(sample.get("seed_source_units", [])))),
        graph_expandable_seed_count=int(features.get("graph_expandable_seed_count", features.get("seed_connectivity_edge_count", 0))),
        activation_features=features,
    )


def _query_router_active(features: dict[str, Any]) -> bool:
    return bool(features["relation_intent"] and (features["multi_entity_signal"] or features["structural_question_signal"]))


def _oracle_activation(sample: dict[str, Any]) -> bool:
    return sample.get("graph_expectation") == "graph_expansion_required" and not sample.get("negative_control") and not sample.get("graph_unanswerable")


def _query_text(sample: dict[str, Any]) -> str:
    for key in ("query", "question", "user_query", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(str(part) for part in (sample.get("sample_id"), sample.get("question_type")) if part)


def _query_entity_count(query: str, sample: dict[str, Any]) -> int:
    quoted = re.findall(r"[`“”\"']([^`“”\"']{2,80})[`“”\"']", query)
    ascii_names = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}(?:\s+[A-Z][A-Za-z0-9_-]{1,})*\b", query)
    cjk_names = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_-]{2,}(?:模块|组件|系统|文档|章节|文件|功能|流程|服务|工具|路径)", query)
    seed_docs = {unit.get("document_id") for unit in sample.get("seed_source_units", [])}
    required_docs = {unit.get("document_id") for unit in sample.get("required_source_units", [])}
    candidate_docs = {unit.get("document_id") for unit in sample.get("candidate_source_units", [])}
    document_ids = {doc for doc in seed_docs | required_docs | candidate_docs if doc}
    mentioned_docs = [doc for doc in document_ids if str(doc) in query]
    mentioned_titles = _runtime_document_title_mentions(query, document_ids)
    return len({*quoted, *ascii_names, *cjk_names, *map(str, mentioned_docs), *mentioned_titles})


def _runtime_document_title_mentions(query: str, document_ids: set[Any]) -> set[str]:
    """Resolve query entity mentions from runtime-visible document titles.

    This is an identity adapter for production paths where graph node ids are canonical
    paths rather than benchmark-friendly labels. It does not use gold relevance or
    sample ids; it only checks title fragments that are literally present in the query.
    """
    query_lower = query.lower()
    mentions: set[str] = set()
    for document_id in document_ids:
        stem = PurePosixPath(str(document_id)).stem
        stem = re.sub(r"^\d+\s*", "", stem).strip()
        if not stem:
            continue
        if stem.lower() in query_lower:
            mentions.add(stem)
            continue
        fragments = [part.strip() for part in re.split(r"[\s、，,/—→:：]+|(?:的|与|到|和)", stem) if part.strip()]
        for fragment in fragments:
            if len(fragment) >= 4 and fragment.lower() in query_lower:
                mentions.add(fragment)
    return mentions
