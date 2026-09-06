from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Protocol

from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import (
    LateInteractionRetrievalResult,
    dense_candidates,
    hybrid_candidates,
    late_candidates,
    union_candidates,
)
from opk_rag.runtime_v2.retriever_aware_fusion import RetrieverAwareFusionPolicy, retriever_aware_fusion


GUARD_POLICY_VERSION = "task0117-runtime-observable-confidence-v1"
DEFAULT_GUARD_THRESHOLD = 0.54
DETERMINISTIC_GUARD_V2_POLICY_VERSION = "task0121-deterministic-guard-v2"
DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD = 0.9966115298579041
DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD = 0.7989456995


class LateRetriever(Protocol):
    def retrieve(self, evaluation_unit_id: str, question: str, *, top_k: int) -> LateInteractionRetrievalResult:
        ...


@dataclass(frozen=True)
class RuntimePolicyConfig:
    late_interaction_enabled: bool = True
    explicit_dense_only_override: bool = False
    guard_threshold: float = DEFAULT_GUARD_THRESHOLD
    retrieval_top_k: int = 20
    dense_retrieval_latency_ms: float = 4.0


@dataclass(frozen=True)
class PolicyExecution:
    policy_id: str
    final_candidates: list[dict[str, Any]]
    dense_candidates: list[dict[str, Any]]
    late_candidates: list[dict[str, Any]]
    guard_decision: dict[str, Any] | None
    retrievers_executed: list[str]
    latency_ms: dict[str, float]
    fallback_used: bool = False


@dataclass(frozen=True)
class DeterministicGuardV2:
    normalized_score_entropy_threshold: float = DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD
    top20_score_mean_threshold: float = DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD
    policy_version: str = DETERMINISTIC_GUARD_V2_POLICY_VERSION

    def features(self, dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return guard_v2_features(dense_rows)

    def decide(self, dense_rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
        start = time.perf_counter()
        features = self.features(dense_rows)
        triggered = (
            features["normalized_score_entropy"] >= self.normalized_score_entropy_threshold
            or features["top20_score_mean"] >= self.top20_score_mean_threshold
        )
        decision = {
            "guard_triggered": triggered,
            "guard_features": features,
            "guard_policy_version": self.policy_version,
            "normalized_score_entropy_threshold": self.normalized_score_entropy_threshold,
            "top20_score_mean_threshold": self.top20_score_mean_threshold,
            "runtime_observable_only": True,
            "guard_evaluation_latency_ms": (time.perf_counter() - start) * 1000,
        }
        return triggered, decision


def guard_v2_features(dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_dense_score(row) for row in dense_rows[:20]]
    return {
        "normalized_score_entropy": _normalized_entropy_task0120(scores),
        "top20_score_mean": statistics.mean(scores) if scores else 0.0,
        "dense_top20_score_count": len(scores),
        "runtime_observable_only": True,
        "feature_semantics": "task0120_dense_score_top20",
        "feature_requires_no_additional_model_call": True,
    }


def should_trigger_late_interaction_v2(dense_rows: list[dict[str, Any]], *, guard: DeterministicGuardV2 | None = None) -> tuple[bool, dict[str, Any]]:
    return (guard or DeterministicGuardV2()).decide(dense_rows)


def guard_features(dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_dense_score(row) for row in dense_rows[:5]]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top5 = scores[4] if len(scores) > 4 else scores[-1] if scores else 0.0
    docs = {row.get("document_id") for row in dense_rows[:5] if row.get("document_id")}
    sections = {row.get("section_id") for row in dense_rows[:5] if row.get("section_id")}
    normalized = _normalize_scores(scores)
    return {
        "dense_top1_score": top1,
        "dense_topk_score_mean": statistics.mean(scores) if scores else 0.0,
        "top1_top2_margin": top1 - top2,
        "top1_top5_margin": top1 - top5,
        "score_entropy": _entropy(normalized),
        "candidate_document_diversity": len(docs),
        "candidate_section_diversity": len(sections),
        "candidate_count": len(dense_rows),
        "retrieval_confidence": retrieval_confidence(dense_rows),
        "runtime_observable_only": True,
    }


def retrieval_confidence(dense_rows: list[dict[str, Any]]) -> float:
    features = guard_features_without_confidence(dense_rows)
    top1 = _clamp(features["dense_top1_score"], 0.0, 1.0)
    margin = _clamp(features["top1_top2_margin"], 0.0, 1.0)
    diversity_penalty = min(features["candidate_document_diversity"], 5) / 5
    entropy_penalty = _clamp(features["score_entropy"], 0.0, 1.0)
    return _clamp((0.55 * top1) + (0.30 * margin) + (0.15 * diversity_penalty) - (0.10 * entropy_penalty), 0.0, 1.0)


def guard_features_without_confidence(dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_dense_score(row) for row in dense_rows[:5]]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top5 = scores[4] if len(scores) > 4 else scores[-1] if scores else 0.0
    docs = {row.get("document_id") for row in dense_rows[:5] if row.get("document_id")}
    sections = {row.get("section_id") for row in dense_rows[:5] if row.get("section_id")}
    return {
        "dense_top1_score": top1,
        "dense_topk_score_mean": statistics.mean(scores) if scores else 0.0,
        "top1_top2_margin": top1 - top2,
        "top1_top5_margin": top1 - top5,
        "score_entropy": _entropy(_normalize_scores(scores)),
        "candidate_document_diversity": len(docs),
        "candidate_section_diversity": len(sections),
        "candidate_count": len(dense_rows),
    }


def should_trigger_late_interaction(dense_rows: list[dict[str, Any]], *, threshold: float = DEFAULT_GUARD_THRESHOLD) -> tuple[bool, dict[str, Any]]:
    start = time.perf_counter()
    features = guard_features(dense_rows)
    triggered = features["retrieval_confidence"] < threshold
    decision = {
        "guard_triggered": triggered,
        "guard_features": features,
        "guard_policy_version": GUARD_POLICY_VERSION,
        "guard_threshold": threshold,
        "runtime_observable_only": True,
        "guard_evaluation_latency_ms": (time.perf_counter() - start) * 1000,
    }
    return triggered, decision


def execute_dense_policy(evaluation_unit_id: str, dense_rows: list[dict[str, Any]], *, config: RuntimePolicyConfig | None = None) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    candidates = dense_candidates(dense_rows[: cfg.retrieval_top_k])
    return PolicyExecution(
        policy_id="dense_default",
        final_candidates=candidates,
        dense_candidates=candidates,
        late_candidates=[],
        guard_decision=None,
        retrievers_executed=["dense"],
        latency_ms=_latency(cfg.dense_retrieval_latency_ms, 0.0, 0.0),
    )


def execute_late_policy(
    evaluation_unit_id: str,
    question: str,
    dense_rows: list[dict[str, Any]],
    late_retriever: LateRetriever,
    *,
    config: RuntimePolicyConfig | None = None,
) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    if not cfg.late_interaction_enabled or cfg.explicit_dense_only_override:
        return execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
    try:
        late_result = late_retriever.retrieve(evaluation_unit_id, question, top_k=cfg.retrieval_top_k)
    except Exception:
        fallback = execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
        return PolicyExecution(**{**fallback.__dict__, "fallback_used": True})
    late = late_candidates(late_result.hits[: cfg.retrieval_top_k])
    return PolicyExecution(
        policy_id="late_default",
        final_candidates=late,
        dense_candidates=[],
        late_candidates=late,
        guard_decision=None,
        retrievers_executed=["late_interaction"],
        latency_ms=_latency(0.0, late_result.retrieval_latency_ms, 0.0),
    )


def execute_hybrid_policy(
    evaluation_unit_id: str,
    question: str,
    dense_rows: list[dict[str, Any]],
    late_retriever: LateRetriever,
    *,
    config: RuntimePolicyConfig | None = None,
) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    dense = dense_candidates(dense_rows[: cfg.retrieval_top_k])
    if not cfg.late_interaction_enabled or cfg.explicit_dense_only_override:
        return execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
    try:
        late_result = late_retriever.retrieve(evaluation_unit_id, question, top_k=cfg.retrieval_top_k)
    except Exception:
        fallback = execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
        return PolicyExecution(**{**fallback.__dict__, "fallback_used": True})
    late = late_candidates(late_result.hits[: cfg.retrieval_top_k])
    final = hybrid_candidates(dense, late)[: cfg.retrieval_top_k]
    return PolicyExecution(
        policy_id="dense_late_hybrid_default",
        final_candidates=final,
        dense_candidates=dense,
        late_candidates=late,
        guard_decision=None,
        retrievers_executed=["dense", "late_interaction"],
        latency_ms=_latency(cfg.dense_retrieval_latency_ms, late_result.retrieval_latency_ms, 0.2),
    )


def execute_guarded_policy(
    evaluation_unit_id: str,
    question: str,
    dense_rows: list[dict[str, Any]],
    late_retriever: LateRetriever,
    *,
    config: RuntimePolicyConfig | None = None,
) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    dense = dense_candidates(dense_rows[: cfg.retrieval_top_k])
    if not cfg.late_interaction_enabled or cfg.explicit_dense_only_override:
        return execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
    triggered, decision = should_trigger_late_interaction(dense_rows, threshold=cfg.guard_threshold)
    if not triggered:
        decision["retrievers_executed"] = ["dense"]
        return PolicyExecution(
            policy_id="guarded_late_interaction",
            final_candidates=dense,
            dense_candidates=dense,
            late_candidates=[],
            guard_decision=decision,
            retrievers_executed=["dense"],
            latency_ms=_latency(cfg.dense_retrieval_latency_ms, 0.0, decision["guard_evaluation_latency_ms"]),
        )
    try:
        late_result = late_retriever.retrieve(evaluation_unit_id, question, top_k=cfg.retrieval_top_k)
    except Exception:
        decision["retrievers_executed"] = ["dense"]
        fallback = execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
        return PolicyExecution(**{**fallback.__dict__, "policy_id": "guarded_late_interaction", "guard_decision": decision, "fallback_used": True})
    late = late_candidates(late_result.hits[: cfg.retrieval_top_k])
    final = union_candidates(dense, late, policy="candidate_union")[: cfg.retrieval_top_k]
    decision["retrievers_executed"] = ["dense", "late_interaction"]
    return PolicyExecution(
        policy_id="guarded_late_interaction",
        final_candidates=final,
        dense_candidates=dense,
        late_candidates=late,
        guard_decision=decision,
        retrievers_executed=["dense", "late_interaction"],
        latency_ms=_latency(cfg.dense_retrieval_latency_ms, late_result.retrieval_latency_ms, decision["guard_evaluation_latency_ms"] + 0.2),
    )


def execute_guarded_v2_policy(
    evaluation_unit_id: str,
    question: str,
    dense_rows: list[dict[str, Any]],
    late_retriever: LateRetriever,
    *,
    config: RuntimePolicyConfig | None = None,
    guard: DeterministicGuardV2 | None = None,
) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    dense = dense_candidates(dense_rows[: cfg.retrieval_top_k])
    if not cfg.late_interaction_enabled or cfg.explicit_dense_only_override:
        return execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
    triggered, decision = should_trigger_late_interaction_v2(dense_rows, guard=guard)
    if not triggered:
        decision["retrievers_executed"] = ["dense"]
        return PolicyExecution(
            policy_id="guarded_clean_late_v2",
            final_candidates=dense,
            dense_candidates=dense,
            late_candidates=[],
            guard_decision=decision,
            retrievers_executed=["dense"],
            latency_ms=_latency(cfg.dense_retrieval_latency_ms, 0.0, decision["guard_evaluation_latency_ms"]),
        )
    try:
        late_result = late_retriever.retrieve(evaluation_unit_id, question, top_k=cfg.retrieval_top_k)
    except Exception:
        decision["retrievers_executed"] = ["dense"]
        fallback = execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
        return PolicyExecution(**{**fallback.__dict__, "policy_id": "guarded_clean_late_v2", "guard_decision": decision, "fallback_used": True})
    late = late_candidates(late_result.hits[: cfg.retrieval_top_k])
    final = union_candidates(dense, late, policy="candidate_union")[: cfg.retrieval_top_k]
    decision["retrievers_executed"] = ["dense", "late_interaction"]
    return PolicyExecution(
        policy_id="guarded_clean_late_v2",
        final_candidates=final,
        dense_candidates=dense,
        late_candidates=late,
        guard_decision=decision,
        retrievers_executed=["dense", "late_interaction"],
        latency_ms=_latency(cfg.dense_retrieval_latency_ms, late_result.retrieval_latency_ms, decision["guard_evaluation_latency_ms"] + 0.2),
    )


def execute_guarded_v2_retriever_aware_fusion_policy(
    evaluation_unit_id: str,
    question: str,
    dense_rows: list[dict[str, Any]],
    late_retriever: LateRetriever,
    *,
    config: RuntimePolicyConfig | None = None,
    guard: DeterministicGuardV2 | None = None,
    fusion_policy: RetrieverAwareFusionPolicy | None = None,
) -> PolicyExecution:
    cfg = config or RuntimePolicyConfig()
    dense = dense_candidates(dense_rows[: cfg.retrieval_top_k])
    if not cfg.late_interaction_enabled or cfg.explicit_dense_only_override:
        return execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
    triggered, decision = should_trigger_late_interaction_v2(dense_rows, guard=guard)
    if not triggered:
        decision["retrievers_executed"] = ["dense"]
        return PolicyExecution(
            policy_id="guarded_clean_late_v2_retriever_aware_fusion",
            final_candidates=dense,
            dense_candidates=dense,
            late_candidates=[],
            guard_decision=decision,
            retrievers_executed=["dense"],
            latency_ms=_latency(cfg.dense_retrieval_latency_ms, 0.0, decision["guard_evaluation_latency_ms"]),
        )
    try:
        late_result = late_retriever.retrieve(evaluation_unit_id, question, top_k=cfg.retrieval_top_k)
    except Exception:
        decision["retrievers_executed"] = ["dense"]
        fallback = execute_dense_policy(evaluation_unit_id, dense_rows, config=cfg)
        return PolicyExecution(**{**fallback.__dict__, "policy_id": "guarded_clean_late_v2_retriever_aware_fusion", "guard_decision": decision, "fallback_used": True})
    late = late_candidates(late_result.hits[: cfg.retrieval_top_k])
    start = time.perf_counter()
    final = retriever_aware_fusion(dense, late, policy=fusion_policy)
    fusion_latency = (time.perf_counter() - start) * 1000
    decision["retrievers_executed"] = ["dense", "late_interaction"]
    return PolicyExecution(
        policy_id="guarded_clean_late_v2_retriever_aware_fusion",
        final_candidates=final,
        dense_candidates=dense,
        late_candidates=late,
        guard_decision=decision,
        retrievers_executed=["dense", "late_interaction"],
        latency_ms=_latency(cfg.dense_retrieval_latency_ms, late_result.retrieval_latency_ms, decision["guard_evaluation_latency_ms"] + fusion_latency),
    )


def _latency(dense_ms: float, late_ms: float, overhead_ms: float) -> dict[str, float]:
    return {
        "dense_retrieval_latency": dense_ms,
        "late_retrieval_latency": late_ms,
        "guard_or_fusion_latency": overhead_ms,
        "end_to_end_retrieval_latency": dense_ms + late_ms + overhead_ms,
    }


def _dense_score(row: dict[str, Any]) -> float:
    score = row.get("retrieval_score")
    if isinstance(score, (int, float)):
        return float(score)
    rank = int(row.get("retrieval_rank") or 999999)
    return 1.0 / max(rank, 1)


def _normalize_scores(scores: list[float]) -> list[float]:
    total = sum(max(score, 0.0) for score in scores)
    if total <= 0:
        return []
    return [max(score, 0.0) / total for score in scores]


def _entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 1.0
    raw = -sum(p * math.log(p) for p in probabilities if p > 0)
    return raw / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0


def _normalized_entropy_task0120(values: list[float]) -> float:
    total = sum(max(v, 0.0) for v in values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    probabilities = [max(v, 0.0) / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probabilities) / math.log(len(values))


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
