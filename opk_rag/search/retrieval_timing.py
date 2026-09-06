from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator


RETRIEVAL_TIMING_STAGES = (
    "request_validation",
    "query_preparation",
    "query_preprocessing",
    "query_tokenization",
    "query_embedding",
    "embedding_tokenization",
    "embedding_host_to_device",
    "embedding_forward",
    "embedding_pooling_normalization",
    "embedding_device_to_host",
    "agent_control",
    "retriever_selection",
    "guard_evaluation",
    "recovery_decision",
    "recovery_execution",
    "retrieval_total",
    "qdrant_request_build",
    "qdrant_network_round_trip",
    "qdrant_response_parse",
    "vector_search",
    "lexical_search",
    "candidate_fusion",
    "retrieval_fusion",
    "structure_activation_decision",
    "structure_expansion",
    "document_relation_lookup",
    "same_document_chunk_expansion",
    "graph_lookup",
    "graph_neighbor_lookup",
    "graph_candidate_merge",
    "candidate_deduplication",
    "candidate_score_adjustment",
    "candidate_materialization",
    "reranking_tokenization",
    "reranking_host_to_device",
    "reranking_inference",
    "reranking_postprocessing",
    "reranking_total",
    "evidence_composition",
    "citation_preparation",
    "result_postprocessing",
    "result_serialization",
    "response_serialization",
)


@dataclass
class _ActiveTimer:
    stage: str
    start_ns: int
    child_ns: int = 0


@dataclass
class RetrievalTimingObserver:
    enabled: bool = False
    stage_latency_ns: dict[str, int] = field(default_factory=lambda: {stage: 0 for stage in RETRIEVAL_TIMING_STAGES})
    stage_inclusive_latency_ns: dict[str, int] = field(default_factory=lambda: {stage: 0 for stage in RETRIEVAL_TIMING_STAGES})
    stage_executed: dict[str, bool] = field(default_factory=lambda: {stage: False for stage in RETRIEVAL_TIMING_STAGES})
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    details: dict[str, Any] = field(default_factory=dict)
    _stack: list[_ActiveTimer] = field(default_factory=list, init=False, repr=False)
    _overhead_ns: int = 0

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        overhead_start = time.perf_counter_ns()
        timer = _ActiveTimer(stage=stage, start_ns=time.perf_counter_ns())
        self._stack.append(timer)
        self._overhead_ns += timer.start_ns - overhead_start
        try:
            yield
        finally:
            _synchronize_cuda_if_available()
            exit_start = time.perf_counter_ns()
            finished = self._stack.pop()
            elapsed_ns = max(0, exit_start - finished.start_ns)
            exclusive_ns = max(0, elapsed_ns - finished.child_ns)
            self.stage_latency_ns[stage] = self.stage_latency_ns.get(stage, 0) + exclusive_ns
            self.stage_inclusive_latency_ns[stage] = self.stage_inclusive_latency_ns.get(stage, 0) + elapsed_ns
            self.stage_executed[stage] = True
            if self._stack:
                self._stack[-1].child_ns += elapsed_ns
            self._overhead_ns += max(0, time.perf_counter_ns() - exit_start)

    def mark_not_executed(self, stage: str) -> None:
        self.stage_latency_ns.setdefault(stage, 0)
        self.stage_inclusive_latency_ns.setdefault(stage, 0)
        self.stage_executed.setdefault(stage, False)

    def increment(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counts[name] += amount

    def set_detail(self, name: str, value: Any) -> None:
        if self.enabled:
            self.details[name] = value

    @property
    def overhead_ms(self) -> float:
        return round(self._overhead_ns / 1_000_000, 6)

    def snapshot(self, *, retrieval_total_latency_ms: float | None = None) -> dict[str, Any]:
        stages = {}
        for stage in RETRIEVAL_TIMING_STAGES:
            exclusive_ms = round(float(self.stage_latency_ns.get(stage, 0)) / 1_000_000, 6)
            inclusive_ms = round(float(self.stage_inclusive_latency_ns.get(stage, 0)) / 1_000_000, 6)
            stages[stage] = {
                "stage_executed": bool(self.stage_executed.get(stage, False)),
                "latency_ms": exclusive_ms,
                "exclusive_latency_ms": exclusive_ms,
                "inclusive_latency_ms": inclusive_ms,
            }
        observed = sum(item["latency_ms"] for item in stages.values())
        unattributed = None
        ratio = None
        invalid = False
        if retrieval_total_latency_ms is not None:
            unattributed = round(float(retrieval_total_latency_ms) - observed, 6)
            invalid = unattributed < 0
            ratio = abs(unattributed) / float(retrieval_total_latency_ms) if retrieval_total_latency_ms else None
        return {
            "schema_version": "opk-rag.retrieval-timing-observation.v1",
            "enabled": self.enabled,
            "stages": stages,
            "counts": dict(self.counts),
            "details": dict(self.details),
            "instrumentation_overhead_ms": self.overhead_ms,
            "sum_observed_stage_latency_ms": round(observed, 6),
            "retrieval_total_latency_ms": round(float(retrieval_total_latency_ms), 6) if retrieval_total_latency_ms is not None else None,
            "unattributed_latency_ms": unattributed,
            "latency_reconciliation_error_ratio": round(ratio, 6) if ratio is not None else None,
            "valid_sample": not invalid,
        }


def _synchronize_cuda_if_available() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return
