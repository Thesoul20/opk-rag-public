from __future__ import annotations

from statistics import median
from typing import Any


def evidence_gain_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    second_rounds = [row for row in rows if row.get("round_index") == 2]
    extra_searches = len(second_rounds)
    new_docs = sum(int(row.get("new_document_count") or 0) for row in second_rounds)
    new_evidence = sum(int(row.get("new_evidence_identity_count") or 0) for row in second_rounds)
    duplicates = sum(int(row.get("duplicate_evidence_count") or 0) for row in second_rounds)
    no_gain = sum(1 for row in second_rounds if int(row.get("new_evidence_identity_count") or 0) == 0)
    answerability_gain = sum(
        1
        for row in second_rounds
        if row.get("answerability_before") in {"insufficient_evidence", "partial_evidence"} and row.get("answerability_after") in {"answerable", "partially_answerable"}
    )
    return {
        "schema_version": "opk-rag.task0067-evidence-gain-report.v1",
        "second_round_search_count": extra_searches,
        "second_round_new_document_count": new_docs,
        "second_round_new_evidence_identity_count": new_evidence,
        "second_round_duplicate_evidence_count": duplicates,
        "no_evidence_gain_sample_count": no_gain,
        "evidence_gain_per_extra_search": new_evidence / extra_searches if extra_searches else 0,
        "answerability_gain_per_extra_search": answerability_gain / extra_searches if extra_searches else 0,
    }


def answerability_transition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    second_rounds = [row for row in rows if row.get("round_index") == 2]
    return {
        "schema_version": "opk-rag.task0067-answerability-transition-report.v1",
        "first_round_insufficient_count": sum(1 for row in rows if row.get("round_index") == 1 and row.get("answerability_after") in {"insufficient_evidence", "partial_evidence"}),
        "reformulation_to_answerable_count": sum(1 for row in second_rounds if row.get("answerability_after") in {"answerable", "partially_answerable"}),
        "reformulation_still_insufficient_count": sum(1 for row in second_rounds if row.get("answerability_after") in {"insufficient_evidence", "partial_evidence"}),
        "reformulation_to_unanswerable_count": sum(1 for row in second_rounds if row.get("answerability_after") == "unanswerable"),
        "answerability_regression_count": sum(1 for row in second_rounds if row.get("answerability_before") in {"answerable", "partially_answerable"} and row.get("answerability_after") not in {"answerable", "partially_answerable"}),
    }


def cost_latency_report(*, reformulation_traces: list[dict[str, Any]], retrieval_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    reformulation_latencies = [int(((trace.get("provider") or {}).get("latency_ms")) or 0) for trace in reformulation_traces]
    latencies = [int(row.get("latency_ms") or 0) for row in retrieval_rounds] + reformulation_latencies
    input_tokens = [((trace.get("provider") or {}).get("input_tokens")) for trace in reformulation_traces]
    output_tokens = [((trace.get("provider") or {}).get("output_tokens")) for trace in reformulation_traces]
    token_sources = sorted(set(str(((trace.get("provider") or {}).get("token_source")) or "unavailable") for trace in reformulation_traces))
    token_source = token_sources[0] if len(token_sources) == 1 else ("unavailable" if not token_sources else "mixed")
    input_sum = sum(int(value or 0) for value in input_tokens)
    output_sum = sum(int(value or 0) for value in output_tokens)
    return {
        "schema_version": "opk-rag.task0067-cost-latency-report.v1",
        "pricing_source": "not_configured",
        "cost_status": "not_calculated",
        "token_source": token_source,
        "token_sources": token_sources,
        "total_input_tokens": input_sum,
        "total_output_tokens": output_sum,
        "total_tokens": input_sum + output_sum,
        "average_tokens_per_sample": (input_sum + output_sum) / len({trace.get("sample_id") for trace in reformulation_traces}) if reformulation_traces else 0,
        "average_tokens_per_reformulation": (input_sum + output_sum) / len(reformulation_traces) if reformulation_traces else 0,
        "total_reformulation_latency_ms": sum(reformulation_latencies),
        "average_reformulation_latency_ms": sum(reformulation_latencies) / len(reformulation_latencies) if reformulation_latencies else 0,
        "average_additional_latency_per_sample_ms": sum(reformulation_latencies) / len({trace.get("sample_id") for trace in reformulation_traces}) if reformulation_traces else 0,
        "total_latency_ms": sum(latencies),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_ms": median(latencies) if latencies else 0,
        "p95_latency_ms": _p95(latencies),
        "reformulation_call_count": len(reformulation_traces),
        "extra_search_call_count": sum(1 for row in retrieval_rounds if row.get("round_index") == 2),
    }


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]
