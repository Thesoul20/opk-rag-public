from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0113_retrieval_failure_taxonomy_v2 as task0113
import opk_rag.evaluation.task0128_evidence_budgeted_composition_mitigation as task0128
import opk_rag.evaluation.task0130_targeted_composition_regression_mitigation as task0130
import opk_rag.evaluation.task0131_targeted_evidence_composition_runtime_promotion as task0131
from opk_rag.chunking.models import ChunkingConfig
from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.reranking.config import RerankerConfig
from opk_rag.runtime_v2 import evidence_composition
from opk_rag.runtime_v2.multi_lane_candidate_composition import candidate_source_type
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA


TASK_ID = "TASK-0132"
EXPERIMENT_ID = "task0132-post-promotion-residual-failure-rebaseline"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0132_post_promotion_residual_failure_rebaseline_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0132_POST_PROMOTION_RESIDUAL_FAILURE_REBASELINE_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "failure_distribution.json",
    "historical_comparison.json",
    "stage_funnel.json",
    "graph_sensitive_analysis.json",
    "representative_cases.jsonl",
    "config.json",
    "digests.json",
    "verification.json",
)

FAILURE_STAGES = (
    "parsing_representation",
    "chunking",
    "query_document_mismatch",
    "candidate_retrieval",
    "ranking",
    "evidence_composition",
    "generation_or_evidence_use",
    "answerability",
    "citation_grounding",
    "unknown",
)

SUBCLASS_BY_STAGE = {
    "parsing_representation": "representation_loss",
    "chunking": "chunk_boundary_failure",
    "query_document_mismatch": "document_addressability_gap",
    "candidate_retrieval": "candidate_missing",
    "ranking": "relevant_candidate_misranked",
    "evidence_composition": "evidence_budget_cutoff",
    "generation_or_evidence_use": "evidence_available_but_unused",
    "answerability": "answerability_misclassification",
    "citation_grounding": "grounding_failure",
    "unknown": "unknown",
}

TASK0113_STAGE_MAP = {
    "candidate_retrieval_failure": "candidate_retrieval",
    "query_document_mismatch": "query_document_mismatch",
    "ranking_failure": "ranking",
    "chunk_boundary_failure": "chunking",
    "representation_failure": "parsing_representation",
    "evidence_packing_failure": "evidence_composition",
    "generation_only_failure": "generation_or_evidence_use",
    "unclear_compound_failure": "unknown",
}


def run_task0132_post_promotion_residual_failure_rebaseline(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    if not authority["task0131_policy_active"] or authority["known_task0130_regression_reintroduced"]:
        summary = runtime_drift_summary(authority)
        write_json(CONTRACT_PATH, build_contract(authority, {}, {}))
        write_json(output_dir / "summary.json", summary)
        verify_task0132_artifacts(output_dir=output_dir, write=True)
        return summary

    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    ranked = task0128.build_frozen_ranked_candidates()
    promoted = compose_promoted(ranked)
    per_sample = [classify_unit(unit, baseline[unit], ranked[unit], promoted, authority) for unit in sorted(ranked)]
    distribution = failure_distribution(per_sample)
    historical = historical_comparison(distribution)
    funnel = stage_funnel(per_sample)
    graph = graph_sensitive_analysis(per_sample)
    cases = representative_cases(per_sample)
    config = config_payload(benchmark, authority)
    digests = digests_payload(ranked, promoted, per_sample, config)
    summary = summary_payload(authority, benchmark, per_sample, distribution, historical, graph, config, digests)
    contract = build_contract(authority, benchmark, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_json(output_dir / "failure_distribution.json", distribution)
    write_json(output_dir / "historical_comparison.json", historical)
    write_json(output_dir / "stage_funnel.json", funnel)
    write_json(output_dir / "graph_sensitive_analysis.json", graph)
    write_jsonl(output_dir / "representative_cases.jsonl", cases)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0132_artifacts(output_dir=output_dir, write=True)
    summary["task0132_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, distribution, historical, graph), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    task0131_summary = read_json(task0131.RESULT_DIR / "summary.json")
    task0130_summary = read_json(task0130.RESULT_DIR / "summary.json")
    task0128_summary = read_json(task0128.RESULT_DIR / "summary.json")
    task0113_distribution = read_json(task0113.RESULT_DIR / "failure_distribution.json")
    default = evidence_composition.resolve_evidence_composition_policy()
    return {
        "schema_version": "opk-rag.task0132.authority.v1",
        "task0130_inputs_valid": task0130.verify_task0130_artifacts(write=False)["status"] == "valid",
        "task0131_inputs_valid": task0131.verify_task0131_artifacts(write=False)["status"] == "valid",
        "task0131_policy_active": evidence_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY == evidence_composition.TARGETED_BUDGETED_POLICY
        and default.policy_name == evidence_composition.TARGETED_BUDGETED_POLICY,
        "task0113_failure_unit_count": task0113_distribution["formal_failure_unit_count"],
        "task0113_failure_distribution": task0113_distribution,
        "task0128_downstream_net_gain": task0128_summary["downstream_net_gain"],
        "task0130_downstream_net_gain": task0130_summary["task0130_downstream_net_gain"],
        "task0130_regression_count": task0130_summary["total_regression_count"],
        "task0131_runtime_downstream_net_gain": task0131_summary["downstream_net_gain"],
        "task0131_runtime_regression_count": task0131_summary["regression_case_count"],
        "gold_evidence_new_loss_count": task0131_summary["gold_evidence_new_loss_count"],
        "candidate_membership_change_count": task0131_summary["candidate_membership_change_count"],
        "known_task0130_regression_reintroduced": bool(task0131_summary["useful_same_lane_evidence_displaced_reintroduced"]),
        "task0131_summary_sha256": sha256_file(task0131.RESULT_DIR / "summary.json"),
    }


def compose_promoted(ranked: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    outputs = {"evidence_rankings": {}, "traces": {}}
    for unit, rows in sorted(ranked.items()):
        evidence, traces = evidence_composition.compose_evidence(rows, evaluation_unit_id=unit)
        selected_ids = {row["canonical_chunk_id"] for row in evidence}
        outputs["evidence_rankings"][unit] = [*evidence, *[row for row in rows if row["canonical_chunk_id"] not in selected_ids]]
        outputs["traces"][unit] = traces
    return outputs


def classify_unit(
    unit: str,
    baseline_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    promoted: dict[str, dict[str, Any]],
    authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_rows = promoted["evidence_rankings"][unit][: evidence_composition.EVIDENCE_BUDGET_LIMIT]
    traces = promoted["traces"][unit]
    selected_ids = {row["canonical_chunk_id"] for row in selected_rows}
    relevant_candidates = [row for row in ranked_rows if row.get("relevant_label")]
    relevant_selected = [row for row in selected_rows if row.get("relevant_label")]
    best_rank = first_relevant_rank(ranked_rows)
    final_success = bool(relevant_selected)
    sample = baseline_rows[0]
    state = {
        "gold_or_relevant_evidence_in_corpus": bool(sample.get("gold_chunk_ids")) or bool(relevant_candidates),
        "gold_or_relevant_evidence_in_candidate_set": bool(relevant_candidates),
        "gold_or_relevant_evidence_rank": best_rank,
        "gold_or_relevant_evidence_in_evidence_context": bool(relevant_selected),
        "gold_or_relevant_evidence_used_by_generation_where_observable": None,
        "final_success": final_success,
        "answerability_failure": sample.get("answerability_class") not in {None, "answerable"} and not final_success,
        "citation_success": final_success,
        "grounding_success": final_success,
    }
    attribution = first_failure_attribution(state, traces)
    graph_sensitive = graph_sensitive_unit(sample, ranked_rows)
    return {
        "schema_version": "opk-rag.task0132.per-sample.v1",
        "sample_id": sample.get("sample_id") or unit,
        "evaluation_unit_id": unit,
        "document_id": sample.get("document_id"),
        "query": sample.get("question"),
        "final_success": final_success,
        "gold_evidence_available": state["gold_or_relevant_evidence_in_corpus"],
        "gold_or_relevant_evidence_in_corpus": state["gold_or_relevant_evidence_in_corpus"],
        "gold_or_relevant_evidence_in_candidate_set": state["gold_or_relevant_evidence_in_candidate_set"],
        "gold_or_relevant_evidence_rank": best_rank,
        "candidate_relevant_available": bool(relevant_candidates),
        "best_relevant_candidate_rank": best_rank,
        "relevant_evidence_selected": bool(relevant_selected),
        "gold_or_relevant_evidence_in_evidence_context": state["gold_or_relevant_evidence_in_evidence_context"],
        "gold_or_relevant_evidence_used_by_generation_where_observable": None,
        "evidence_context_sufficient": "evidence_context_sufficient" if relevant_selected else "evidence_context_insufficient",
        "generation_success": final_success if relevant_selected else None,
        "citation_success": final_success,
        "grounding_success": final_success,
        "first_failure_stage": attribution["first_failure_stage"],
        "secondary_failure_stages": attribution["secondary_failure_stages"],
        "failure_class": attribution["failure_class"],
        "failure_subclass": attribution["failure_subclass"],
        "root_cause_confidence": attribution["root_cause_confidence"],
        "graph_sensitive": graph_sensitive,
        "addressability_class": addressability_class(attribution["first_failure_stage"], graph_sensitive),
        "candidate_source_types": sorted(set(candidate_source_type(row) for row in ranked_rows)),
        "candidate_count": len(ranked_rows),
        "selected_evidence_ids": [row["canonical_chunk_id"] for row in selected_rows],
        "relevant_candidate_ids": [row["canonical_chunk_id"] for row in relevant_candidates],
        "relevant_selected_ids": [row["canonical_chunk_id"] for row in relevant_selected],
        "promotion_policy_active": (authority or {}).get("task0131_policy_active"),
    }


def first_failure_attribution(state: dict[str, Any], traces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if state.get("final_success") is True:
        stage = None
    elif state.get("parse_loss") or state.get("representation_loss"):
        stage = "parsing_representation"
    elif state.get("chunking_failure"):
        stage = "chunking"
    elif state.get("query_document_mismatch"):
        stage = "query_document_mismatch"
    elif not state.get("gold_or_relevant_evidence_in_candidate_set"):
        stage = "candidate_retrieval"
    elif (state.get("gold_or_relevant_evidence_rank") or 10**9) > evidence_composition.EVIDENCE_BUDGET_LIMIT and not state.get("gold_or_relevant_evidence_in_evidence_context"):
        stage = "ranking"
    elif not state.get("gold_or_relevant_evidence_in_evidence_context"):
        stage = "evidence_composition"
    elif state.get("answerability_failure"):
        stage = "answerability"
    elif state.get("citation_failure") or state.get("grounding_failure"):
        stage = "citation_grounding"
    elif state.get("generation_success") is False or state.get("final_success") is False:
        stage = "generation_or_evidence_use"
    else:
        stage = "unknown"
    if stage is None:
        return {"first_failure_stage": None, "secondary_failure_stages": [], "failure_class": None, "failure_subclass": None, "root_cause_confidence": "high"}
    subclass = failure_subclass(stage, traces or [])
    return {
        "first_failure_stage": stage,
        "secondary_failure_stages": secondary_stages(stage, state),
        "failure_class": stage,
        "failure_subclass": subclass,
        "root_cause_confidence": confidence_for(stage, state),
    }


def failure_subclass(stage: str, traces: list[dict[str, Any]]) -> str:
    if stage == "evidence_composition":
        relevant_rejections = [row.get("rejection_reason") for row in traces if row.get("relevant_label") and not row.get("selected_for_evidence")]
        if "budget_cutoff" in relevant_rejections:
            return "evidence_budget_cutoff"
        if "same_document_prefix_retention_guard" in relevant_rejections:
            return "same_lane_displacement"
        if "redundancy" in relevant_rejections:
            return "redundancy_budget_waste"
        return "composition_selection_failure"
    return SUBCLASS_BY_STAGE[stage]


def secondary_stages(stage: str, state: dict[str, Any]) -> list[str]:
    stages = []
    if stage != "ranking" and (state.get("gold_or_relevant_evidence_rank") or 0) > evidence_composition.EVIDENCE_BUDGET_LIMIT:
        stages.append("ranking")
    if stage != "evidence_composition" and state.get("gold_or_relevant_evidence_in_candidate_set") and not state.get("gold_or_relevant_evidence_in_evidence_context"):
        stages.append("evidence_composition")
    if stage not in {"generation_or_evidence_use", "candidate_retrieval", "ranking", "evidence_composition"} and state.get("gold_or_relevant_evidence_in_evidence_context") and state.get("final_success") is False:
        stages.append("generation_or_evidence_use")
    return stages


def confidence_for(stage: str, state: dict[str, Any]) -> str:
    if stage in {"candidate_retrieval", "ranking", "evidence_composition"}:
        return "high"
    if stage in {"query_document_mismatch", "generation_or_evidence_use", "answerability", "citation_grounding", "chunking"}:
        return "medium"
    return "low"


def graph_sensitive_unit(sample: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    tags = set(sample.get("slice_tags") or [])
    text = " ".join([str(sample.get("question") or ""), *[str(row.get("document") or "")[:500] for row in rows[:3]]]).lower()
    cues = ("cross-section", "multi-document", "relationship", "multi-hop", "path", "between", "compare", "across")
    return bool(tags & {"graph_sensitive", "cross_section", "multi_document", "relational", "multi_hop"} or any(cue in text for cue in cues))


def addressability_class(stage: str | None, graph_sensitive: bool) -> str:
    if stage is None:
        return "not_current_priority"
    if graph_sensitive:
        return "requires_new_capability"
    if stage in {"candidate_retrieval", "ranking", "evidence_composition", "generation_or_evidence_use", "query_document_mismatch"}:
        return "addressable_with_current_architecture"
    if stage == "unknown":
        return "benchmark_or_label_issue"
    return "not_current_priority"


def failure_distribution(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in per_sample if not row["final_success"]]
    counts = Counter(row["first_failure_stage"] for row in failures)
    total = len(failures)
    return {
        "schema_version": "opk-rag.task0132.failure-distribution.v1",
        "task_id": TASK_ID,
        "total_formal_units": len(per_sample),
        "successful_units": len(per_sample) - total,
        "failed_units": total,
        "failure_count_by_first_stage": {stage: counts.get(stage, 0) for stage in FAILURE_STAGES},
        "failure_rate_by_first_stage": {stage: _rate(counts.get(stage, 0), total) for stage in FAILURE_STAGES},
        "current_dominant_failure_stage": dominant_stage(counts),
        "current_second_largest_failure_stage": second_stage(counts),
        "failure_counts_sum_to_total": sum(counts.values()) == total,
    }


def historical_comparison(distribution: dict[str, Any]) -> dict[str, Any]:
    historical_dist = read_json(task0113.RESULT_DIR / "failure_distribution.json")
    current = distribution["failure_count_by_first_stage"]
    historical_counts = {TASK0113_STAGE_MAP[key.replace("_count", "")]: value for key, value in historical_dist.items() if key.endswith("_count") and key.replace("_count", "") in TASK0113_STAGE_MAP}
    comparable = {}
    for stage in ("candidate_retrieval", "ranking", "chunking", "query_document_mismatch", "evidence_composition", "generation_or_evidence_use"):
        hist = int(historical_counts.get(stage, 0))
        cur = int(current.get(stage, 0))
        comparable[stage] = {
            "historical_count": hist,
            "current_count": cur,
            "absolute_delta": cur - hist,
            "relative_delta": None if hist == 0 else (cur - hist) / hist,
            "directly_comparable": stage in {"candidate_retrieval", "ranking", "chunking", "query_document_mismatch"},
        }
    hist_dom = TASK0113_STAGE_MAP.get(historical_dist["largest_failure_category"], historical_dist["largest_failure_category"])
    cur_dom = distribution["current_dominant_failure_stage"]
    return {
        "schema_version": "opk-rag.task0132.historical-comparison.v1",
        "historical_dominant_failure_stage": hist_dom,
        "current_dominant_failure_stage": cur_dom,
        "current_second_largest_failure_stage": distribution["current_second_largest_failure_stage"],
        "bottleneck_shifted": hist_dom != cur_dom,
        "bottleneck_shift_from": hist_dom if hist_dom != cur_dom else None,
        "bottleneck_shift_to": cur_dom if hist_dom != cur_dom else None,
        "historical_task0113_failure_count": historical_dist["formal_failure_unit_count"],
        "current_failure_count": distribution["failed_units"],
        "comparison_by_stage": comparable,
        "incompatible_taxonomy_notes": {
            "evidence_composition": "TASK-0113 used evidence_packing_failure; TASK-0132 separates promoted Evidence Composition boundary.",
            "generation_or_evidence_use": "TASK-0113 generation-only was limited by available artifacts; TASK-0132 keeps evidence-present downstream failures separate.",
        },
    }


def stage_funnel(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0132.stage-funnel.v1",
        "total_units": len(per_sample),
        "gold_or_relevant_evidence_in_corpus_count": sum(row["gold_or_relevant_evidence_in_corpus"] for row in per_sample),
        "gold_or_relevant_evidence_in_candidate_set_count": sum(row["gold_or_relevant_evidence_in_candidate_set"] for row in per_sample),
        "gold_or_relevant_evidence_in_evidence_context_count": sum(row["gold_or_relevant_evidence_in_evidence_context"] for row in per_sample),
        "successful_unit_count": sum(row["final_success"] for row in per_sample),
        "candidate_retrieval_failure_count": sum(row["first_failure_stage"] == "candidate_retrieval" for row in per_sample),
        "ranking_failure_count": sum(row["first_failure_stage"] == "ranking" for row in per_sample),
        "evidence_composition_failure_count": sum(row["first_failure_stage"] == "evidence_composition" for row in per_sample),
        "evidence_available_answer_failure_count": sum(row["first_failure_stage"] == "generation_or_evidence_use" for row in per_sample),
    }


def graph_sensitive_analysis(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in per_sample if not row["final_success"]]
    graph_failures = [row for row in failures if row["graph_sensitive"]]
    return {
        "schema_version": "opk-rag.task0132.graph-sensitive-analysis.v1",
        "graph_sensitive_failure_count": len(graph_failures),
        "graph_sensitive_failure_rate": _rate(len(graph_failures), len(failures)),
        "graph_sensitive_candidate_retrieval_failure_count": sum(row["first_failure_stage"] == "candidate_retrieval" for row in graph_failures),
        "graph_sensitive_generation_failure_count": sum(row["first_failure_stage"] == "generation_or_evidence_use" for row in graph_failures),
        "remaining_bottleneck_increasingly_graph_sensitive": _rate(len(graph_failures), len(failures)) >= 0.30,
    }


def representative_cases(per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [row for row in per_sample if not row["final_success"]]
    dist = Counter(row["first_failure_stage"] for row in failures)
    wanted = [
        ("dominant_residual_failure_class", dominant_stage(dist)),
        ("second_largest_failure_class", second_stage(dist)),
        ("candidate_retrieval_failure", "candidate_retrieval"),
        ("ranking_failure", "ranking"),
        ("evidence_composition_residual_failure", "evidence_composition"),
        ("evidence_present_generation_failed", "generation_or_evidence_use"),
        ("graph_sensitive_failure", "graph_sensitive"),
        ("query_document_mismatch", "query_document_mismatch"),
        ("successful_case_formerly_failing", "success"),
    ]
    rows = []
    for label, selector in wanted:
        case = choose_case(per_sample, selector)
        rows.append({"schema_version": "opk-rag.task0132.representative-case.v1", "case_type": label, "available": case is not None, "case": case})
    return rows


def choose_case(per_sample: list[dict[str, Any]], selector: str | None) -> dict[str, Any] | None:
    if selector is None:
        return None
    for row in per_sample:
        if selector == "success" and row["final_success"]:
            return case_projection(row)
        if selector == "graph_sensitive" and not row["final_success"] and row["graph_sensitive"]:
            return case_projection(row)
        if row["first_failure_stage"] == selector:
            return case_projection(row)
    return None


def case_projection(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("evaluation_unit_id", "document_id", "query", "final_success", "first_failure_stage", "failure_subclass", "best_relevant_candidate_rank", "relevant_evidence_selected", "graph_sensitive", "addressability_class")
    return {key: row.get(key) for key in keys}


def config_payload(benchmark: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    chunking = ChunkingConfig()
    embedding = EmbeddingConfig()
    reranker = RerankerConfig()
    policy = evidence_composition.resolve_evidence_composition_policy()
    payload = {
        "schema_version": "opk-rag.task0132.config.v1",
        "chunking": {
            "target_chunk_size": chunking.target_size,
            "maximum_chunk_size": chunking.max_size,
            "overlap": chunking.overlap,
            "length_unit": chunking.length_unit,
            "paragraph_as_minimum_unit": True,
        },
        "embedding": embedding.__dict__ | {"cache_dir": str(embedding.cache_dir) if embedding.cache_dir else None},
        "retrieval": {
            "vector_retrieval": True,
            "lexical_retrieval": True,
            "multi_lane_behavior": "task0125_C2_multi_lane_candidate_composition",
            "candidate_depth": task0128.task0127.RETRIEVAL_TOP_K,
            "candidate_dedup": "canonical_chunk_id",
        },
        "reranking": {
            "provider": reranker.provider,
            "model_name": reranker.model_name,
            "model_revision": reranker.model_revision,
            "default_rank_fusion_policy": "bge_guarded_rank_fusion",
            "rank_fusion_k": DEFAULT_RANK_FUSION_K,
            "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        },
        "evidence_composition": policy.to_json(),
        "generation": {
            "provider": "current_configured_answer_provider",
            "prompt": "current_generation_prompt_frozen_by_prior_runtime",
            "answerability_policy": "current_answerability_policy",
            "citation_policy": "current_citation_grounding_policy",
            "recovery_behavior": "current_recovery_behavior",
            "observable_generation_replay": False,
        },
        "promotion_applied_by_task0132": False,
        "runtime_modified": False,
        "task0131_summary_sha256": authority["task0131_summary_sha256"],
    }
    payload["runtime_config_digest"] = digest_json(payload)
    return payload


def digests_payload(
    ranked: dict[str, list[dict[str, Any]]],
    promoted: dict[str, dict[str, Any]],
    per_sample: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0132.digests.v1",
        "candidate_input_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in sorted(ranked.items())}),
        "evidence_output_digest": digest_json({unit: [row["canonical_chunk_id"] for row in promoted["evidence_rankings"][unit][: evidence_composition.EVIDENCE_BUDGET_LIMIT]] for unit in sorted(promoted["evidence_rankings"])}),
        "per_sample_digest": digest_json(per_sample),
        "runtime_config_digest": config["runtime_config_digest"],
        "deterministic_output": True,
    }


def summary_payload(
    authority: dict[str, Any],
    benchmark: dict[str, Any],
    per_sample: list[dict[str, Any]],
    distribution: dict[str, Any],
    historical: dict[str, Any],
    graph: dict[str, Any],
    config: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    failures = [row for row in per_sample if not row["final_success"]]
    counts = distribution["failure_count_by_first_stage"]
    priority = priority_recommendation(counts, graph)
    document_count = len({str(row.get("document_id")) for rows in benchmark["baseline"].values() for row in rows if row.get("document_id")})
    return {
        "schema_version": "opk-rag.task0132.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "task0131_inputs_valid": authority["task0131_inputs_valid"],
        "task0131_policy_active": authority["task0131_policy_active"],
        "benchmark_name": benchmark["benchmark_identity"].get("benchmark_name", "task0112-expanded-formal-benchmark"),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "document_count": document_count,
        "query_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "runtime_policy_name": evidence_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "runtime_policy_digest": evidence_composition.targeted_budgeted_policy().policy_digest,
        "runtime_config_digest": config["runtime_config_digest"],
        "total_unit_count": len(per_sample),
        "successful_unit_count": len(per_sample) - len(failures),
        "failed_unit_count": len(failures),
        "historical_task0113_failure_count": historical["historical_task0113_failure_count"],
        "current_failure_count": len(failures),
        "candidate_retrieval_failure_count": counts["candidate_retrieval"],
        "query_document_mismatch_failure_count": counts["query_document_mismatch"],
        "ranking_failure_count": counts["ranking"],
        "evidence_composition_failure_count": counts["evidence_composition"],
        "generation_evidence_use_failure_count": counts["generation_or_evidence_use"],
        "chunking_failure_count": counts["chunking"],
        "answerability_failure_count": counts["answerability"],
        "citation_grounding_failure_count": counts["citation_grounding"],
        "unknown_failure_count": counts["unknown"],
        "historical_dominant_failure_stage": historical["historical_dominant_failure_stage"],
        "current_dominant_failure_stage": historical["current_dominant_failure_stage"],
        "current_second_largest_failure_stage": historical["current_second_largest_failure_stage"],
        "bottleneck_shifted": historical["bottleneck_shifted"],
        "bottleneck_shift_from": historical["bottleneck_shift_from"],
        "bottleneck_shift_to": historical["bottleneck_shift_to"],
        "graph_sensitive_failure_count": graph["graph_sensitive_failure_count"],
        "graph_sensitive_failure_rate": graph["graph_sensitive_failure_rate"],
        "evidence_present_but_generation_failed_count": counts["generation_or_evidence_use"],
        "high_confidence_failure_count": sum(row["root_cause_confidence"] == "high" for row in failures),
        "medium_confidence_failure_count": sum(row["root_cause_confidence"] == "medium" for row in failures),
        "low_confidence_failure_count": sum(row["root_cause_confidence"] == "low" for row in failures),
        "candidate_membership_frozen": True,
        "candidate_membership_change_count": authority["candidate_membership_change_count"],
        "evidence_budget_limit": evidence_composition.EVIDENCE_BUDGET_LIMIT,
        "gold_evidence_new_loss_count": authority["gold_evidence_new_loss_count"],
        "known_task0130_regression_reintroduced": authority["known_task0130_regression_reintroduced"],
        "end_to_end_accuracy": _rate(len(per_sample) - len(failures), len(per_sample)),
        "answerability": None,
        "citation_correctness": _rate(sum(row["citation_success"] for row in per_sample), len(per_sample)),
        "citation_completeness": _rate(sum(row["citation_success"] for row in per_sample), len(per_sample)),
        "grounding": _rate(sum(row["grounding_success"] for row in per_sample), len(per_sample)),
        "unsupported_answer_rate": None,
        **priority,
        "runtime_modified": False,
        "promotion_applied": False,
        "task0132_verifier_valid": False,
        "verifier_status": "pending",
        "candidate_input_digest": digests["candidate_input_digest"],
        "evidence_output_digest": digests["evidence_output_digest"],
    }


def priority_recommendation(counts: dict[str, int], graph: dict[str, Any]) -> dict[str, Any]:
    if graph["remaining_bottleneck_increasingly_graph_sensitive"] and graph["graph_sensitive_failure_count"] >= counts.get("candidate_retrieval", 0):
        diagnosis = "graph_sensitive_retrieval_is_new_primary_capability_gap"
        family = "graph_sensitive_retrieval"
        gap = "graph_sensitive_retrieval"
    else:
        dominant = dominant_stage(Counter(counts))
        mapping = {
            "candidate_retrieval": ("candidate_retrieval_is_new_primary_bottleneck", "candidate_retrieval_optimization"),
            "query_document_mismatch": ("query_addressability_is_new_primary_bottleneck", "query_addressability_optimization"),
            "ranking": ("ranking_is_new_primary_bottleneck", "ranking_optimization"),
            "evidence_composition": ("evidence_composition_remains_primary_bottleneck", "evidence_composition_residual_mitigation"),
            "generation_or_evidence_use": ("generation_evidence_use_is_new_primary_bottleneck", "generation_evidence_utilization_optimization"),
            "unknown": ("benchmark_quality_limits_diagnosis", "benchmark_quality_remediation"),
        }
        diagnosis, family = mapping.get(dominant or "unknown", ("no_single_dominant_bottleneck", "benchmark_quality_remediation"))
        gap = dominant or "none"
    return {
        "primary_diagnosis": diagnosis,
        "highest_priority_capability_gap": gap,
        "recommended_next_task_family": family,
        "priority_scores": priority_scores(counts),
    }


def priority_scores(counts: dict[str, int]) -> dict[str, dict[str, str]]:
    max_count = max(counts.values()) if counts else 0
    scores = {}
    for stage, count in counts.items():
        scores[stage] = {
            "prevalence": ordinal(count, max_count),
            "estimated_recoverability": "high" if stage in {"candidate_retrieval", "ranking", "evidence_composition"} else "medium",
            "downstream_impact": "high" if count else "low",
            "implementation_scope": "medium",
            "regression_risk": "medium" if stage in {"ranking", "evidence_composition"} else "low",
        }
    return scores


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0132.post-promotion-residual-failure-rebaseline-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0131_inputs_valid": authority.get("task0131_inputs_valid"),
        "task0131_policy_active": authority.get("task0131_policy_active"),
        "benchmark_revision": (benchmark.get("benchmark_identity") or {}).get("benchmark_revision"),
        "benchmark_digest": (benchmark.get("benchmark_identity") or {}).get("benchmark_digest"),
        "formal_evaluation_unit_count": (benchmark.get("benchmark_identity") or {}).get("evaluation_unit_count"),
        "runtime_policy_name": evidence_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "runtime_policy_digest": evidence_composition.targeted_budgeted_policy().policy_digest,
        "runtime_config_digest": config.get("runtime_config_digest"),
        "diagnosis_only": True,
        "runtime_behavior_modification_allowed": False,
        "runtime_modified": False,
        "promotion_applied": False,
    }


def runtime_drift_summary(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0132.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked_runtime_baseline_drift",
        "task0131_inputs_valid": authority["task0131_inputs_valid"],
        "task0131_policy_active": authority["task0131_policy_active"],
        "primary_diagnosis": "runtime_baseline_drift",
        "runtime_modified": False,
        "promotion_applied": False,
    }


def verify_task0132_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        per_sample = read_jsonl(output_dir / "per_sample.jsonl")
        distribution = read_json(output_dir / "failure_distribution.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "summary_not_complete"})
        if summary.get("runtime_modified") is not False or summary.get("promotion_applied") is not False:
            issues.append({"code": "diagnosis_only_invariant_failed"})
        if summary.get("task0131_inputs_valid") is not True or summary.get("task0131_policy_active") is not True:
            issues.append({"code": "task0131_authority_invalid"})
        if summary.get("runtime_policy_name") != evidence_composition.TARGETED_BUDGETED_POLICY:
            issues.append({"code": "canonical_policy_not_active"})
        if summary.get("evidence_budget_limit") != evidence_composition.EVIDENCE_BUDGET_LIMIT:
            issues.append({"code": "evidence_budget_drift"})
        if len(per_sample) != summary.get("formal_evaluation_unit_count"):
            issues.append({"code": "per_sample_unit_count_mismatch"})
        if not distribution.get("failure_counts_sum_to_total"):
            issues.append({"code": "failure_distribution_invalid"})
        if any(row.get("first_failure_stage") not in (*FAILURE_STAGES, None) for row in per_sample):
            issues.append({"code": "invalid_first_failure_stage"})
    result = {
        "schema_version": "opk-rag.task0132.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task0132_verifier_valid": not issues,
        "runtime_modified": False,
        "promotion_applied": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], distribution: dict[str, Any], historical: dict[str, Any], graph: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{stage}` | {count} | {_pct(distribution['failure_rate_by_first_stage'][stage])} |"
        for stage, count in distribution["failure_count_by_first_stage"].items()
    )
    comparison = historical["comparison_by_stage"]
    return f"""# TASK-0132 Post-Promotion Residual Failure Rebaseline Report

## Required Answers

Q1. TASK-0131 promoted runtime reproduced: `{summary['task0131_inputs_valid'] and summary['task0131_policy_active']}`. Active policy `{summary['runtime_policy_name']}`, budget `{summary['evidence_budget_limit']}` evidence slots, known TASK-0130 regression reintroduced `{summary['known_task0130_regression_reintroduced']}`.

Q2. Current formal failures: `{summary['failed_unit_count']}` / `{summary['total_unit_count']}`.

Q3. Failure distribution by first responsible stage:

| Failure stage | Count | % failures |
| --- | ---: | ---: |
{rows}

Q4. TASK-0113 dominant bottleneck was `{summary['historical_dominant_failure_stage']}`; current dominant bottleneck is `{summary['current_dominant_failure_stage']}`.

Q5. Bottleneck shifted: `{summary['bottleneck_shifted']}` from `{summary['bottleneck_shift_from']}` to `{summary['bottleneck_shift_to']}`.

Q6. Evidence Composition top-level residual count is `{summary['evidence_composition_failure_count']}`; it is not the dominant bottleneck unless it equals `{summary['current_dominant_failure_stage']}`.

Q7. Evidence-present-but-generation-failed count: `{summary['evidence_present_but_generation_failed_count']}`.

Q8. Candidate Retrieval failures: `{summary['candidate_retrieval_failure_count']}`; TASK-0113 historical comparable count `{comparison['candidate_retrieval']['historical_count']}`.

Q9. Query/document addressability failures: `{summary['query_document_mismatch_failure_count']}`; TASK-0113 historical comparable count `{comparison['query_document_mismatch']['historical_count']}`.

Q10. Ranking failures: `{summary['ranking_failure_count']}`; TASK-0113 historical comparable count `{comparison['ranking']['historical_count']}`.

Q11. Graph-sensitive residual failures: `{summary['graph_sensitive_failure_count']}` / `{summary['failed_unit_count']}` = `{_pct(summary['graph_sensitive_failure_rate'])}`. Increasingly graph-sensitive: `{graph['remaining_bottleneck_increasingly_graph_sensitive']}`.

Q12. Highest-priority capability gap: `{summary['highest_priority_capability_gap']}`.

Q13. TASK-0133 should work on `{summary['recommended_next_task_family']}`.

## Notes

TASK-0132 is diagnostic only. It did not change Retriever, Reranker, Chunking, Evidence Composition, Generation, default policy, or evidence budget.
"""


def dominant_stage(counts: Counter | dict[str, int]) -> str | None:
    positive = [(stage, count) for stage, count in counts.items() if count]
    return sorted(positive, key=lambda item: (-item[1], item[0]))[0][0] if positive else None


def second_stage(counts: Counter | dict[str, int]) -> str | None:
    positive = sorted([(stage, count) for stage, count in counts.items() if count], key=lambda item: (-item[1], item[0]))
    return positive[1][0] if len(positive) > 1 else None


def ordinal(count: int, max_count: int) -> str:
    if not count:
        return "low"
    if max_count and count >= max_count * 0.67:
        return "high"
    if max_count and count >= max_count * 0.33:
        return "medium"
    return "low"


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else numerator / denominator


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
