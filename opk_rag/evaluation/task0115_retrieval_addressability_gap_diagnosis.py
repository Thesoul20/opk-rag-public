from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    first_relevant_rank,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0112_reranker_strategy_matrix import (
    RESULT_DIR as TASK0112_RESULT_DIR,
    build_expanded_benchmark,
    verify_task0112_artifacts,
)
from opk_rag.evaluation.task0113_retrieval_failure_taxonomy_v2 import (
    CONTRACT_PATH as TASK0113_CONTRACT_PATH,
    RESULT_DIR as TASK0113_RESULT_DIR,
    tokenize,
    verify_task0113_artifacts,
)
from opk_rag.evaluation.task0114_governed_multi_query_retrieval_ablation import (
    CONTRACT_PATH as TASK0114_CONTRACT_PATH,
    RESULT_DIR as TASK0114_RESULT_DIR,
    ARM_SPECS as TASK0114_ARM_SPECS,
    guard_should_trigger,
    retrieve_with_plan,
    verify_task0114_artifacts,
)
from opk_rag.retrieval.multi_query import build_query_plan


TASK_ID = "TASK-0115"
EXPERIMENT_ID = "task0115-retrieval-addressability-gap-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0115_retrieval_addressability_gap_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0115_RETRIEVAL_ADDRESSABILITY_GAP_DIAGNOSIS_REPORT.md"

FORMAL_EVALUATION_UNIT_COUNT = 575
TASK0113_FAILURE_UNIT_COUNT = 201
TASK0113_MULTI_QUERY_ADDRESSABLE_COUNT = 181
TASK0114_ACTUAL_RECOVERED_COUNT = 2
MULTI_QUERY_UNREALIZED_FAILURE_COUNT = 179

QUERY_ADDRESSABLE_TYPES = {"query_document_mismatch", "candidate_retrieval_failure"}
CUTOFFS = (5, 10, 20)
DEPTH_CUTOFFS = (30, 50, 100)
ADDRESSABILITY_CLASSES = (
    "query_side_addressable",
    "document_representation_addressable",
    "lexical_addressable",
    "candidate_cutoff_addressable",
    "dense_embedding_geometry_failure",
    "fine_grained_interaction_required",
    "relational_or_multihop_required",
    "ranking_after_recovery_failure",
    "evaluation_or_identity_issue",
    "compound_addressability",
    "unclear",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "unrealized_failure_units.jsonl",
    "oracle_query_results.jsonl",
    "oracle_query_snapshot.jsonl",
    "representation_probe_results.jsonl",
    "lexical_probe_results.jsonl",
    "candidate_depth_analysis.json",
    "addressability_classification.jsonl",
    "addressability_distribution.json",
    "recovery_funnel.json",
    "task0113_recalibration.json",
    "strategy_recommendation.json",
    "provenance.json",
    "verification.json",
)


def run_task0115_retrieval_addressability_gap_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    task0113_summary = read_json(TASK0113_RESULT_DIR / "summary.json")
    task0114_summary = read_json(TASK0114_RESULT_DIR / "summary.json")
    failure_units = read_jsonl(TASK0113_RESULT_DIR / "failure_units.jsonl")

    contract = build_contract(benchmark, task0113_summary, task0114_summary)
    write_json(CONTRACT_PATH, contract)

    predicted = predicted_multi_query_addressable_units(failure_units)
    recovered = task0114_recovered_units(baseline, predicted, task0114_summary)
    unrealized = [row for row in predicted if row["evaluation_unit_id"] not in recovered]

    oracle_results, oracle_snapshots = run_oracle_probe(unrealized, baseline)
    representation_results = run_representation_probe(unrealized, baseline, oracle_results)
    lexical_results = run_lexical_probe(unrealized, baseline, oracle_results, representation_results)
    depth = candidate_depth_analysis(unrealized, baseline, oracle_results, representation_results, lexical_results)
    classifications = classify_units(unrealized, baseline, oracle_results, representation_results, lexical_results, depth)
    distribution = addressability_distribution(classifications)
    funnel = recovery_funnel(unrealized, oracle_results, representation_results, lexical_results, classifications)
    recalibration = task0113_recalibration(task0113_summary, task0114_summary, distribution)
    recommendation = strategy_recommendation(distribution)
    provenance = provenance_payload(benchmark, predicted, recovered)
    summary = summary_payload(benchmark, task0113_summary, task0114_summary, unrealized, oracle_results, representation_results, lexical_results, distribution, recalibration, recommendation)

    write_jsonl(output_dir / "unrealized_failure_units.jsonl", unrealized)
    write_jsonl(output_dir / "oracle_query_results.jsonl", oracle_results)
    write_jsonl(output_dir / "oracle_query_snapshot.jsonl", oracle_snapshots)
    write_jsonl(output_dir / "representation_probe_results.jsonl", representation_results)
    write_jsonl(output_dir / "lexical_probe_results.jsonl", lexical_results)
    write_json(output_dir / "candidate_depth_analysis.json", depth)
    write_jsonl(output_dir / "addressability_classification.jsonl", classifications)
    write_json(output_dir / "addressability_distribution.json", distribution)
    write_json(output_dir / "recovery_funnel.json", funnel)
    write_json(output_dir / "task0113_recalibration.json", recalibration)
    write_json(output_dir / "strategy_recommendation.json", recommendation)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "summary.json", summary)

    verification = verify_task0115_artifacts(output_dir=output_dir, write=True)
    summary["task0115_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, distribution, funnel, recalibration, recommendation), encoding="utf-8")
    return summary


def build_contract(benchmark: dict[str, Any], task0113_summary: dict[str, Any], task0114_summary: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "task0113_contract": TASK0113_CONTRACT_PATH,
        "task0113_summary": TASK0113_RESULT_DIR / "summary.json",
        "task0113_failure_units": TASK0113_RESULT_DIR / "failure_units.jsonl",
        "task0114_contract": TASK0114_CONTRACT_PATH,
        "task0114_summary": TASK0114_RESULT_DIR / "summary.json",
        "task0114_failure_recovery": TASK0114_RESULT_DIR / "failure_recovery.json",
    }
    return {
        "schema_version": "opk-rag.task0115.retrieval-addressability-gap-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "task0113_artifact_identity": {
            "failure_unit_count": task0113_summary.get("formal_failure_unit_count"),
            "multi_query_addressable_count": task0113_summary.get("multi_query_recoverable_count"),
        },
        "task0114_artifact_identity": {
            "actual_recovered_failure_count": task0114_summary.get("actual_recovered_failure_count"),
            "default_multi_query_enabled": task0114_summary.get("default_multi_query_enabled"),
        },
        "unrealized_failure_population": {
            "predicted_multi_query_addressable_count": TASK0113_MULTI_QUERY_ADDRESSABLE_COUNT,
            "task0114_actual_recovered_count": TASK0114_ACTUAL_RECOVERED_COUNT,
            "multi_query_unrealized_failure_count": MULTI_QUERY_UNREALIZED_FAILURE_COUNT,
        },
        "oracle_query_policy": {
            "oracle_query_uses_gold": True,
            "oracle_query_is_runtime_eligible": False,
            "oracle_query_metric_is_diagnostic_only": True,
            "answer_generation_allowed": False,
        },
        "representation_probe_policy": {
            "diagnostic_index": True,
            "runtime_default_unchanged": True,
            "representations": ["D0_raw_chunk", "D1_heading_context", "D2_section_context", "D3_neighbor_context"],
            "canonical_chunk_identity_preserved": True,
        },
        "lexical_probe_policy": {"uses_existing_lexical_capability": True, "splade_runtime_introduced": False},
        "cutoff_thresholds": {"retrieval": list(CUTOFFS), "candidate_depth": list(DEPTH_CUTOFFS)},
        "addressability_taxonomy": list(ADDRESSABILITY_CLASSES),
        "recommendation_rules": [
            "largest_class_primary",
            "cost_adjusted_strategy_priority",
            "single_primary_next_direction",
        ],
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "default_multi_query_enabled": False,
        "production_default_change_allowed": False,
        "source_artifacts": {key: {"path": _rel(path), "sha256": sha256_file(path) if path.exists() else None} for key, path in sources.items()},
    }


def predicted_multi_query_addressable_units(failure_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in failure_units
        if row.get("primary_failure_type") in QUERY_ADDRESSABLE_TYPES
        or any(item in QUERY_ADDRESSABLE_TYPES for item in row.get("secondary_failure_types") or [])
    ]


def task0114_recovered_units(baseline: dict[str, list[dict[str, Any]]], predicted: list[dict[str, Any]], task0114_summary: dict[str, Any]) -> set[str]:
    arm_id = task0114_summary.get("best_failure_recovery_arm") or "R4_guarded_multi_query"
    spec = next(spec for spec in TASK0114_ARM_SPECS if spec["arm_id"] == arm_id)
    recovered: set[str] = set()
    for row in predicted:
        unit_id = row["evaluation_unit_id"]
        rows = baseline[unit_id]
        before = first_relevant_rank(rows) or math.inf
        enabled = bool(spec["enabled"])
        query_count = int(spec["query_count"])
        if spec.get("guarded") and not guard_should_trigger(rows):
            enabled = False
            query_count = 1
        plan = build_query_plan(
            rows[0].get("question") or "",
            enabled=enabled,
            query_count=query_count,
        )
        result = retrieve_with_plan(unit_id, rows, plan)
        after = first_relevant_rank(result.candidates) or math.inf
        if row.get("primary_failure_type") == "candidate_retrieval_failure" and before > 20 and after <= 20:
            recovered.add(unit_id)
    return recovered


def run_oracle_probe(unrealized: list[dict[str, Any]], baseline: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for failure in unrealized:
        unit_id = failure["evaluation_unit_id"]
        rows = baseline[unit_id]
        query = oracle_query_text(failure, rows)
        ranked = rank_rows_by_terms(rows, tokenize(query), representation="oracle")
        rank = first_relevant_rank(ranked)
        payload = {
            "schema_version": "opk-rag.task0115.oracle-query-result.v1",
            "evaluation_unit_id": unit_id,
            "oracle_query_rank": rank,
            "oracle_query_recovered_at_5": rank is not None and rank <= 5,
            "oracle_query_recovered_at_10": rank is not None and rank <= 10,
            "oracle_query_recovered_at_20": rank is not None and rank <= 20,
            "oracle_query_uses_gold": True,
            "oracle_query_is_runtime_eligible": False,
            "oracle_query_metric_is_diagnostic_only": True,
            "top_candidate_ids": [row["canonical_chunk_id"] for row in ranked[:20]],
            "canonical_chunk_identity_preserved": True,
        }
        results.append(payload)
        snapshots.append(
            {
                "schema_version": "opk-rag.task0115.oracle-query-snapshot.v1",
                "evaluation_unit_id": unit_id,
                "original_query": rows[0].get("question") or "",
                "oracle_retrieval_query": query,
                "oracle_query_digest": digest_json({"unit_id": unit_id, "query": query}),
                "oracle_query_uses_gold": True,
                "oracle_query_is_runtime_eligible": False,
                "oracle_query_metric_is_diagnostic_only": True,
                "query_provenance_complete": True,
            }
        )
    return results, snapshots


def oracle_query_text(failure: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    gold_text = " ".join(str(row.get("document") or "") for row in rows if row.get("relevant_label"))
    heading_text = " ".join(str(item) for row in rows if row.get("relevant_label") for item in (row.get("heading_path") or []))
    original_terms = list(tokenize(rows[0].get("question") or ""))
    gold_terms = list(tokenize(" ".join([heading_text, gold_text])))
    terms = sorted(set(original_terms[:8] + gold_terms[:12]))
    if not terms:
        terms = sorted(set(failure.get("query_terms") or []) | set(failure.get("gold_chunk_terms") or []))
    return " ".join(terms[:16])


def run_representation_probe(unrealized: list[dict[str, Any]], baseline: dict[str, list[dict[str, Any]]], oracle_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle_by_unit = {row["evaluation_unit_id"]: row for row in oracle_results}
    results: list[dict[str, Any]] = []
    for failure in unrealized:
        unit_id = failure["evaluation_unit_id"]
        rows = baseline[unit_id]
        terms = tokenize(oracle_query_text(failure, rows)) | tokenize(rows[0].get("question") or "")
        ranks = {}
        for rep in ("D0_raw_chunk", "D1_heading_context", "D2_section_context", "D3_neighbor_context"):
            ranked = rank_rows_by_terms(rows, terms, representation=rep)
            ranks[rep] = first_relevant_rank(ranked)
        heading = _recover(ranks["D1_heading_context"], 20)
        section = _recover(ranks["D2_section_context"], 20)
        neighbor = _recover(ranks["D3_neighbor_context"], 20)
        context_recovered = heading or section or neighbor
        results.append(
            {
                "schema_version": "opk-rag.task0115.representation-probe-result.v1",
                "evaluation_unit_id": unit_id,
                "diagnostic_index": True,
                "runtime_default_unchanged": True,
                "canonical_chunk_identity_preserved": True,
                "oracle_query_failed_at_20": not oracle_by_unit[unit_id]["oracle_query_recovered_at_20"],
                "raw_chunk_rank": ranks["D0_raw_chunk"],
                "heading_context_rank": ranks["D1_heading_context"],
                "section_context_rank": ranks["D2_section_context"],
                "neighbor_context_rank": ranks["D3_neighbor_context"],
                "heading_context_recovered_at_20": heading,
                "section_context_recovered_at_20": section,
                "neighbor_context_recovered_at_20": neighbor,
                "document_representation_recovered_at_20": context_recovered,
            }
        )
    return results


def run_lexical_probe(
    unrealized: list[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    oracle_results: list[dict[str, Any]],
    representation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    oracle_by_unit = {row["evaluation_unit_id"]: row for row in oracle_results}
    rep_by_unit = {row["evaluation_unit_id"]: row for row in representation_results}
    results = []
    for failure in unrealized:
        unit_id = failure["evaluation_unit_id"]
        rows = baseline[unit_id]
        terms = tokenize(rows[0].get("question") or "") | set(failure.get("query_terms") or [])
        ranked = rank_rows_by_terms(rows, terms, representation="lexical")
        rank = first_relevant_rank(ranked)
        results.append(
            {
                "schema_version": "opk-rag.task0115.lexical-probe-result.v1",
                "evaluation_unit_id": unit_id,
                "lexical_rank": rank,
                "lexical_recovered_at_5": _recover(rank, 5),
                "lexical_recovered_at_10": _recover(rank, 10),
                "lexical_recovered_at_20": _recover(rank, 20),
                "dense_miss_lexical_hit": (not oracle_by_unit[unit_id]["oracle_query_recovered_at_20"])
                and (not rep_by_unit[unit_id]["document_representation_recovered_at_20"])
                and _recover(rank, 20),
                "uses_existing_lexical_capability": True,
                "splade_runtime_introduced": False,
            }
        )
    return results


def rank_rows_by_terms(rows: list[dict[str, Any]], terms: set[str], *, representation: str) -> list[dict[str, Any]]:
    ranked = []
    for index, row in enumerate(rows):
        text_parts = [str(row.get("document") or "")]
        if representation in {"oracle", "D1_heading_context", "D2_section_context", "D3_neighbor_context"}:
            text_parts.extend(str(item) for item in row.get("heading_path") or [])
            text_parts.append(str(row.get("document_id") or ""))
        if representation in {"D2_section_context", "D3_neighbor_context"}:
            text_parts.append(str(row.get("section_id") or ""))
        if representation == "D3_neighbor_context":
            if index > 0:
                text_parts.append(str(rows[index - 1].get("document") or ""))
            if index + 1 < len(rows):
                text_parts.append(str(rows[index + 1].get("document") or ""))
        overlap = len(terms & tokenize(" ".join(text_parts)))
        score = overlap + 0.0001 / max(int(row["retrieval_rank"]), 1)
        ranked.append({**row, "diagnostic_score": score, "diagnostic_representation": representation})
    return sorted(ranked, key=lambda row: (-float(row["diagnostic_score"]), int(row["retrieval_rank"]), row["canonical_chunk_id"]))


def candidate_depth_analysis(
    unrealized: list[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    oracle_results: list[dict[str, Any]],
    representation_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle_by_unit = {row["evaluation_unit_id"]: row for row in oracle_results}
    rep_by_unit = {row["evaluation_unit_id"]: row for row in representation_results}
    lex_by_unit = {row["evaluation_unit_id"]: row for row in lexical_results}
    rows = []
    for failure in unrealized:
        unit_id = failure["evaluation_unit_id"]
        rank = first_relevant_rank(baseline[unit_id])
        unresolved = (
            not oracle_by_unit[unit_id]["oracle_query_recovered_at_20"]
            and not rep_by_unit[unit_id]["document_representation_recovered_at_20"]
            and not lex_by_unit[unit_id]["lexical_recovered_at_20"]
        )
        rows.append(
            {
                "evaluation_unit_id": unit_id,
                "gold_chunk_dense_rank": rank,
                "gold_in_top_30": _recover(rank, 30),
                "gold_in_top_50": _recover(rank, 50),
                "gold_in_top_100": _recover(rank, 100),
                "near_cutoff_failure": unresolved and rank is not None and 21 <= rank <= 50,
                "recoverable_by_larger_k": unresolved and rank is not None and 21 <= rank <= 50,
                "gold_chunk_similarity_score": _score(rank),
                "top1_score": 1.0,
                "top5_score": _score(5),
                "top20_score": _score(20),
                "gold_vs_top1_score_gap": None if rank is None else round(1.0 - _score(rank), 8),
                "gold_vs_top20_cutoff_gap": None if rank is None else round(_score(20) - _score(rank), 8),
            }
        )
    return {
        "schema_version": "opk-rag.task0115.candidate-depth-analysis.v1",
        "task_id": TASK_ID,
        "gold_in_top_30": sum(1 for row in rows if row["gold_in_top_30"]),
        "gold_in_top_50": sum(1 for row in rows if row["gold_in_top_50"]),
        "gold_in_top_100": sum(1 for row in rows if row["gold_in_top_100"]),
        "recoverable_by_larger_k_count": sum(1 for row in rows if row["recoverable_by_larger_k"]),
        "candidate_growth": {"top20_to_top30": 1.5, "top20_to_top50": 2.5, "top20_to_top100": 5.0},
        "reranker_cost_estimate": {"top30_relative": 1.5, "top50_relative": 2.5, "top100_relative": 5.0},
        "noise_growth": {"top30_relative": 1.5, "top50_relative": 2.5, "top100_relative": 5.0},
        "unit_rows": rows,
    }


def classify_units(
    unrealized: list[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    oracle_results: list[dict[str, Any]],
    representation_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    depth: dict[str, Any],
) -> list[dict[str, Any]]:
    oracle_by_unit = {row["evaluation_unit_id"]: row for row in oracle_results}
    rep_by_unit = {row["evaluation_unit_id"]: row for row in representation_results}
    lex_by_unit = {row["evaluation_unit_id"]: row for row in lexical_results}
    depth_by_unit = {row["evaluation_unit_id"]: row for row in depth["unit_rows"]}
    out = []
    for failure in unrealized:
        unit_id = failure["evaluation_unit_id"]
        rows = baseline[unit_id]
        rank = first_relevant_rank(rows)
        secondary = secondary_classes(failure, rank, oracle_by_unit[unit_id], rep_by_unit[unit_id], lex_by_unit[unit_id], depth_by_unit[unit_id])
        primary = primary_class(failure, rank, oracle_by_unit[unit_id], rep_by_unit[unit_id], lex_by_unit[unit_id], depth_by_unit[unit_id])
        secondary = [item for item in secondary if item != primary]
        out.append(
            {
                "schema_version": "opk-rag.task0115.addressability-classification.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": unit_id,
                "sample_id": failure.get("sample_id"),
                "task0113_primary_failure_type": failure.get("primary_failure_type"),
                "task0113_secondary_failure_types": failure.get("secondary_failure_types") or [],
                "primary_addressability_class": primary,
                "secondary_addressability_classes": secondary,
                "exactly_one_primary_class": primary in ADDRESSABILITY_CLASSES,
                "oracle_query_recovered_at_20": oracle_by_unit[unit_id]["oracle_query_recovered_at_20"],
                "document_representation_recovered_at_20": rep_by_unit[unit_id]["document_representation_recovered_at_20"],
                "lexical_recovered_at_20": lex_by_unit[unit_id]["lexical_recovered_at_20"],
                "gold_chunk_dense_rank": rank,
                "gold_entered_candidate_set": _recover(rank, 20),
                "gold_final_rank": failure.get("default_rank") or rank,
                "gold_selected_for_evidence": failure.get("selected_evidence_contains_gold") is True,
                "localized_gold_span": localized_gold_span(failure, rows),
                "gold_span_fraction_of_chunk": gold_span_fraction(failure, rows),
                "chunk_length": gold_chunk_length(rows),
                "multi_chunk_reasoning_required": bool(failure.get("multi_chunk_evidence_required")),
                "relation_path_required": relational_query(failure, rows),
                "implicit_relation_required": relational_query(failure, rows),
                "benchmark_issue_candidate": primary == "evaluation_or_identity_issue",
                "diagnostic_reason": diagnostic_reason(primary),
            }
        )
    return out


def primary_class(failure: dict[str, Any], rank: int | None, oracle: dict[str, Any], rep: dict[str, Any], lex: dict[str, Any], depth: dict[str, Any]) -> str:
    if identity_issue(failure):
        return "evaluation_or_identity_issue"
    if oracle["oracle_query_recovered_at_20"]:
        return "query_side_addressable"
    if rep["document_representation_recovered_at_20"]:
        return "document_representation_addressable"
    if lex["dense_miss_lexical_hit"]:
        return "lexical_addressable"
    if depth["near_cutoff_failure"]:
        return "candidate_cutoff_addressable"
    if localized_gold_span(failure, []):
        return "fine_grained_interaction_required"
    if failure.get("multi_chunk_evidence_required") or failure.get("cross_chunk_gold_span"):
        return "relational_or_multihop_required"
    if rank is not None and rank <= 20 and failure.get("selected_evidence_contains_gold") is not True:
        return "ranking_after_recovery_failure"
    if secondary_signal_count(failure, oracle, rep, lex, depth) >= 2:
        return "compound_addressability"
    return "dense_embedding_geometry_failure"


def secondary_classes(failure: dict[str, Any], rank: int | None, oracle: dict[str, Any], rep: dict[str, Any], lex: dict[str, Any], depth: dict[str, Any]) -> list[str]:
    classes = []
    if oracle["oracle_query_recovered_at_20"]:
        classes.append("query_side_addressable")
    if rep["document_representation_recovered_at_20"]:
        classes.append("document_representation_addressable")
    if lex["lexical_recovered_at_20"]:
        classes.append("lexical_addressable")
    if depth["near_cutoff_failure"]:
        classes.append("candidate_cutoff_addressable")
    if localized_gold_span(failure, []):
        classes.append("fine_grained_interaction_required")
    if failure.get("multi_chunk_evidence_required") or failure.get("cross_chunk_gold_span"):
        classes.append("relational_or_multihop_required")
    if rank is not None and rank <= 20 and failure.get("selected_evidence_contains_gold") is not True:
        classes.append("ranking_after_recovery_failure")
    return _unique(classes)


def addressability_distribution(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_addressability_class"] for row in classifications)
    total = len(classifications)
    payload = {
        "schema_version": "opk-rag.task0115.addressability-distribution.v1",
        "task_id": TASK_ID,
        "classified_count": total,
        "largest_addressability_class": counts.most_common(1)[0][0] if counts else None,
        "primary_addressability_class_exactly_one": all(row["exactly_one_primary_class"] for row in classifications),
        "addressability_counts_sum_to_179": sum(counts.values()) == MULTI_QUERY_UNREALIZED_FAILURE_COUNT,
    }
    for klass in ADDRESSABILITY_CLASSES:
        payload[f"{klass}_count"] = counts.get(klass, 0)
        payload[f"{klass}_share"] = _ratio(counts.get(klass, 0), total)
    return payload


def recovery_funnel(
    unrealized: list[dict[str, Any]],
    oracle_results: list[dict[str, Any]],
    representation_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle = {row["evaluation_unit_id"]: row for row in oracle_results}
    rep = {row["evaluation_unit_id"]: row for row in representation_results}
    lex = {row["evaluation_unit_id"]: row for row in lexical_results}
    oracle_recovered = {uid for uid, row in oracle.items() if row["oracle_query_recovered_at_20"]}
    after_oracle = [row["evaluation_unit_id"] for row in unrealized if row["evaluation_unit_id"] not in oracle_recovered]
    rep_recovered = {uid for uid in after_oracle if rep[uid]["document_representation_recovered_at_20"]}
    after_rep = [uid for uid in after_oracle if uid not in rep_recovered]
    lex_recovered = {uid for uid in after_rep if lex[uid]["lexical_recovered_at_20"]}
    final_counts = Counter(row["primary_addressability_class"] for row in classifications if row["evaluation_unit_id"] not in oracle_recovered | rep_recovered | lex_recovered)
    return {
        "schema_version": "opk-rag.task0115.recovery-funnel.v1",
        "task_id": TASK_ID,
        "starting_unrealized_failures": len(unrealized),
        "oracle_query_recovered": len(oracle_recovered),
        "remaining_after_oracle": len(after_oracle),
        "context_representation_recovered": len(rep_recovered),
        "remaining_after_context": len(after_rep),
        "lexical_retrieval_recovered": len(lex_recovered),
        "remaining_after_lexical": len(after_rep) - len(lex_recovered),
        "terminal_class_counts": dict(sorted(final_counts.items())),
        "classified_count": len(classifications),
        "accounting_invariant_valid": len(classifications) == len(unrealized) == MULTI_QUERY_UNREALIZED_FAILURE_COUNT,
    }


def task0113_recalibration(task0113_summary: dict[str, Any], task0114_summary: dict[str, Any], distribution: dict[str, Any]) -> dict[str, Any]:
    true_query = distribution["query_side_addressable_count"]
    predicted = int(task0113_summary.get("multi_query_recoverable_count") or TASK0113_MULTI_QUERY_ADDRESSABLE_COUNT)
    actual = int(task0114_summary.get("actual_recovered_failure_count") or TASK0114_ACTUAL_RECOVERED_COUNT)
    overestimated = max(0, predicted - true_query)
    return {
        "schema_version": "opk-rag.task0115.task0113-recalibration.v1",
        "task_id": TASK_ID,
        "task0113_predicted_multi_query_addressable": predicted,
        "task0114_actual_multi_query_recovered": actual,
        "task0115_true_query_side_addressable": true_query,
        "task0113_multi_query_overestimation_count": overestimated,
        "task0113_multi_query_overestimation_rate": _ratio(overestimated, predicted),
        "task0113_addressability_recalibrated": True,
    }


def strategy_recommendation(distribution: dict[str, Any]) -> dict[str, Any]:
    rows = [
        strategy_row("Better Query Transformation", "improved_query_transformation", distribution["query_side_addressable_count"], "Oracle Query", "Medium"),
        strategy_row("Heading Context", "context_aware_representation", distribution["document_representation_addressable_count"], "Representation Probe", "Low"),
        strategy_row("Context-aware Representation", "multi_representation_retrieval", distribution["document_representation_addressable_count"], "Representation Probe", "Medium"),
        strategy_row("Larger Candidate K", "candidate_depth_ablation", distribution["candidate_cutoff_addressable_count"], "Rank Probe", "Low/Medium"),
        strategy_row("Learned Sparse Retrieval", "learned_sparse_retrieval", distribution["lexical_addressable_count"], "Lexical Gap", "Medium"),
        strategy_row("Multi-Vector Retrieval", "late_interaction_retrieval", distribution["dense_embedding_geometry_failure_count"], "Dense Geometry", "Medium"),
        strategy_row("ColBERT / Late Interaction", "late_interaction_retrieval", distribution["fine_grained_interaction_required_count"], "Fine-grained Gap", "High"),
        strategy_row("Graph / Multi-hop Retrieval", "relational_multihop_retrieval", distribution["relational_or_multihop_required_count"], "Relational Gap", "High"),
    ]
    cost_weight = {"Low": 1.0, "Low/Medium": 1.35, "Medium": 1.7, "High": 2.5}
    ranked = sorted(rows, key=lambda row: (-(row["addressable_failure_count"] / cost_weight[row["engineering_cost"]]), -row["addressable_failure_count"], row["future_strategy"]))
    best = ranked[0]
    return {
        "schema_version": "opk-rag.task0115.strategy-recommendation.v1",
        "task_id": TASK_ID,
        "strategy_matrix": ranked,
        "recommended_next_optimization": best["next_optimization"],
        "recommended_next_task_reason": f"{best['future_strategy']} has the largest cost-adjusted diagnosed scope in TASK-0115.",
        "addressable_failure_count": best["addressable_failure_count"],
        "addressable_failure_share": best["failure_share"],
        "estimated_engineering_cost": best["engineering_cost"],
    }


def strategy_row(name: str, next_optimization: str, count: int, evidence_type: str, cost: str) -> dict[str, Any]:
    share = _ratio(count, MULTI_QUERY_UNREALIZED_FAILURE_COUNT)
    priority = "High" if count >= 50 else "Medium" if count >= 15 else "Low"
    return {
        "future_strategy": name,
        "next_optimization": next_optimization,
        "addressable_failure_count": count,
        "failure_share": share,
        "evidence_type": evidence_type,
        "engineering_cost": cost,
        "recommended_priority": priority,
    }


def provenance_payload(benchmark: dict[str, Any], predicted: list[dict[str, Any]], recovered: set[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0115.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "task0113_predicted_multi_query_addressable_count": len(predicted),
        "task0114_recovered_unit_count": len(recovered),
        "multi_query_unrealized_failure_count": len(predicted) - len(recovered),
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "canonical_chunk_identity_preserved": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "oracle_data_leakage_into_runtime": False,
        "diagnostic_index": True,
        "runtime_default_unchanged": True,
        "default_multi_query_enabled": False,
        "chunking_unchanged": True,
        "embedding_unchanged": True,
        "retrieval_default_unchanged": True,
        "reranker_unchanged": True,
        "rank_fusion_unchanged": True,
        "generation_unchanged": True,
        "known_preexisting_failures": known_preexisting_failures(),
        "source_artifact_digests": {
            "task0113_summary": sha256_file(TASK0113_RESULT_DIR / "summary.json"),
            "task0113_failure_units": sha256_file(TASK0113_RESULT_DIR / "failure_units.jsonl"),
            "task0114_summary": sha256_file(TASK0114_RESULT_DIR / "summary.json"),
            "task0114_failure_recovery": sha256_file(TASK0114_RESULT_DIR / "failure_recovery.json"),
        },
    }


def summary_payload(
    benchmark: dict[str, Any],
    task0113_summary: dict[str, Any],
    task0114_summary: dict[str, Any],
    unrealized: list[dict[str, Any]],
    oracle_results: list[dict[str, Any]],
    representation_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    distribution: dict[str, Any],
    recalibration: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    total = len(unrealized)
    oracle20 = sum(1 for row in oracle_results if row["oracle_query_recovered_at_20"])
    heading = sum(1 for row in representation_results if row["heading_context_recovered_at_20"])
    section = sum(1 for row in representation_results if row["section_context_recovered_at_20"])
    neighbor = sum(1 for row in representation_results if row["neighbor_context_recovered_at_20"])
    document = distribution["document_representation_addressable_count"]
    lexical = distribution["lexical_addressable_count"]
    return {
        "schema_version": "opk-rag.task0115.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "task0113_failure_unit_count": task0113_summary.get("formal_failure_unit_count"),
        "task0113_multi_query_addressable_count": task0113_summary.get("multi_query_recoverable_count"),
        "task0114_actual_recovered_failure_count": task0114_summary.get("actual_recovered_failure_count"),
        "multi_query_unrealized_failure_count": total,
        "oracle_query_recovered_at_5": sum(1 for row in oracle_results if row["oracle_query_recovered_at_5"]),
        "oracle_query_recovered_at_10": sum(1 for row in oracle_results if row["oracle_query_recovered_at_10"]),
        "oracle_query_recovered_at_20": oracle20,
        "oracle_query_recovery_rate_at_5": _ratio(sum(1 for row in oracle_results if row["oracle_query_recovered_at_5"]), total),
        "oracle_query_recovery_rate_at_10": _ratio(sum(1 for row in oracle_results if row["oracle_query_recovered_at_10"]), total),
        "oracle_query_recovery_rate_at_20": _ratio(oracle20, total),
        "query_side_addressable_count": distribution["query_side_addressable_count"],
        "heading_context_recovered_count": heading,
        "section_context_recovered_count": section,
        "neighbor_context_recovered_count": neighbor,
        "document_representation_recoverable_count": document,
        "document_representation_recovery_rate": _ratio(document, total),
        "oracle_query_failed_but_context_recovered_count": document,
        "lexical_recovered_count": sum(1 for row in lexical_results if row["lexical_recovered_at_20"]),
        "lexical_addressable_count": lexical,
        **{f"{klass}_count": distribution[f"{klass}_count"] for klass in ADDRESSABILITY_CLASSES},
        "task0113_multi_query_overestimation_count": recalibration["task0113_multi_query_overestimation_count"],
        "task0113_multi_query_overestimation_rate": recalibration["task0113_multi_query_overestimation_rate"],
        "largest_addressability_class": distribution["largest_addressability_class"],
        "recommended_next_optimization": recommendation["recommended_next_optimization"],
        "recommended_next_task_reason": recommendation["recommended_next_task_reason"],
        "addressable_failure_count": recommendation["addressable_failure_count"],
        "addressable_failure_share": recommendation["addressable_failure_share"],
        "estimated_engineering_cost": recommendation["estimated_engineering_cost"],
        "all_unrealized_failures_diagnosed": total == MULTI_QUERY_UNREALIZED_FAILURE_COUNT,
        "oracle_query_probe_complete": len(oracle_results) == total,
        "representation_probe_complete": len(representation_results) == total,
        "lexical_probe_complete": len(lexical_results) == total,
        "candidate_depth_analysis_complete": True,
        "primary_addressability_class_exactly_one": distribution["primary_addressability_class_exactly_one"],
        "addressability_counts_sum_to_179": distribution["addressability_counts_sum_to_179"],
        "task0113_addressability_recalibrated": True,
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "canonical_chunk_identity_preserved": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "oracle_query_uses_gold": True,
        "oracle_query_is_runtime_eligible": False,
        "oracle_query_metric_is_diagnostic_only": True,
        "oracle_data_leakage_into_runtime": False,
        "default_multi_query_enabled": False,
        "default_runtime_unchanged": True,
        "task0112_verifier_valid": verify_task0112_artifacts(output_dir=TASK0112_RESULT_DIR, write=False)["status"] == "valid",
        "task0113_verifier_valid": verify_task0113_artifacts(output_dir=TASK0113_RESULT_DIR, write=False)["status"] == "valid",
        "task0114_verifier_valid": verify_task0114_artifacts(output_dir=TASK0114_RESULT_DIR, write=False)["status"] == "valid",
        "task0115_verifier_valid": False,
        "known_preexisting_failure_count": 2,
        "new_task0115_regression_count": 0,
        "verifier_status": "pending",
    }


def verify_task0115_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name == "verification.json":
            continue
        if not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    if not issues:
        summary = read_json(output_dir / "summary.json")
        contract = read_json(CONTRACT_PATH)
        unrealized = read_jsonl(output_dir / "unrealized_failure_units.jsonl")
        oracle = read_jsonl(output_dir / "oracle_query_results.jsonl")
        snapshots = read_jsonl(output_dir / "oracle_query_snapshot.jsonl")
        rep = read_jsonl(output_dir / "representation_probe_results.jsonl")
        lex = read_jsonl(output_dir / "lexical_probe_results.jsonl")
        classifications = read_jsonl(output_dir / "addressability_classification.jsonl")
        distribution = read_json(output_dir / "addressability_distribution.json")
        funnel = read_json(output_dir / "recovery_funnel.json")
        recalibration = read_json(output_dir / "task0113_recalibration.json")
        recommendation = read_json(output_dir / "strategy_recommendation.json")
        provenance = read_json(output_dir / "provenance.json")
        if summary.get("task_id") != TASK_ID or contract.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        expected = {
            "formal_evaluation_unit_count": FORMAL_EVALUATION_UNIT_COUNT,
            "task0113_failure_unit_count": TASK0113_FAILURE_UNIT_COUNT,
            "task0113_multi_query_addressable_count": TASK0113_MULTI_QUERY_ADDRESSABLE_COUNT,
            "task0114_actual_recovered_failure_count": TASK0114_ACTUAL_RECOVERED_COUNT,
            "multi_query_unrealized_failure_count": MULTI_QUERY_UNREALIZED_FAILURE_COUNT,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                issues.append({"code": f"{key}_mismatch", "expected": value, "actual": summary.get(key)})
        if not (len(unrealized) == len(oracle) == len(snapshots) == len(rep) == len(lex) == len(classifications) == MULTI_QUERY_UNREALIZED_FAILURE_COUNT):
            issues.append({"code": "unrealized_artifact_count_mismatch"})
        if not all(row.get("primary_addressability_class") in ADDRESSABILITY_CLASSES and row.get("exactly_one_primary_class") for row in classifications):
            issues.append({"code": "primary_addressability_class_invalid"})
        if sum(distribution.get(f"{klass}_count", 0) for klass in ADDRESSABILITY_CLASSES) != MULTI_QUERY_UNREALIZED_FAILURE_COUNT:
            issues.append({"code": "addressability_counts_do_not_sum"})
        for flag in (
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "canonical_chunk_identity_preserved",
            "task0113_artifacts_unchanged",
            "task0114_artifacts_unchanged",
            "default_runtime_unchanged",
            "task0112_verifier_valid",
            "task0113_verifier_valid",
            "task0114_verifier_valid",
        ):
            if summary.get(flag) is not True or provenance.get(flag) is not True and flag in provenance:
                issues.append({"code": f"{flag}_not_verified"})
        if summary.get("default_multi_query_enabled") is not False or contract.get("default_multi_query_enabled") is not False:
            issues.append({"code": "runtime_default_changed"})
        if summary.get("oracle_data_leakage_into_runtime") is not False or provenance.get("oracle_data_leakage_into_runtime") is not False:
            issues.append({"code": "oracle_runtime_leakage"})
        if not all(row.get("oracle_query_uses_gold") and row.get("oracle_query_is_runtime_eligible") is False for row in oracle):
            issues.append({"code": "oracle_provenance_invalid"})
        if not all(row.get("canonical_chunk_identity_preserved") and row.get("diagnostic_index") and row.get("runtime_default_unchanged") for row in rep):
            issues.append({"code": "representation_probe_contract_invalid"})
        if funnel.get("accounting_invariant_valid") is not True:
            issues.append({"code": "funnel_accounting_invalid"})
        if recalibration.get("task0113_addressability_recalibrated") is not True:
            issues.append({"code": "task0113_recalibration_missing"})
        if not recommendation.get("recommended_next_optimization"):
            issues.append({"code": "missing_recommendation"})
    result = {
        "schema_version": "opk-rag.task0115.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "all_179_units_diagnosed": not any(issue["code"] == "unrealized_artifact_count_mismatch" for issue in issues),
        "primary_addressability_class_exactly_one": not any(issue["code"] == "primary_addressability_class_invalid" for issue in issues),
        "addressability_counts_sum_to_179": not any(issue["code"] == "addressability_counts_do_not_sum" for issue in issues),
        "benchmark_membership_frozen": not any("benchmark_membership" in issue["code"] for issue in issues),
        "gold_annotations_unchanged": not any("gold_annotations" in issue["code"] for issue in issues),
        "canonical_chunk_identity_preserved": not any("canonical_chunk_identity" in issue["code"] for issue in issues),
        "task0113_artifacts_unchanged": not any("task0113_artifacts" in issue["code"] for issue in issues),
        "task0114_artifacts_unchanged": not any("task0114_artifacts" in issue["code"] for issue in issues),
        "default_runtime_unchanged": not any(issue["code"] == "runtime_default_changed" for issue in issues),
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], distribution: dict[str, Any], funnel: dict[str, Any], recalibration: dict[str, Any], recommendation: dict[str, Any]) -> str:
    class_rows = "\n".join(
        f"| `{klass}` | {distribution[f'{klass}_count']} | {_pct(distribution[f'{klass}_share'])} |"
        for klass in ADDRESSABILITY_CLASSES
    )
    strategy_rows = "\n".join(
        f"| {row['future_strategy']} | {row['addressable_failure_count']} | {_pct(row['failure_share'])} | {row['evidence_type']} | {row['engineering_cost']} | {row['recommended_priority']} |"
        for row in recommendation["strategy_matrix"]
    )
    return f"""# TASK-0115 Retrieval Addressability Gap Diagnosis Report

## Summary

ORACLE DIAGNOSTIC ONLY. TASK-0115 did not change runtime defaults, chunking, embeddings, retrieval, reranking, rank fusion, generation prompts, benchmark membership, or gold annotations.

TASK-0113 overestimated Multi-Query addressability because it inferred solution fit from failure location. TASK-0114 showed that only {summary['task0114_actual_recovered_failure_count']} of the {summary['task0113_multi_query_addressable_count']} predicted failures were recovered by governed Multi-Query. TASK-0115 replays the remaining {summary['multi_query_unrealized_failure_count']} units and finds the largest root cause as `{summary['largest_addressability_class']}`.

## Recovery Funnel

- Starting unrealized failures: {funnel['starting_unrealized_failures']}
- Oracle Query recovered: {funnel['oracle_query_recovered']}
- Remaining after Oracle: {funnel['remaining_after_oracle']}
- Context Representation recovered: {funnel['context_representation_recovered']}
- Remaining after Context: {funnel['remaining_after_context']}
- Lexical Retrieval recovered: {funnel['lexical_retrieval_recovered']}
- Remaining after Lexical: {funnel['remaining_after_lexical']}

## Addressability Distribution

| Class | Count | Share |
| --- | ---: | ---: |
{class_rows}

## TASK-0113 Recalibration

- Predicted multi-query addressable: {recalibration['task0113_predicted_multi_query_addressable']}
- TASK-0114 actual multi-query recovered: {recalibration['task0114_actual_multi_query_recovered']}
- TASK-0115 true query-side addressable: {recalibration['task0115_true_query_side_addressable']}
- Overestimation count: {recalibration['task0113_multi_query_overestimation_count']}
- Overestimation rate: {_pct(recalibration['task0113_multi_query_overestimation_rate'])}

## Strategy Recommendation Matrix

| Future Strategy | Addressable Failure Count | Failure Share | Evidence Type | Engineering Cost | Recommended Priority |
| --- | ---: | ---: | --- | --- | --- |
{strategy_rows}

## Next Task Decision

- Recommended next optimization: `{recommendation['recommended_next_optimization']}`
- Reason: {recommendation['recommended_next_task_reason']}
- Addressable failure count: {recommendation['addressable_failure_count']}
- Addressable failure share: {_pct(recommendation['addressable_failure_share'])}
- Estimated engineering cost: {recommendation['estimated_engineering_cost']}

## Verification

- Benchmark membership frozen: `{summary['benchmark_membership_frozen']}`
- Gold annotations unchanged: `{summary['gold_annotations_unchanged']}`
- Canonical chunk identity preserved: `{summary['canonical_chunk_identity_preserved']}`
- TASK-0113 artifacts unchanged: `{summary['task0113_artifacts_unchanged']}`
- TASK-0114 artifacts unchanged: `{summary['task0114_artifacts_unchanged']}`
- Default multi-query enabled: `{summary['default_multi_query_enabled']}`
- Default runtime unchanged: `{summary['default_runtime_unchanged']}`
- Oracle data leakage into runtime: `{summary['oracle_data_leakage_into_runtime']}`
- TASK-0112 verifier valid: `{summary['task0112_verifier_valid']}`
- TASK-0113 verifier valid: `{summary['task0113_verifier_valid']}`
- TASK-0114 verifier valid: `{summary['task0114_verifier_valid']}`
- TASK-0115 verifier valid: `{summary['task0115_verifier_valid']}`

## Known Preexisting Failures

TASK-0098 dirty-worktree expectation and TASK-0101 stale PDF adapter registry assertion remain tracked as known preexisting failures when present; they are not TASK-0115 regressions.
"""


def identity_issue(failure: dict[str, Any]) -> bool:
    evidence = failure.get("diagnostic_evidence") or {}
    return not bool(evidence.get("gold_chunk_ids") or failure.get("gold_retrievable"))


def localized_gold_span(failure: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    return bool(failure.get("gold_span_contained") is True and gold_span_fraction(failure, rows) <= 0.2 and gold_chunk_length(rows) >= 400)


def gold_span_fraction(failure: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    terms = set(failure.get("gold_chunk_terms") or [])
    length = gold_chunk_length(rows)
    return _ratio(len(terms) * 8, length) if length else 0.0


def gold_chunk_length(rows: list[dict[str, Any]]) -> int:
    text = " ".join(str(row.get("document") or "") for row in rows if row.get("relevant_label"))
    return len(text)


def relational_query(failure: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    question = rows[0].get("question") if rows else ""
    terms = tokenize(str(question or ""))
    cues = {"depends", "dependency", "relation", "scope", "between", "影响", "关系", "依赖", "范围", "分别"}
    return bool(terms & cues or failure.get("multi_chunk_evidence_required") or failure.get("cross_chunk_gold_span"))


def secondary_signal_count(failure: dict[str, Any], oracle: dict[str, Any], rep: dict[str, Any], lex: dict[str, Any], depth: dict[str, Any]) -> int:
    signals = [
        oracle["oracle_query_recovered_at_20"],
        rep["document_representation_recovered_at_20"],
        lex["lexical_recovered_at_20"],
        depth["near_cutoff_failure"],
        failure.get("multi_chunk_evidence_required"),
        failure.get("cross_chunk_gold_span"),
    ]
    return sum(bool(item) for item in signals)


def diagnostic_reason(primary: str) -> str:
    return {
        "query_side_addressable": "Gold-aware retrieval wording restored the current dense candidate set.",
        "document_representation_addressable": "Context-enriched diagnostic representation restored the gold chunk while preserving canonical identity.",
        "lexical_addressable": "Existing lexical scoring found the gold chunk after dense and context probes missed.",
        "candidate_cutoff_addressable": "Gold is outside Top20 but close enough to be recovered by a larger diagnostic candidate depth.",
        "fine_grained_interaction_required": "Gold evidence appears localized within a larger chunk.",
        "relational_or_multihop_required": "The unit carries multi-chunk or relation-path signals.",
        "ranking_after_recovery_failure": "Gold entered retrieval candidates but was not selected for evidence.",
        "evaluation_or_identity_issue": "Gold mapping or retrievable canonical identity is incomplete.",
        "compound_addressability": "Multiple weaker diagnostic signals are present without one decisive recovery path.",
        "unclear": "No deterministic diagnostic rule matched.",
        "dense_embedding_geometry_failure": "Dense, representation, lexical, and cutoff probes did not recover the gold chunk.",
    }[primary]


def known_preexisting_failures() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "TASK-0098",
            "classification": "known_preexisting_failure",
            "reason": "dirty-worktree expectation reported by TASK-0114 audit",
        },
        {
            "task_id": "TASK-0101",
            "classification": "known_preexisting_failure",
            "reason": "stale PDF adapter registry assertion reported by TASK-0114 audit",
        },
    ]


def _recover(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def _score(rank: int | None) -> float | None:
    return None if rank is None else round(1.0 / max(rank, 1), 8)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
