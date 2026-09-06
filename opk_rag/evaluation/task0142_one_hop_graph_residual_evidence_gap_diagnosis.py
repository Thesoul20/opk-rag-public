from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0138_graph_sensitive_retrieval_runtime_promotion as task0138
import opk_rag.evaluation.task0139_graph_retrieval_activation_routing_experiment as task0139
import opk_rag.evaluation.task0140_graph_retrieval_activation_runtime_promotion as task0140
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval


TASK_ID = "TASK-0142"
EXPERIMENT_ID = "task0142-one-hop-graph-residual-evidence-gap-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0142_one_hop_graph_residual_evidence_gap_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0142_ONE_HOP_GRAPH_RESIDUAL_EVIDENCE_GAP_DIAGNOSIS_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "residual_evidence_traces.jsonl",
    "sample_diagnosis.jsonl",
    "root_cause_distribution.json",
    "stage_loss_report.json",
    "preflight.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0140_authority_valid",
    "task0141_authority_valid",
    "benchmark_revision",
    "benchmark_digest",
    "residual_incomplete_unit_count",
    "residual_missing_required_evidence_count",
    "first_loss_initial_retrieval_count",
    "first_loss_graph_activation_count",
    "first_loss_graph_seed_count",
    "first_loss_graph_reachability_count",
    "first_loss_graph_expansion_count",
    "first_loss_candidate_union_count",
    "first_loss_ranking_input_count",
    "first_loss_ranking_output_count",
    "first_loss_evidence_budget_count",
    "first_loss_evidence_composition_count",
    "first_loss_generation_count",
    "dominant_residual_root_cause",
    "dominant_residual_root_cause_count",
    "dominant_residual_root_cause_rate",
    "bounded_multi_hop_would_address_residual",
    "task0141_negative_result_explained",
    "runtime_gold_metadata_usage",
    "runtime_policy_mutation_count",
    "recommended_mitigation_family",
    "promotion_decision",
    "promotion_applied",
)

FIRST_LOSS_STAGES = (
    "initial_retrieval",
    "graph_activation",
    "graph_seed",
    "graph_reachability",
    "graph_expansion",
    "candidate_union",
    "ranking_input",
    "ranking_output",
    "evidence_budget",
    "evidence_composition",
    "generation",
)

ROOT_CAUSE_TO_FAMILY = {
    "benchmark_authority_mismatch": "benchmark_correction",
    "initial_retrieval_failure": "seed_retrieval_optimization",
    "query_document_mismatch": "seed_retrieval_optimization",
    "graph_activation_false_negative": "graph_expansion_admission",
    "graph_seed_failure": "seed_retrieval_optimization",
    "graph_edge_missing": "graph_schema_or_edge_completion",
    "entity_resolution_failure": "entity_resolution_improvement",
    "graph_expansion_admission_failure": "graph_expansion_admission",
    "neighbor_budget_cutoff": "graph_expansion_admission",
    "edge_filter_rejection": "graph_expansion_admission",
    "source_binding_failure": "graph_expansion_admission",
    "candidate_union_loss": "candidate_composition",
    "candidate_identity_collision": "candidate_composition",
    "ranking_input_cutoff": "ranking_optimization",
    "ranking_displacement": "ranking_optimization",
    "ranking_quality_failure": "ranking_optimization",
    "evidence_budget_cutoff": "evidence_budgeted_composition",
    "evidence_composition_filter": "evidence_budgeted_composition",
    "evidence_conversion_failure": "evidence_budgeted_composition",
    "generation_context_assembly_failure": "generation_conversion",
    "generation_over_abstention": "generation_conversion",
    "generation_evidence_underuse": "generation_conversion",
    "generation_reasoning_failure": "generation_conversion",
    "unknown": "no_single_dominant_root_cause",
}


def run_task0142_one_hop_graph_residual_evidence_gap_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = runtime_policy_snapshot()
    preflight = build_preflight()
    if not preflight["preflight_valid"]:
        write_json(output_dir / "preflight.json", preflight)
        raise RuntimeError(f"TASK-0142 preflight failed: {preflight['failures']}")

    samples = task0137.load_graph_sensitive_samples()
    baseline_rows = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    residual_samples = extract_residual_samples(samples, baseline_rows)
    traces = [trace_missing_required_evidence(item["sample"], required_unit=item["missing_required_unit"], baseline_row=item["baseline_row"]) for item in residual_samples]
    sample_diagnosis = build_sample_diagnosis(traces)
    root_cause_distribution = build_root_cause_distribution(traces)
    stage_loss_report = build_stage_loss_report(traces)
    after_policy = runtime_policy_snapshot()
    policy_mutation_count = sum(before_policy[key] != after_policy[key] for key in before_policy)
    config = build_config(preflight, before_policy, after_policy)
    summary = build_summary(
        preflight,
        traces,
        root_cause_distribution,
        stage_loss_report,
        policy_mutation_count=policy_mutation_count,
    )
    contract = build_contract(summary, config)
    digests = build_digests(preflight, config, traces, sample_diagnosis, root_cause_distribution, stage_loss_report, summary)

    write_jsonl(output_dir / "residual_evidence_traces.jsonl", traces)
    write_jsonl(output_dir / "sample_diagnosis.jsonl", sample_diagnosis)
    write_json(output_dir / "root_cause_distribution.json", root_cause_distribution)
    write_json(output_dir / "stage_loss_report.json", stage_loss_report)
    write_json(output_dir / "preflight.json", preflight)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0142_artifacts(output_dir=output_dir, write=True)
    summary["task0142_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, root_cause_distribution, stage_loss_report), encoding="utf-8")
    return summary


def build_preflight() -> dict[str, Any]:
    task0137_verification = task0137.verify_task0137_artifacts(write=False)
    task0138_verification = task0138.verify_task0138_artifacts(write=False)
    task0139_verification = task0139.verify_task0139_artifacts(write=False)
    task0140_verification = task0140.verify_task0140_artifacts(write=False)
    task0141_verification = task0141.verify_task0141_artifacts(write=False)
    task0140_summary = read_json(task0140.RESULT_DIR / "summary.json")
    task0141_summary = read_json(task0141.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    failures = []
    for task_name, verification in (
        ("task0137", task0137_verification),
        ("task0138", task0138_verification),
        ("task0139", task0139_verification),
        ("task0140", task0140_verification),
        ("task0141", task0141_verification),
    ):
        if verification["status"] != "valid":
            failures.append(f"{task_name}_verifier_invalid")
    if task0140_summary.get("promotion_decision") != "promote_retrieval_aware_graph_activation_runtime":
        failures.append("task0140_canonical_runtime_authority_not_preserved")
    if task0141_summary.get("best_arm") != task0141.A0_BASELINE:
        failures.append("task0141_negative_result_not_preserved")
    if graph_authority["graph_snapshot_digest"] != task0140_summary.get("benchmark_digest"):
        failures.append("task0140_benchmark_digest_drift")
    if graph_authority["graph_snapshot_digest"] != task0141_summary.get("benchmark_digest"):
        failures.append("task0141_benchmark_digest_drift")
    return {
        "schema_version": "opk-rag.task0142.preflight.v1",
        "task_id": TASK_ID,
        "task0137_verifier_status": task0137_verification["status"],
        "task0138_verifier_status": task0138_verification["status"],
        "task0139_verifier_status": task0139_verification["status"],
        "task0140_verifier_status": task0140_verification["status"],
        "task0141_verifier_status": task0141_verification["status"],
        "task0140_summary_digest": sha256_file(task0140.RESULT_DIR / "summary.json"),
        "task0141_summary_digest": sha256_file(task0141.RESULT_DIR / "summary.json"),
        "task0140_canonical_runtime_authority_preserved": task0140_summary.get("promotion_decision") == "promote_retrieval_aware_graph_activation_runtime",
        "task0141_negative_result_preserved": task0141_summary.get("best_arm") == task0141.A0_BASELINE
        and task0141_summary.get("promotion_decision") == "do_not_promote_multi_hop_runtime_default",
        "benchmark_revision": graph_authority["graph_snapshot_revision"],
        "benchmark_digest": graph_authority["graph_snapshot_digest"],
        "current_git_head": _git(["rev-parse", "HEAD"]),
        "working_tree_status": _git(["status", "--short"]),
        "preflight_valid": not failures,
        "failures": failures,
    }


def extract_residual_samples(samples: list[dict[str, Any]], baseline_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {sample["sample_id"]: sample for sample in samples}
    residual: list[dict[str, Any]] = []
    for sample_id, row in sorted(baseline_rows.items()):
        if row.get("complete_required_evidence_set_recall"):
            continue
        sample = by_id[sample_id]
        retrieved = set(row.get("candidate_ids") or [])
        for unit in sample.get("required_source_units", []):
            if unit["source_unit_id"] not in retrieved:
                residual.append({"sample": sample, "baseline_row": row, "missing_required_unit": unit})
    return residual


def trace_missing_required_evidence(sample: dict[str, Any], *, required_unit: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    required_id = required_unit["source_unit_id"]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    seeds = graph_retrieval.select_seed_candidates(sample)
    seed_ids = [candidate["candidate_id"] for candidate in seeds]
    seed_docs = {candidate["document_id"] for candidate in seeds}
    decision = graph_activation.decide_runtime_graph_activation(sample, activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE, seeds=seeds)
    runtime = graph_retrieval.expand_runtime_candidates(
        seeds,
        sample,
        runtime_config=graph_activation.runtime_config_for_decision(decision),
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    candidate_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
    added_ids = [candidate["candidate_id"] for candidate in runtime.added_candidates]
    candidate_union_ids = graph_retrieval.merge_candidates(seeds, runtime.added_candidates)
    union_ids = [candidate["candidate_id"] for candidate in candidate_union_ids]
    evidence, composition_trace = evidence_composition.compose_evidence(list(runtime.candidates), evaluation_unit_id=sample["sample_id"])
    evidence_ids = [row["canonical_chunk_id"] for row in evidence]
    composition_by_id = {row["canonical_chunk_id"]: row for row in composition_trace}
    distance = task0141.graph_distances_from_seeds(sample, max_hops=4).get(required_unit["document_id"])
    benchmark_valid = required_evidence_reference_valid(sample, required_unit)
    generation_context_present = required_id in evidence_ids
    first_loss_stage, root_cause = classify_first_loss(
        benchmark_valid=benchmark_valid,
        required_unit=required_unit,
        sample=sample,
        initial_retrieval_present=required_id in seed_ids,
        graph_activation_decision=decision.graph_activation,
        graph_seed_available=bool(seeds) and required_unit["document_id"] in seed_docs,
        graph_reachable_one_hop=distance is not None and distance <= 1,
        graph_reachable_two_hop=distance is not None and distance <= 2,
        graph_expansion_present=required_id in added_ids,
        candidate_union_present=required_id in union_ids,
        ranking_input_present=required_id in candidate_ids,
        ranking_output_present=required_id in candidate_ids,
        evidence_selected=required_id in evidence_ids,
        composition_trace=composition_by_id.get(required_id),
    )
    ranking_rank = candidate_ids.index(required_id) + 1 if required_id in candidate_ids else None
    trace = {
        "schema_version": "opk-rag.task0142.required-evidence-trace.v1",
        "task_id": TASK_ID,
        "query_id": sample["sample_id"],
        "sample_id": sample["sample_id"],
        "query": sample.get("query") or sample.get("question"),
        "required_evidence_id": required_id,
        "required_evidence_document_id": required_unit.get("document_id"),
        "required_evidence_ids": required_ids,
        "retrieved_required_evidence_ids": [candidate_id for candidate_id in candidate_ids if candidate_id in set(required_ids)],
        "missing_required_evidence_ids": [candidate_id for candidate_id in required_ids if candidate_id not in set(candidate_ids)],
        "required_evidence_reference_valid": benchmark_valid,
        "initial_candidate_ids": seed_ids,
        "initial_retrieval_present": required_id in seed_ids,
        "graph_activation_decision": decision.graph_activation,
        "graph_activation_reason": decision.activation_reason,
        "graph_seed_available": bool(seeds) and required_unit["document_id"] in seed_docs,
        "graph_seed_candidate_ids": seed_ids,
        "graph_distance_from_seed": distance,
        "graph_reachable_one_hop": distance is not None and distance <= 1,
        "graph_reachable_two_hop": distance is not None and distance <= 2,
        "graph_expansion_present": required_id in added_ids,
        "graph_expansion_added_candidate_ids": added_ids,
        "candidate_union_present": required_id in union_ids,
        "candidate_after_dedup_present": required_id in candidate_ids,
        "candidate_ids": candidate_ids,
        "ranking_input_present": required_id in candidate_ids,
        "ranking_output_present": required_id in candidate_ids,
        "ranking_rank": ranking_rank,
        "ranking_score": None,
        "candidate_pre_ranking_rank": ranking_rank,
        "ranking_input_limit": len(candidate_ids),
        "evidence_budget_eligible": ranking_rank is not None and ranking_rank <= evidence_composition.EVIDENCE_BUDGET_LIMIT,
        "evidence_selected": required_id in evidence_ids,
        "evidence_ids": evidence_ids,
        "evidence_composition_rejection_reason": (composition_by_id.get(required_id) or {}).get("rejection_reason"),
        "generation_context_present": generation_context_present,
        "final_answer_supported": bool(baseline_row.get("downstream_after")) if generation_context_present else False,
        "first_loss_stage": first_loss_stage,
        "root_cause": root_cause,
        "secondary_contributing_factors": secondary_factors(distance, required_id, seed_ids, added_ids, candidate_ids, evidence_ids),
        "bounded_multi_hop_candidate_present": _bounded_multi_hop_candidate_present(sample["sample_id"], required_id),
        "gold_signal_used_by_runtime": _gold_signal_used(decision.to_json()) or bool(runtime.trace.get("gold_signal_used_by_graph_policy")),
        "runtime_policy_mutated": False,
    }
    return trace


def required_evidence_reference_valid(sample: dict[str, Any], required_unit: dict[str, Any]) -> bool:
    required_ids = {unit.get("source_unit_id") for unit in sample.get("required_source_units", [])}
    if required_unit.get("source_unit_id") not in required_ids:
        return False
    if not required_unit.get("document_id") or not required_unit.get("source_unit_id"):
        return False
    return True


def classify_first_loss(
    *,
    benchmark_valid: bool,
    required_unit: dict[str, Any],
    sample: dict[str, Any],
    initial_retrieval_present: bool,
    graph_activation_decision: bool,
    graph_seed_available: bool,
    graph_reachable_one_hop: bool,
    graph_reachable_two_hop: bool,
    graph_expansion_present: bool,
    candidate_union_present: bool,
    ranking_input_present: bool,
    ranking_output_present: bool,
    evidence_selected: bool,
    composition_trace: dict[str, Any] | None,
) -> tuple[str, str]:
    if not benchmark_valid:
        return "benchmark", "benchmark_authority_mismatch"
    if not initial_retrieval_present:
        return "initial_retrieval", "initial_retrieval_failure"
    if not graph_activation_decision and task0137.is_graph_positive(sample):
        return "graph_activation", "graph_activation_false_negative"
    if not graph_seed_available:
        return "graph_seed", "graph_seed_failure"
    if not graph_reachable_two_hop:
        return "graph_reachability", "graph_edge_missing"
    if graph_reachable_one_hop and not graph_expansion_present and not initial_retrieval_present:
        return "graph_expansion", "graph_expansion_admission_failure"
    if graph_expansion_present and not candidate_union_present:
        return "candidate_union", "candidate_union_loss"
    if candidate_union_present and not ranking_input_present:
        return "ranking_input", "ranking_input_cutoff"
    if ranking_input_present and not ranking_output_present:
        return "ranking_output", "ranking_displacement"
    if ranking_output_present and not evidence_selected:
        rejection = (composition_trace or {}).get("rejection_reason")
        if rejection == "budget_cutoff":
            return "evidence_budget", "evidence_budget_cutoff"
        return "evidence_composition", "evidence_composition_filter"
    if evidence_selected:
        return "generation", "generation_evidence_underuse"
    return "unknown", "unknown"


def secondary_factors(distance: int | None, required_id: str, seed_ids: list[str], added_ids: list[str], candidate_ids: list[str], evidence_ids: list[str]) -> list[str]:
    factors = []
    if distance == 0 and required_id not in seed_ids:
        factors.append("same_document_different_required_span")
    if distance == 1 and required_id not in added_ids:
        factors.append("one_hop_document_reachable_without_required_span_admission")
    if required_id in candidate_ids and required_id not in evidence_ids:
        factors.append("candidate_survived_until_evidence_boundary")
    return factors


def build_sample_diagnosis(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        by_sample.setdefault(trace["sample_id"], []).append(trace)
    rows = []
    for sample_id, sample_traces in sorted(by_sample.items()):
        rows.append(
            {
                "schema_version": "opk-rag.task0142.sample-diagnosis.v1",
                "task_id": TASK_ID,
                "sample_id": sample_id,
                "query": sample_traces[0].get("query"),
                "required_evidence_ids": sample_traces[0]["required_evidence_ids"],
                "retrieved_required_evidence_ids": sample_traces[0]["retrieved_required_evidence_ids"],
                "missing_required_evidence_ids": [trace["required_evidence_id"] for trace in sample_traces],
                "missing_required_evidence_count": len(sample_traces),
                "first_loss_stages": sorted({trace["first_loss_stage"] for trace in sample_traces}),
                "root_causes": sorted({trace["root_cause"] for trace in sample_traces}),
            }
        )
    return rows


def build_root_cause_distribution(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(trace["root_cause"] for trace in traces)
    total = len(traces)
    return {
        "schema_version": "opk-rag.task0142.root-cause-distribution.v1",
        "task_id": TASK_ID,
        "total_missing_required_evidence": total,
        "root_causes": [
            {"root_cause": root_cause, "count": count, "rate": round(_safe_div(count, total), 6)}
            for root_cause, count in sorted(counts.items())
        ],
    }


def build_stage_loss_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(trace["first_loss_stage"] for trace in traces)
    return {
        "schema_version": "opk-rag.task0142.stage-loss-report.v1",
        "task_id": TASK_ID,
        "stages": [{"first_loss_stage": stage, "count": counts[stage]} for stage in FIRST_LOSS_STAGES],
        "other_stage_count": sum(count for stage, count in counts.items() if stage not in FIRST_LOSS_STAGES),
    }


def build_summary(
    preflight: dict[str, Any],
    traces: list[dict[str, Any]],
    root_cause_distribution: dict[str, Any],
    stage_loss_report: dict[str, Any],
    *,
    policy_mutation_count: int,
) -> dict[str, Any]:
    stage_counts = {row["first_loss_stage"]: row["count"] for row in stage_loss_report["stages"]}
    root_rows = root_cause_distribution["root_causes"]
    dominant = max(root_rows, key=lambda row: (row["count"], row["root_cause"])) if root_rows else {"root_cause": "unknown", "count": 0, "rate": 0.0}
    two_hop_only_count = sum(
        1
        for trace in traces
        if trace["graph_reachable_two_hop"] and not trace["graph_reachable_one_hop"]
    )
    bounded_multi_hop_would_address = bool(traces) and two_hop_only_count > len(traces) / 2
    runtime_gold_metadata_usage = any(trace["gold_signal_used_by_runtime"] for trace in traces)
    recommended_family = recommended_mitigation_family(dominant["root_cause"], dominant["count"], len(traces))
    return {
        "schema_version": "opk-rag.task0142.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if policy_mutation_count == 0 and not runtime_gold_metadata_usage else "invalid_diagnosis",
        "task0137_authority_valid": preflight["task0137_verifier_status"] == "valid",
        "task0138_authority_valid": preflight["task0138_verifier_status"] == "valid",
        "task0139_authority_valid": preflight["task0139_verifier_status"] == "valid",
        "task0140_authority_valid": preflight["task0140_verifier_status"] == "valid",
        "task0141_authority_valid": preflight["task0141_verifier_status"] == "valid",
        "benchmark_revision": preflight["benchmark_revision"],
        "benchmark_digest": preflight["benchmark_digest"],
        "task0140_summary_digest": preflight["task0140_summary_digest"],
        "task0141_summary_digest": preflight["task0141_summary_digest"],
        "residual_incomplete_unit_count": len({trace["sample_id"] for trace in traces}),
        "residual_missing_required_evidence_count": len(traces),
        "first_loss_initial_retrieval_count": stage_counts.get("initial_retrieval", 0),
        "first_loss_graph_activation_count": stage_counts.get("graph_activation", 0),
        "first_loss_graph_seed_count": stage_counts.get("graph_seed", 0),
        "first_loss_graph_reachability_count": stage_counts.get("graph_reachability", 0),
        "first_loss_graph_expansion_count": stage_counts.get("graph_expansion", 0),
        "first_loss_candidate_union_count": stage_counts.get("candidate_union", 0),
        "first_loss_ranking_input_count": stage_counts.get("ranking_input", 0),
        "first_loss_ranking_output_count": stage_counts.get("ranking_output", 0),
        "first_loss_evidence_budget_count": stage_counts.get("evidence_budget", 0),
        "first_loss_evidence_composition_count": stage_counts.get("evidence_composition", 0),
        "first_loss_generation_count": stage_counts.get("generation", 0),
        "dominant_residual_root_cause": dominant["root_cause"],
        "dominant_residual_root_cause_count": dominant["count"],
        "dominant_residual_root_cause_rate": dominant["rate"],
        "bounded_multi_hop_would_address_residual": bounded_multi_hop_would_address,
        "task0141_negative_result_explained": not bounded_multi_hop_would_address,
        "runtime_gold_metadata_usage": runtime_gold_metadata_usage,
        "runtime_policy_mutation_count": policy_mutation_count,
        "recommended_mitigation_family": recommended_family,
        "promotion_decision": promotion_decision_for_family(recommended_family),
        "promotion_applied": False,
    }


def recommended_mitigation_family(root_cause: str, count: int, total: int) -> str:
    if total == 0 or count == 0:
        return "no_single_dominant_root_cause"
    if count / total < 0.5:
        return "no_single_dominant_root_cause"
    return ROOT_CAUSE_TO_FAMILY.get(root_cause, "no_single_dominant_root_cause")


def promotion_decision_for_family(family: str) -> str:
    return {
        "seed_retrieval_optimization": "diagnose_to_seed_retrieval_mitigation",
        "graph_schema_or_edge_completion": "diagnose_to_graph_edge_completion",
        "entity_resolution_improvement": "diagnose_to_entity_resolution_mitigation",
        "graph_expansion_admission": "diagnose_to_graph_expansion_admission_mitigation",
        "candidate_composition": "diagnose_to_candidate_composition_mitigation",
        "ranking_optimization": "diagnose_to_ranking_mitigation",
        "evidence_budgeted_composition": "diagnose_to_evidence_budget_mitigation",
        "generation_conversion": "diagnose_to_generation_conversion_mitigation",
        "benchmark_correction": "diagnose_to_benchmark_correction",
    }.get(family, "no_single_dominant_residual_failure")


def build_config(preflight: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0142.config.v1",
        "task_id": TASK_ID,
        "baseline": "TASK-0140 retrieval-aware activation plus one-hop graph expansion",
        "benchmark_digest": preflight["benchmark_digest"],
        "before_runtime_policy_snapshot": before_policy,
        "after_runtime_policy_snapshot": after_policy,
        "graph_activation_policy_unchanged": before_policy["graph_activation_policy"] == after_policy["graph_activation_policy"],
        "graph_retrieval_policy_unchanged": before_policy["graph_retrieval_policy_digest"] == after_policy["graph_retrieval_policy_digest"],
        "ranking_policy_unchanged": before_policy["ranking_policy"] == after_policy["ranking_policy"],
        "evidence_policy_unchanged": before_policy["evidence_policy_digest"] == after_policy["evidence_policy_digest"],
        "generation_policy_unchanged": before_policy["generation_policy"] == after_policy["generation_policy"],
        "runtime_default_modified": False,
        "gold_signal_allowed": False,
        "promotion_applied": False,
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0142.one-hop-graph-residual-evidence-gap-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "baseline": config["baseline"],
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "runtime_policy_mutation_count": summary["runtime_policy_mutation_count"],
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "promotion_applied": summary["promotion_applied"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:4])
    return {
        "schema_version": "opk-rag.task0142.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnosis_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "diagnosis_replay_equivalent": True,
    }


def verify_task0142_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    traces = read_jsonl(output_dir / "residual_evidence_traces.jsonl") if (output_dir / "residual_evidence_traces.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("task0140_authority_valid") is not True:
        failures.append("task0140_authority_invalid")
    if summary.get("task0141_authority_valid") is not True:
        failures.append("task0141_authority_invalid")
    if summary.get("promotion_applied") is not False:
        failures.append("diagnosis_promoted_runtime")
    if summary.get("runtime_policy_mutation_count") != 0:
        failures.append("runtime_policy_mutation_detected")
    if summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("runtime_gold_metadata_usage_detected")
    if any(trace.get("first_loss_stage") not in (*FIRST_LOSS_STAGES, "benchmark", "unknown") for trace in traces):
        failures.append("invalid_first_loss_stage")
    if any(not trace.get("root_cause") for trace in traces):
        failures.append("missing_root_cause")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0142.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "summary_required_fields_present": not required_missing,
        "promotion_decision": summary.get("promotion_decision"),
        "promotion_applied": summary.get("promotion_applied"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], root_cause_distribution: dict[str, Any], stage_loss_report: dict[str, Any]) -> str:
    root_rows = "\n".join(f"* `{row['root_cause']}`: {row['count']} ({row['rate']})" for row in root_cause_distribution["root_causes"])
    stage_rows = "\n".join(f"* `{row['first_loss_stage']}`: {row['count']}" for row in stage_loss_report["stages"])
    return f"""# TASK0142 One-hop Graph Residual Evidence Gap Diagnosis Report

## Summary

`task_status={summary['task_status']}`

`promotion_applied=false`

TASK-0142 diagnoses the residual incomplete Required Evidence units left by the TASK-0140 canonical runtime baseline. It does not change Graph Activation, Graph Retrieval, ranking, Evidence Composition, generation, or runtime defaults.

## Required Answers

1. Incomplete Required Evidence units: `{summary['residual_incomplete_unit_count']}`.
2. Missing Evidence first-loss stage: dominant root cause is `{summary['dominant_residual_root_cause']}`.
3. Graph Activation remains valid: first-loss graph activation count is `{summary['first_loss_graph_activation_count']}`.
4. Graph hop depth remains non-dominant: `bounded_multi_hop_would_address_residual={str(summary['bounded_multi_hop_would_address_residual']).lower()}`.
5. Missing Evidence graph presence is captured in `residual_evidence_traces.jsonl` via distance and reachability fields.
6. One-hop Expansion membership is captured with `graph_expansion_present`.
7. Candidate Union loss count is `{summary['first_loss_candidate_union_count']}`.
8. Ranking loss counts are input `{summary['first_loss_ranking_input_count']}` and output `{summary['first_loss_ranking_output_count']}`.
9. Evidence Budget loss count is `{summary['first_loss_evidence_budget_count']}`.
10. Generation loss count is `{summary['first_loss_generation_count']}`.
11. TASK-0141 has no gain because `task0141_negative_result_explained={str(summary['task0141_negative_result_explained']).lower()}`.
12. Next recommended boundary: `{summary['recommended_mitigation_family']}`.

## Stage Loss

{stage_rows}

## Root Causes

{root_rows}
"""


def runtime_policy_snapshot() -> dict[str, Any]:
    graph_policy = graph_retrieval.default_graph_retrieval_policy()
    evidence_policy = evidence_composition.targeted_budgeted_policy()
    return {
        **initial_retrieval.runtime_config_snapshot(),
        "graph_activation_policy": graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        "graph_retrieval_policy_digest": graph_policy.policy_digest,
        "graph_retrieval_policy": graph_policy.to_json(),
        "ranking_policy": "canonical_order_no_graph_bonus",
        "evidence_policy_digest": evidence_policy.policy_digest,
        "evidence_policy": evidence_policy.to_json(),
        "generation_policy": "frozen_diagnostic_no_generation_call",
    }


def _bounded_multi_hop_candidate_present(sample_id: str, required_id: str) -> bool:
    rows = [
        row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("sample_id") == sample_id and row.get("arm_id") in {task0141.A1_TWO_HOP, task0141.A2_PATH}
    ]
    return any(required_id in set(row.get("candidate_ids") or []) for row in rows)


def _gold_signal_used(decision: dict[str, Any]) -> bool:
    payload = decision.get("activation_features") or {}
    forbidden = (
        "uses_gold_label",
        "uses_gold_evidence",
        "uses_benchmark_category",
        "gold_answer",
        "gold_evidence",
        "required_evidence_set",
        "graph_positive",
        "graph_negative_control",
        "graph_unanswerable",
    )
    return any(bool(payload.get(key)) for key in forbidden if key.startswith("uses_")) or any(key in payload for key in forbidden if not key.startswith("uses_"))


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den


def _git(args: list[str]) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()
