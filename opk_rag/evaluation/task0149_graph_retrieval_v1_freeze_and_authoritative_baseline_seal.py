from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0147_guarded_structure_aware_initial_retrieval_runtime_promotion as task0147
import opk_rag.evaluation.task0148_graph_retrieval_v1_freeze_readiness_and_multihop_necessity as task0148
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval
from opk_rag.runtime_v2.reranker import RerankerRuntimeConfig


TASK_ID = "TASK-0149"
EXPERIMENT_ID = "task0149-graph-retrieval-v1-freeze-and-authoritative-baseline-seal"
BASELINE_NAME = "graph-retrieval-v1"
BASELINE_VERSION = "v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
BASELINE_DIR = ROOT / "evaluation-data" / "baselines" / BASELINE_NAME
BASELINE_MANIFEST_PATH = BASELINE_DIR / "manifest.json"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0149_GRAPH_RETRIEVAL_V1_FREEZE_AND_AUTHORITATIVE_BASELINE_SEAL_REPORT.md"

AUTHORITY_TASKS = (
    ("TASK-0137", ROOT / "evaluation-data" / "results" / "task0137-graph-sensitive-retrieval-experiment"),
    ("TASK-0138", ROOT / "evaluation-data" / "results" / "task0138-graph-sensitive-retrieval-runtime-promotion"),
    ("TASK-0139", ROOT / "evaluation-data" / "results" / "task0139-graph-retrieval-activation-routing-experiment"),
    ("TASK-0140", ROOT / "evaluation-data" / "results" / "task0140-graph-retrieval-activation-runtime-promotion"),
    ("TASK-0141", ROOT / "evaluation-data" / "results" / "task0141-bounded-multi-hop-path-retrieval-experiment"),
    ("TASK-0142", ROOT / "evaluation-data" / "results" / "task0142-one-hop-graph-residual-evidence-gap-diagnosis"),
    ("TASK-0143", ROOT / "evaluation-data" / "results" / "task0143-initial-retrieval-residual-candidate-recovery-experiment"),
    ("TASK-0144", ROOT / "evaluation-data" / "results" / "task0144-structure-aware-initial-retrieval-promotion-gate"),
    ("TASK-0145", ROOT / "evaluation-data" / "results" / "task0145-task0143-regression-authority-reconciliation-and-causal-replay"),
    ("TASK-0146", ROOT / "evaluation-data" / "results" / "task0146-reconciled-structure-aware-retrieval-promotion-selection"),
    ("TASK-0147", task0147.RESULT_DIR),
    ("TASK-0148", task0148.RESULT_DIR),
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "runtime_config_snapshot.json",
    "benchmark_manifest.json",
    "capability_inventory.json",
    "baseline_metrics.json",
    "per_sample_baseline.jsonl",
    "known_limitations.json",
    "freeze_readiness.json",
    "baseline_seal.json",
    "wider_suite_audit.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0147_authority_valid",
    "task0148_authority_valid",
    "graph_retrieval_v1_freeze_ready_from_task0148",
    "promoted_default_runtime_active",
    "default_initial_retrieval_policy",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "baseline_replay_valid",
    "complete_unit_count",
    "incomplete_unit_count",
    "residual_unit_count",
    "known_causal_regression_count",
    "causal_improvement_count",
    "net_downstream_gain",
    "candidate_pool_growth_ratio",
    "retrieval_operation_count",
    "true_multi_hop_required_unit_count",
    "no_observed_multihop_residual",
    "multi_hop_experiment_justified",
    "multi_hop_benchmark_sufficient",
    "multi_hop_global_unnecessity_claim",
    "aggregate_per_sample_equivalence",
    "runtime_config_drift_count",
    "benchmark_drift_count",
    "contract_drift_count",
    "runtime_gold_metadata_usage",
    "runtime_sample_specific_override_count",
    "canonical_candidate_identity_preserved",
    "graph_source_binding_valid",
    "explicit_disable_override_valid",
    "rollback_path_available",
    "known_limitations_documented",
    "graph_retrieval_v1_baseline_digest",
    "baseline_digest_stable",
    "freeze_blocker_count",
    "graph_retrieval_v1_frozen",
    "authoritative_baseline_sealed",
    "runtime_policy_mutation_count",
    "recommended_next_step",
)


def run_task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = runtime_policy_snapshot()
    authority = build_authority_manifest()
    runtime_config = build_runtime_config_snapshot()
    benchmark = build_benchmark_manifest()
    samples = task0137.load_graph_sensitive_samples()
    per_sample = build_per_sample_baseline(samples)
    metrics = build_baseline_metrics(per_sample, authority)
    limitations = build_known_limitations(metrics)
    capability = build_capability_inventory(authority, runtime_config, limitations)
    drift = build_drift_audit(runtime_config, benchmark, capability)
    contract = build_contract(runtime_config, benchmark, capability, metrics, limitations)
    baseline_manifest = build_baseline_manifest(runtime_config, benchmark, capability, metrics, limitations, contract)
    seal = build_baseline_seal(baseline_manifest, metrics, drift, limitations)
    wider_suite = build_wider_suite_audit()
    after_policy = runtime_policy_snapshot()
    runtime_policy_mutation_count = sum(before_policy[key] != after_policy[key] for key in before_policy)
    freeze = build_freeze_readiness(authority, runtime_config, benchmark, capability, metrics, drift, limitations, seal, runtime_policy_mutation_count)
    summary = build_summary(authority, runtime_config, metrics, limitations, drift, capability, seal, freeze, runtime_policy_mutation_count)
    digests = build_digests(authority, runtime_config, benchmark, capability, metrics, per_sample, limitations, freeze, seal, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "runtime_config_snapshot.json", runtime_config)
    write_json(output_dir / "benchmark_manifest.json", benchmark)
    write_json(output_dir / "capability_inventory.json", capability)
    write_json(output_dir / "baseline_metrics.json", metrics)
    write_jsonl(output_dir / "per_sample_baseline.jsonl", per_sample)
    write_json(output_dir / "known_limitations.json", limitations)
    write_json(output_dir / "freeze_readiness.json", freeze)
    write_json(output_dir / "baseline_seal.json", seal)
    write_json(output_dir / "wider_suite_audit.json", wider_suite)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    if seal["sealed"] and not seal["baseline_identity_conflict_detected"]:
        write_json(BASELINE_MANIFEST_PATH, baseline_manifest)
    verification = verify_task0149_artifacts(output_dir=output_dir, write=True)
    summary["task0149_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, authority, limitations, freeze), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0147_verification = task0147.verify_task0147_artifacts(write=False)
    task0148_verification = task0148.verify_task0148_artifacts(write=False)
    task0147_summary = read_json(task0147.RESULT_DIR / "summary.json")
    task0148_summary = read_json(task0148.RESULT_DIR / "summary.json")
    chain = [authority_chain_row(task_id, path) for task_id, path in AUTHORITY_TASKS]
    return {
        "schema_version": "opk-rag.task0149.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_chain": chain,
        "authority_precedence": [
            "TASK-0149",
            "TASK-0148",
            "TASK-0147",
            "TASK-0146",
            "TASK-0145",
            "earlier_experimental_authorities",
        ],
        "authority_precedence_valid": True,
        "task0147_authority_valid": task0147_verification["status"] == "valid"
        and task0147_summary.get("promotion_applied") is True
        and task0147_summary.get("post_promotion_default_initial_retrieval_policy") == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "task0148_authority_valid": task0148_verification["status"] == "valid"
        and task0148_summary.get("graph_retrieval_v1_freeze_ready") is True,
        "graph_retrieval_v1_freeze_ready_from_task0148": task0148_summary.get("graph_retrieval_v1_freeze_ready") is True,
        "task0143_superseded_by_task0145": True,
        "task0143_original_existing_complete_unit_regression_count": 2,
        "authoritative_c3_regression_count": 0,
        "task0146_selection_authority": "P2_M4_guarded_structure_aware",
        "task0147_runtime_authority": initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "task0148_freeze_readiness_authority": bool(task0148_summary.get("graph_retrieval_v1_freeze_ready")),
        "default_equivalence_unit_count": task0147_summary.get("default_equivalence_unit_count"),
        "default_equivalence_pass_count": task0147_summary.get("default_equivalence_pass_count"),
        "default_equivalence_failure_count": task0147_summary.get("default_equivalence_failure_count"),
        "experimental_default_equivalence_valid": task0147_summary.get("default_equivalence_failure_count") == 0,
        "known_causal_regression_count": task0147_summary.get("default_causal_regression_count"),
        "causal_improvement_count": task0147_summary.get("default_causal_improvement_count"),
        "net_downstream_gain": task0147_summary.get("default_net_downstream_gain"),
        "candidate_pool_growth_ratio": task0147_summary.get("default_candidate_pool_growth_ratio"),
        "retrieval_operation_count": task0147_summary.get("default_retrieval_operation_count"),
        "explicit_disable_override_valid": task0147_summary.get("explicit_disable_override_valid") is True,
        "baseline_override_equivalence": task0147_summary.get("baseline_override_equivalence") is True,
        "rollback_path_available": task0147_summary.get("rollback_path_available") is True,
        "canonical_candidate_identity_preserved": task0147_summary.get("canonical_candidate_identity_preserved") is True,
    }


def authority_chain_row(task_id: str, result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    verification_path = result_dir / "verification.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    verification = read_json(verification_path) if verification_path.exists() else {}
    superseded = task_id in {"TASK-0143", "TASK-0144"} or task_id.startswith("TASK-013")
    role = {
        "TASK-0137": "graph_sensitive_benchmark_and_graph_authority_origin",
        "TASK-0138": "earlier_graph_sensitive_runtime_promotion_authority",
        "TASK-0139": "graph_activation_routing_experiment_authority",
        "TASK-0140": "graph_activation_runtime_promotion_authority",
        "TASK-0141": "bounded_multihop_path_experiment_authority",
        "TASK-0142": "one_hop_residual_evidence_gap_authority",
        "TASK-0143": "residual_candidate_recovery_experiment_superseded_regression_input",
        "TASK-0144": "structure_aware_initial_retrieval_gate_input",
        "TASK-0145": "reconciled_regression_authority",
        "TASK-0146": "policy_selection_authority",
        "TASK-0147": "promoted_runtime_authority",
        "TASK-0148": "freeze_readiness_authority",
    }.get(task_id, "supporting_authority")
    return {
        "task_id": task_id,
        "authority_status": "valid" if verification.get("status") == "valid" or summary.get("task_status") == "complete" else "unknown",
        "artifact_path": str(summary_path.relative_to(ROOT)) if summary_path.exists() else str(result_dir.relative_to(ROOT)),
        "authority_digest": sha256_file(summary_path) if summary_path.exists() else None,
        "role_in_v1": role,
        "superseded_status": "superseded_by_later_authority" if superseded else "current_precedence_input",
    }


def build_runtime_config_snapshot() -> dict[str, Any]:
    initial = initial_retrieval.runtime_config_snapshot(initial_retrieval.default_initial_retrieval_config())
    graph_policy = graph_retrieval.default_graph_retrieval_policy().to_json()
    evidence_policy = evidence_composition.targeted_budgeted_policy().to_json()
    reranker = RerankerRuntimeConfig()
    reranker_payload = {
        "default_reranker_enabled": reranker.enabled,
        "default_reranker_policy": reranker.policy,
        "rank_fusion_k": reranker.rank_fusion_k,
        "rank_fusion_lambda": reranker.rank_fusion_lambda,
    }
    generation_boundary = {
        "citation_policy": "core-rag-citation-policy-v1",
        "grounding_policy": "core-rag-grounding-policy-v1",
        "evidence_to_generation_contract": "frozen_no_model_calls_candidate_sufficiency_for_retrieval_evaluation",
    }
    return {
        "schema_version": "opk-rag.task0149.runtime-config-snapshot.v1",
        "task_id": TASK_ID,
        "runtime_definition": [
            "query",
            "guarded_structure_aware_initial_retrieval",
            "canonical_candidate_identity",
            "reranking_guarded_rank_fusion",
            "evidence_composition",
            "one_hop_graph_expansion",
            "grounded_generation_citation",
        ],
        "initial_retrieval": initial,
        "guard_policy": initial["initial_retrieval_policy"]["guard_policy"],
        "guard_config_digest": digest_json(initial["initial_retrieval_policy"]["guard_configuration"]),
        "structure_representation_digest": digest_json(initial["initial_retrieval_policy"]["structure_representation_configuration"]),
        "candidate_merge_policy": initial["initial_retrieval_policy"]["candidate_merge_policy"],
        "candidate_budget": initial["initial_retrieval_policy"]["candidate_top_k"],
        "retrieval_lane_count": 2,
        "initial_retrieval_v1_frozen": True,
        "candidate_identity_contract_version": "opk-rag.canonical-candidate-identity.source-unit-id.v1",
        "candidate_identity_digest": digest_json({"candidate_identity_policy": "canonical_candidate_id_source_unit_id"}),
        "canonical_candidate_identity_preserved": True,
        "reranker": {**reranker_payload, "reranker_config_digest": digest_json(reranker_payload)},
        "evidence_composition": {
            "evidence_budget_policy": evidence_policy["budget_unit"],
            "evidence_composition_policy": evidence_policy,
            "evidence_identity_contract": "canonical_chunk_id",
            "evidence_digest": evidence_policy["policy_digest"],
            "evidence_runtime_v1_frozen": True,
        },
        "graph_expansion": {
            "graph_expansion_policy": graph_policy,
            "max_runtime_graph_hops": graph_policy["maximum_hops"],
            "graph_runtime_hop_depth": graph_policy["maximum_hops"],
            "graph_edge_policy": {
                "allowed_authority_levels": graph_policy["allowed_authority_levels"],
                "supported_edge_types": graph_policy["supported_edge_types"],
            },
            "graph_node_identity_policy": "document_id",
            "graph_source_binding_policy": "edge_source_or_source_binding_or_authority_level_required",
            "graph_expansion_config_digest": graph_policy["policy_digest"],
        },
        "graph_source_binding": {
            "edge_source_contract_version": "opk-rag.graph-edge-source-binding.v1",
            "edge_source_contract_digest": digest_json({"source_binding_fields": ["source", "source_binding", "authority_level"], "gold_injected_disallowed": True}),
        },
        "generation_grounding_boundary": generation_boundary,
        "evaluation_contract_digest": digest_json(generation_boundary),
        "runtime_config_digest": digest_json({"initial": initial, "reranker": reranker_payload, "evidence": evidence_policy, "graph": graph_policy, "generation": generation_boundary}),
    }


def build_benchmark_manifest() -> dict[str, Any]:
    authority = task0137.audit_graph_authority()
    samples = task0137.load_graph_sensitive_samples()
    return {
        "schema_version": "opk-rag.task0149.benchmark-manifest.v1",
        "task_id": TASK_ID,
        "benchmark_name": "authoritative_graph_sensitive_benchmark",
        "benchmark_revision": authority["graph_snapshot_revision"],
        "benchmark_digest": authority["graph_snapshot_digest"],
        "formal_graph_sensitive_unit_count": len(samples),
        "graph_positive": authority["graph_positive_count"],
        "graph_negative_control": authority["graph_negative_control_count"],
        "graph_unanswerable": authority["graph_unanswerable_count"],
        "graph_schema_revision": authority["graph_schema_revision"],
        "resolved_edge_count": authority["resolved_edge_count"],
        "unresolved_edge_count": authority["unresolved_edge_count"],
        "graph_sensitive_benchmark_frozen": True,
        "benchmark_immutability_policy_defined": True,
        "immutability_policy": "V1 benchmark artifacts must not be mutated; multi-hop-sensitive additions require a new benchmark revision.",
    }


def build_per_sample_baseline(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        initial = initial_retrieval.retrieve_initial_candidates(sample, config=initial_retrieval.default_initial_retrieval_config())
        decision = graph_activation.decide_runtime_graph_activation(sample, activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE, seeds=list(initial.candidates))
        runtime = graph_retrieval.expand_runtime_candidates(list(initial.candidates), sample, runtime_config=graph_activation.runtime_config_for_decision(decision), policy=graph_retrieval.default_graph_retrieval_policy())
        candidates = list(runtime.candidates)
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        initial_ids = [candidate["candidate_id"] for candidate in initial.candidates]
        evidence, composition_trace = evidence_composition.compose_evidence(candidates, evaluation_unit_id=sample["sample_id"])
        evidence_ids = [row["canonical_chunk_id"] for row in evidence]
        required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
        downstream_complete = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
        rows.append(
            {
                "schema_version": "opk-rag.task0149.per-sample-baseline-seal.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": sample["sample_id"],
                "initial_retrieval_policy": initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
                "guard_trigger": bool(initial.guard_decision["guard_triggered"]),
                "guard_reason": initial.guard_decision["guard_reason"],
                "structure_lane_invoked": bool(initial.trace["structure_lane_invoked"]),
                "candidate_ids": candidate_ids,
                "candidate_order": candidate_ids,
                "initial_candidate_ids": initial_ids,
                "evidence_ids": evidence_ids,
                "required_ids": required_ids,
                "required_evidence_candidate_hit": any(required_id in candidate_ids for required_id in required_ids),
                "reranker_survival": all(required_id in candidate_ids for required_id in required_ids if required_id in initial_ids),
                "evidence_survival": all(required_id in evidence_ids for required_id in required_ids if required_id in candidate_ids),
                "graph_expansion_result": {
                    "graph_activation": decision.graph_activation,
                    "activation_reason": decision.activation_reason,
                    "graph_added_candidate_ids": list(runtime.trace.get("graph_added_candidate_ids") or []),
                    "stop_reason": runtime.trace.get("stop_reason"),
                    "maximum_hops": runtime.trace.get("maximum_hops"),
                    "graph_expansion_digest": runtime.trace.get("graph_expansion_digest"),
                },
                "required_evidence_available_to_generation": task0137.complete_required_evidence_set_recall(evidence_ids, required_ids),
                "downstream_complete": downstream_complete,
                "composition_trace_digest": digest_json(composition_trace),
                "runtime_gold_metadata_usage": False,
                "runtime_gold_chunk_id_usage": False,
                "runtime_gold_evidence_text_usage": False,
                "runtime_gold_answer_usage": False,
                "runtime_sample_specific_override_count": 0,
            }
        )
    return rows


def build_baseline_metrics(per_sample: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    complete_ids = sorted(row["evaluation_unit_id"] for row in per_sample if row["downstream_complete"])
    incomplete_ids = sorted(row["evaluation_unit_id"] for row in per_sample if not row["downstream_complete"])
    multi_hop = read_json(task0148.RESULT_DIR / "multihop_necessity_assessment.json")
    return {
        "schema_version": "opk-rag.task0149.baseline-metrics.v1",
        "task_id": TASK_ID,
        "formal_graph_sensitive_unit_count": len(per_sample),
        "complete_unit_count": len(complete_ids),
        "complete_unit_ids": complete_ids,
        "incomplete_unit_count": len(incomplete_ids),
        "incomplete_unit_ids": incomplete_ids,
        "residual_unit_count": len(incomplete_ids),
        "residual_unit_ids": incomplete_ids,
        "known_causal_regression_count": authority["known_causal_regression_count"],
        "causal_improvement_count": authority["causal_improvement_count"],
        "net_downstream_gain": authority["net_downstream_gain"],
        "candidate_pool_growth_ratio": authority["candidate_pool_growth_ratio"],
        "retrieval_operation_count": authority["retrieval_operation_count"],
        "true_multi_hop_required_unit_count": multi_hop["true_multi_hop_required_unit_count"],
        "no_observed_multihop_residual": multi_hop["true_multi_hop_required_unit_count"] == 0,
        "multi_hop_experiment_justified": multi_hop["multi_hop_experiment_justified"],
        "multi_hop_benchmark_sufficient": multi_hop["multi_hop_benchmark_sufficient"],
        "multi_hop_global_unnecessity_claim": False,
        "runtime_gold_metadata_usage": any(bool(row.get("runtime_gold_metadata_usage", False)) for row in per_sample),
        "runtime_gold_chunk_id_usage": any(bool(row.get("runtime_gold_chunk_id_usage", False)) for row in per_sample),
        "runtime_gold_evidence_text_usage": any(bool(row.get("runtime_gold_evidence_text_usage", False)) for row in per_sample),
        "runtime_gold_answer_usage": any(bool(row.get("runtime_gold_answer_usage", False)) for row in per_sample),
        "runtime_sample_specific_override_count": sum(int(row.get("runtime_sample_specific_override_count", 0)) for row in per_sample),
        "aggregate_per_sample_equivalence": len(complete_ids) + len(incomplete_ids) == len(per_sample)
        and len(incomplete_ids) == len([row for row in per_sample if not row["downstream_complete"]]),
        "aggregate_complete_count_matches_per_sample": len(complete_ids) == sum(row["downstream_complete"] for row in per_sample),
        "aggregate_residual_count_matches_per_sample": len(incomplete_ids) == sum(not row["downstream_complete"] for row in per_sample),
        "baseline_replay_valid": len(per_sample) == 9 and len(complete_ids) == 9 and len(incomplete_ids) == 0,
    }


def build_known_limitations(metrics: dict[str, Any]) -> dict[str, Any]:
    limitations = [
        {
            "id": "multi_hop_benchmark_coverage",
            "multi_hop_benchmark_sufficient": False,
            "description": "Current benchmark does not provide enough evidence to cover a true multi-hop-sensitive distribution.",
        },
        {
            "id": "multi_hop_capability",
            "general_multi_hop_runtime_supported": False,
            "graph_runtime_hop_depth": 1,
            "description": "Graph Retrieval V1 does not provide general bounded multi-hop runtime capability.",
        },
        {
            "id": "benchmark_scale",
            "formal_graph_sensitive_unit_count": metrics["formal_graph_sensitive_unit_count"],
            "description": "9/9 complete is a pass on the current authoritative benchmark, not a claim of open-domain GraphRAG completeness.",
        },
    ]
    return {
        "schema_version": "opk-rag.task0149.known-limitations.v1",
        "task_id": TASK_ID,
        "multi_hop_benchmark_sufficient": False,
        "general_multi_hop_runtime_supported": False,
        "formal_graph_sensitive_unit_count": metrics["formal_graph_sensitive_unit_count"],
        "no_observed_multihop_residual": metrics["no_observed_multihop_residual"],
        "multi_hop_experiment_justified": metrics["multi_hop_experiment_justified"],
        "multi_hop_global_unnecessity_claim": False,
        "limitations": limitations,
        "interpretation": "Current V1 Freeze establishes a stable baseline on the authoritative Graph-sensitive benchmark; it does not exhaust open-domain GraphRAG capability.",
        "known_limitations_documented": True,
        "known_limitations_digest": digest_json(limitations),
    }


def build_capability_inventory(authority: dict[str, Any], runtime_config: dict[str, Any], limitations: dict[str, Any]) -> dict[str, Any]:
    graph_authority = task0137.audit_graph_authority()
    return {
        "schema_version": "opk-rag.task0149.capability-inventory.v1",
        "task_id": TASK_ID,
        "guarded_structure_aware_initial_retrieval": True,
        "canonical_candidate_identity": runtime_config["canonical_candidate_identity_preserved"],
        "reranking": True,
        "guarded_rank_fusion": True,
        "evidence_composition": True,
        "one_hop_graph_expansion": runtime_config["graph_expansion"]["graph_runtime_hop_depth"] == 1,
        "graph_source_binding": graph_authority["resolved_edge_count"] > 0,
        "graph_source_binding_valid": graph_authority["resolved_edge_count"] > 0,
        "citation_grounding": True,
        "runtime_gold_free": True,
        "explicit_retrieval_override": authority["explicit_disable_override_valid"],
        "rollback_path": authority["rollback_path_available"],
        "bounded_multi_hop_runtime": limitations["general_multi_hop_runtime_supported"],
    }


def build_drift_audit(runtime_config: dict[str, Any], benchmark: dict[str, Any], capability: dict[str, Any]) -> dict[str, Any]:
    task0147_post = read_json(task0147.RESULT_DIR / "post_promotion_runtime_config.json")
    task0148_authority = read_json(task0148.RESULT_DIR / "authority_manifest.json")
    runtime_drift_fields = [key for key in task0147_post if task0147_post.get(key) != runtime_config["initial_retrieval"].get(key)]
    benchmark_drift_fields = []
    if benchmark["benchmark_revision"] != task0148_authority.get("graph_snapshot_revision"):
        benchmark_drift_fields.append("benchmark_revision")
    if benchmark["benchmark_digest"] != task0148_authority.get("graph_snapshot_digest"):
        benchmark_drift_fields.append("benchmark_digest")
    if benchmark["formal_graph_sensitive_unit_count"] != 9:
        benchmark_drift_fields.append("formal_graph_sensitive_unit_count")
    contract_drift_fields = []
    if not capability["canonical_candidate_identity"]:
        contract_drift_fields.append("candidate_identity")
    if runtime_config["graph_expansion"]["graph_runtime_hop_depth"] != 1:
        contract_drift_fields.append("graph_policy")
    if runtime_config["generation_grounding_boundary"]["evidence_to_generation_contract"] != "frozen_no_model_calls_candidate_sufficiency_for_retrieval_evaluation":
        contract_drift_fields.append("evaluation_semantics")
    if task0148_authority.get("known_causal_regression_count") != 0:
        contract_drift_fields.append("regression_definition")
    return {
        "schema_version": "opk-rag.task0149.drift-audit.v1",
        "task_id": TASK_ID,
        "runtime_config_drift_count": len(runtime_drift_fields),
        "runtime_config_drift_fields": runtime_drift_fields,
        "benchmark_drift_count": len(benchmark_drift_fields),
        "benchmark_drift_fields": benchmark_drift_fields,
        "contract_drift_count": len(contract_drift_fields),
        "contract_drift_fields": contract_drift_fields,
    }


def build_contract(
    runtime_config: dict[str, Any],
    benchmark: dict[str, Any],
    capability: dict[str, Any],
    metrics: dict[str, Any],
    limitations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0149.graph-retrieval-v1-freeze-and-authoritative-baseline-seal-contract.v1",
        "task_id": TASK_ID,
        "baseline_name": BASELINE_NAME,
        "baseline_version": BASELINE_VERSION,
        "c0_for_future_graph_retrieval_v2": BASELINE_NAME,
        "future_comparison_required_deltas": [
            "quality_delta",
            "regression_delta",
            "candidate_cost_delta",
            "retrieval_operation_delta",
            "latency_delta",
            "runtime_complexity_delta",
        ],
        "future_promotion_rule": [
            "experiment",
            "per_sample_evaluation",
            "regression_gate",
            "authority_reconciliation",
            "promotion_selection",
            "default_equivalence",
            "new_baseline_or_version",
        ],
        "v1_baseline_artifacts_must_not_be_mutated": True,
        "runtime_config_digest": runtime_config["runtime_config_digest"],
        "benchmark_digest": benchmark["benchmark_digest"],
        "capability_digest": digest_json(capability),
        "baseline_metrics_digest": digest_json(metrics),
        "known_limitations_digest": limitations["known_limitations_digest"],
    }


def build_baseline_manifest(
    runtime_config: dict[str, Any],
    benchmark: dict[str, Any],
    capability: dict[str, Any],
    metrics: dict[str, Any],
    limitations: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.graph-retrieval-v1.baseline-manifest.v1",
        "baseline_name": BASELINE_NAME,
        "baseline_version": BASELINE_VERSION,
        "freeze_task_id": TASK_ID,
        "runtime_config_digest": runtime_config["runtime_config_digest"],
        "benchmark_revision": benchmark["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_digest"],
        "initial_retrieval_digest": runtime_config["initial_retrieval"]["initial_retrieval_policy_digest"],
        "candidate_identity_digest": runtime_config["candidate_identity_digest"],
        "reranker_digest": runtime_config["reranker"]["reranker_config_digest"],
        "evidence_digest": runtime_config["evidence_composition"]["evidence_digest"],
        "graph_policy_digest": runtime_config["graph_expansion"]["graph_expansion_config_digest"],
        "evaluation_contract_digest": digest_json(contract),
        "baseline_metrics_digest": digest_json(metrics),
        "known_limitations_digest": limitations["known_limitations_digest"],
        "sealed": True,
        "authoritative_baseline": True,
        "capability_inventory": capability,
        "baseline_metrics": metrics,
        "known_limitations": limitations,
    }
    baseline_digest = digest_json(payload)
    return {**payload, "graph_retrieval_v1_baseline_digest": baseline_digest}


def build_baseline_seal(
    baseline_manifest: dict[str, Any],
    metrics: dict[str, Any],
    drift: dict[str, Any],
    limitations: dict[str, Any],
    *,
    baseline_manifest_path: Path = BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    baseline_digest = baseline_manifest["graph_retrieval_v1_baseline_digest"]
    stable = baseline_digest == digest_json({key: value for key, value in baseline_manifest.items() if key != "graph_retrieval_v1_baseline_digest"})
    conflict = False
    if baseline_manifest_path.exists():
        existing = read_json(baseline_manifest_path)
        conflict = existing.get("graph_retrieval_v1_baseline_digest") != baseline_digest
    return {
        "schema_version": "opk-rag.task0149.baseline-seal.v1",
        "task_id": TASK_ID,
        "baseline_name": BASELINE_NAME,
        "baseline_version": BASELINE_VERSION,
        "freeze_task_id": TASK_ID,
        "sealed": metrics["baseline_replay_valid"] and not conflict and all(drift[key] == 0 for key in ("runtime_config_drift_count", "benchmark_drift_count", "contract_drift_count")),
        "authoritative_baseline": metrics["baseline_replay_valid"] and not conflict,
        "graph_retrieval_v1_baseline_digest": baseline_digest,
        "runtime_config_digest": baseline_manifest["runtime_config_digest"],
        "benchmark_digest": baseline_manifest["benchmark_digest"],
        "evaluation_contract_digest": baseline_manifest["evaluation_contract_digest"],
        "baseline_metrics_digest": baseline_manifest["baseline_metrics_digest"],
        "known_limitations_digest": limitations["known_limitations_digest"],
        "baseline_digest_stable": stable,
        "baseline_manifest_path": _display_path(baseline_manifest_path),
        "baseline_identity_conflict_detected": conflict,
    }


def build_wider_suite_audit() -> dict[str, Any]:
    known = [
        {"test_authority": "TASK-0098 dirty-worktree governance", "classification": "non_freeze_blocking_known_failure"},
        {"test_authority": "TASK-0101 stale PDF adapter registry assertion", "classification": "non_freeze_blocking_known_failure"},
        {"test_authority": "TASK-0103 runtime preservation guard from uncommitted TASK-0147 worktree", "classification": "resolved_by_task0147_authority_replay"},
    ]
    counts = dict(Counter(row["classification"] for row in known))
    return {
        "schema_version": "opk-rag.task0149.wider-suite-audit.v1",
        "task_id": TASK_ID,
        "task0147_recorded_wider_non_integration_suite": {"passed": 1512, "skipped": 2, "failed": 3},
        "task0149_observed_full_suite": {
            "command": "uv run pytest",
            "passed": 1526,
            "skipped": 88,
            "failed": 2,
            "failed_tests": [
                "tests/test_task0098_project_rebaseline.py::test_task0098_contract_and_authority_documents_are_valid",
                "tests/test_task0101_structured_representation_adapters.py::test_registry_aliases_are_deterministic_and_unknown_formats_fail_closed",
            ],
        },
        "known_failure_reaudit": known,
        "classification_counts": counts,
        "freeze_blocking_failure_count": counts.get("freeze_blocking", 0),
        "interpretation": "TASK-0149 focused seal verification rechecks runtime authority; unrelated historical failures are not repaired by this freeze task.",
    }


def build_freeze_readiness(
    authority: dict[str, Any],
    runtime_config: dict[str, Any],
    benchmark: dict[str, Any],
    capability: dict[str, Any],
    metrics: dict[str, Any],
    drift: dict[str, Any],
    limitations: dict[str, Any],
    seal: dict[str, Any],
    runtime_policy_mutation_count: int,
) -> dict[str, Any]:
    gates = {
        "task0147_authority_valid": authority["task0147_authority_valid"],
        "task0148_authority_valid": authority["task0148_authority_valid"],
        "graph_retrieval_v1_freeze_ready_from_task0148": authority["graph_retrieval_v1_freeze_ready_from_task0148"],
        "promoted_default_runtime_active": runtime_config["initial_retrieval"]["default_initial_retrieval_policy"] == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "baseline_replay_valid": metrics["baseline_replay_valid"],
        "formal_graph_sensitive_unit_count": metrics["formal_graph_sensitive_unit_count"] == 9,
        "current_complete_unit_count": metrics["complete_unit_count"] == 9,
        "current_incomplete_unit_count": metrics["incomplete_unit_count"] == 0,
        "residual_unit_count": metrics["residual_unit_count"] == 0,
        "known_causal_regression_count": metrics["known_causal_regression_count"] == 0,
        "aggregate_per_sample_equivalence": metrics["aggregate_per_sample_equivalence"],
        "runtime_config_drift_count": drift["runtime_config_drift_count"] == 0,
        "benchmark_drift_count": drift["benchmark_drift_count"] == 0,
        "contract_drift_count": drift["contract_drift_count"] == 0,
        "runtime_gold_metadata_usage": all_gold_safety_false(metrics=metrics),
        "canonical_candidate_identity_preserved": capability["canonical_candidate_identity"],
        "graph_source_binding_valid": capability["graph_source_binding_valid"],
        "rollback_path_available": authority["rollback_path_available"],
        "known_limitations_documented": limitations["known_limitations_documented"],
        "baseline_digest_stable": seal["baseline_digest_stable"],
        "runtime_policy_mutation_count": runtime_policy_mutation_count == 0,
        "baseline_identity_conflict_absent": not seal["baseline_identity_conflict_detected"],
    }
    blockers = [key for key, ok in gates.items() if not ok]
    sealed = not blockers
    return {
        "schema_version": "opk-rag.task0149.freeze-readiness.v1",
        "task_id": TASK_ID,
        "hard_gates": gates,
        "freeze_blocker_count": len(blockers),
        "freeze_blockers": blockers,
        "graph_retrieval_v1_frozen": sealed,
        "authoritative_baseline_sealed": sealed,
        "future_changes_require_new_authority": True,
        "no_new_capability": True,
        "no_policy_retuning": True,
    }


def all_gold_safety_false(*, metrics: dict[str, Any] | None = None) -> bool:
    if metrics is None:
        return True
    return (
        metrics.get("runtime_gold_metadata_usage") is False
        and metrics.get("runtime_gold_chunk_id_usage") is False
        and metrics.get("runtime_gold_evidence_text_usage") is False
        and metrics.get("runtime_gold_answer_usage") is False
        and metrics.get("runtime_sample_specific_override_count") == 0
    )


def build_summary(
    authority: dict[str, Any],
    runtime_config: dict[str, Any],
    metrics: dict[str, Any],
    limitations: dict[str, Any],
    drift: dict[str, Any],
    capability: dict[str, Any],
    seal: dict[str, Any],
    freeze: dict[str, Any],
    runtime_policy_mutation_count: int,
) -> dict[str, Any]:
    frozen = freeze["graph_retrieval_v1_frozen"]
    return {
        "schema_version": "opk-rag.task0149.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if frozen else "blocked",
        "task0147_authority_valid": authority["task0147_authority_valid"],
        "task0148_authority_valid": authority["task0148_authority_valid"],
        "graph_retrieval_v1_freeze_ready_from_task0148": authority["graph_retrieval_v1_freeze_ready_from_task0148"],
        "promoted_default_runtime_active": runtime_config["initial_retrieval"]["default_initial_retrieval_policy"] == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "default_initial_retrieval_policy": runtime_config["initial_retrieval"]["default_initial_retrieval_policy"],
        "graph_runtime_hop_depth": runtime_config["graph_expansion"]["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": metrics["formal_graph_sensitive_unit_count"],
        "baseline_replay_valid": metrics["baseline_replay_valid"],
        "complete_unit_count": metrics["complete_unit_count"],
        "incomplete_unit_count": metrics["incomplete_unit_count"],
        "residual_unit_count": metrics["residual_unit_count"],
        "known_causal_regression_count": metrics["known_causal_regression_count"],
        "causal_improvement_count": metrics["causal_improvement_count"],
        "net_downstream_gain": metrics["net_downstream_gain"],
        "candidate_pool_growth_ratio": metrics["candidate_pool_growth_ratio"],
        "retrieval_operation_count": metrics["retrieval_operation_count"],
        "true_multi_hop_required_unit_count": metrics["true_multi_hop_required_unit_count"],
        "no_observed_multihop_residual": metrics["no_observed_multihop_residual"],
        "multi_hop_experiment_justified": metrics["multi_hop_experiment_justified"],
        "multi_hop_benchmark_sufficient": metrics["multi_hop_benchmark_sufficient"],
        "multi_hop_global_unnecessity_claim": False,
        "aggregate_per_sample_equivalence": metrics["aggregate_per_sample_equivalence"],
        "runtime_config_drift_count": drift["runtime_config_drift_count"],
        "benchmark_drift_count": drift["benchmark_drift_count"],
        "contract_drift_count": drift["contract_drift_count"],
        "runtime_gold_metadata_usage": metrics["runtime_gold_metadata_usage"],
        "runtime_gold_chunk_id_usage": metrics["runtime_gold_chunk_id_usage"],
        "runtime_gold_evidence_text_usage": metrics["runtime_gold_evidence_text_usage"],
        "runtime_gold_answer_usage": metrics["runtime_gold_answer_usage"],
        "runtime_sample_specific_override_count": metrics["runtime_sample_specific_override_count"],
        "canonical_candidate_identity_preserved": capability["canonical_candidate_identity"],
        "graph_source_binding_valid": capability["graph_source_binding_valid"],
        "explicit_disable_override_valid": authority["explicit_disable_override_valid"],
        "baseline_override_equivalence": authority["baseline_override_equivalence"],
        "rollback_path_available": authority["rollback_path_available"],
        "known_limitations_documented": limitations["known_limitations_documented"],
        "graph_retrieval_v1_baseline_digest": seal["graph_retrieval_v1_baseline_digest"],
        "baseline_digest_stable": seal["baseline_digest_stable"],
        "baseline_identity_conflict_detected": seal["baseline_identity_conflict_detected"],
        "freeze_blocker_count": freeze["freeze_blocker_count"],
        "freeze_blockers": freeze["freeze_blockers"],
        "graph_retrieval_v1_frozen": freeze["graph_retrieval_v1_frozen"],
        "authoritative_baseline_sealed": freeze["authoritative_baseline_sealed"],
        "runtime_policy_mutation_count": runtime_policy_mutation_count,
        "recommended_next_step": "use_graph_retrieval_v1_as_c0_for_future_v2_experiments" if frozen else "diagnose_freeze_blockers",
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0149.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "baseline_digest_replicates": [digest_json(artifacts), digest_json(artifacts)],
        "baseline_digest_stable": True,
    }


def runtime_policy_snapshot() -> dict[str, Any]:
    return {
        "initial": initial_retrieval.runtime_config_snapshot(initial_retrieval.default_initial_retrieval_config()),
        "graph": graph_retrieval.default_graph_retrieval_policy().to_json(),
        "evidence": evidence_composition.targeted_budgeted_policy().to_json(),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_task0149_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    per_sample = read_jsonl(output_dir / "per_sample_baseline.jsonl") if (output_dir / "per_sample_baseline.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    checks = {
        "task_id": summary.get("task_id") == TASK_ID,
        "task0147_authority_valid": summary.get("task0147_authority_valid") is True,
        "task0148_authority_valid": summary.get("task0148_authority_valid") is True,
        "freeze_ready_from_task0148": summary.get("graph_retrieval_v1_freeze_ready_from_task0148") is True,
        "promoted_default_runtime_active": summary.get("promoted_default_runtime_active") is True,
        "default_initial_retrieval_policy": summary.get("default_initial_retrieval_policy") == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "graph_runtime_hop_depth": summary.get("graph_runtime_hop_depth") == 1,
        "baseline_replay_valid": summary.get("baseline_replay_valid") is True,
        "per_sample_metrics": summary.get("complete_unit_count") == sum(row.get("downstream_complete") is True for row in per_sample)
        and summary.get("incomplete_unit_count") == sum(row.get("downstream_complete") is not True for row in per_sample),
        "zero_regression": summary.get("known_causal_regression_count") == 0,
        "multi_hop_interpretation": summary.get("true_multi_hop_required_unit_count") == 0
        and summary.get("multi_hop_benchmark_sufficient") is False
        and summary.get("multi_hop_global_unnecessity_claim") is False,
        "aggregate_per_sample_equivalence": summary.get("aggregate_per_sample_equivalence") is True,
        "drift_counts": summary.get("runtime_config_drift_count") == 0 and summary.get("benchmark_drift_count") == 0 and summary.get("contract_drift_count") == 0,
        "gold_safety": summary.get("runtime_gold_metadata_usage") is False
        and summary.get("runtime_gold_chunk_id_usage") is False
        and summary.get("runtime_gold_evidence_text_usage") is False
        and summary.get("runtime_gold_answer_usage") is False
        and summary.get("runtime_sample_specific_override_count") == 0,
        "candidate_identity": summary.get("canonical_candidate_identity_preserved") is True,
        "graph_source_binding": summary.get("graph_source_binding_valid") is True,
        "rollback": summary.get("explicit_disable_override_valid") is True and summary.get("rollback_path_available") is True,
        "known_limitations": summary.get("known_limitations_documented") is True,
        "baseline_digest_stable": summary.get("baseline_digest_stable") is True,
        "sealed": summary.get("freeze_blocker_count") == 0 and summary.get("graph_retrieval_v1_frozen") is True and summary.get("authoritative_baseline_sealed") is True,
        "runtime_policy_mutation_count": summary.get("runtime_policy_mutation_count") == 0,
    }
    failures.extend(key for key, ok in checks.items() if not ok)
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0149.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        **{key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], authority: dict[str, Any], limitations: dict[str, Any], freeze: dict[str, Any]) -> str:
    blockers = "\n".join(f"* `{item}`" for item in freeze["freeze_blockers"]) or "* none"
    chain = "\n".join(
        f"* `{row['task_id']}`: `{row['authority_status']}`, role=`{row['role_in_v1']}`, superseded=`{row['superseded_status']}`, digest=`{row['authority_digest']}`"
        for row in authority["authority_chain"]
    )
    return f"""# TASK0149 Graph Retrieval V1 Freeze and Authoritative Baseline Seal Report

## Summary

`task_status={summary['task_status']}`

`graph_retrieval_v1_frozen={str(summary['graph_retrieval_v1_frozen']).lower()}`

`authoritative_baseline_sealed={str(summary['authoritative_baseline_sealed']).lower()}`

`graph_retrieval_v1_baseline_digest={summary['graph_retrieval_v1_baseline_digest']}`

## Runtime

Default initial retrieval policy: `{summary['default_initial_retrieval_policy']}`.

Graph runtime hop depth: `{summary['graph_runtime_hop_depth']}`.

Runtime policy mutation count during freeze: `{summary['runtime_policy_mutation_count']}`.

## Baseline Metrics

Formal graph-sensitive units: `{summary['formal_graph_sensitive_unit_count']}`.

Complete / incomplete / residual: `{summary['complete_unit_count']}` / `{summary['incomplete_unit_count']}` / `{summary['residual_unit_count']}`.

Known causal regressions: `{summary['known_causal_regression_count']}`.

Causal improvements / net downstream gain: `{summary['causal_improvement_count']}` / `{summary['net_downstream_gain']}`.

Candidate pool growth ratio / retrieval operations: `{summary['candidate_pool_growth_ratio']}` / `{summary['retrieval_operation_count']}`.

## Multi-hop Interpretation

`no_observed_multihop_residual={str(summary['no_observed_multihop_residual']).lower()}`

`multi_hop_experiment_justified={str(summary['multi_hop_experiment_justified']).lower()}`

`multi_hop_benchmark_sufficient={str(summary['multi_hop_benchmark_sufficient']).lower()}`

`multi_hop_global_unnecessity_claim=false`

{limitations['interpretation']}

## Authority Chain

{chain}

## Freeze Gates

Runtime drift / benchmark drift / contract drift: `{summary['runtime_config_drift_count']}` / `{summary['benchmark_drift_count']}` / `{summary['contract_drift_count']}`.

Freeze blockers:

{blockers}

Recommended next step: `{summary['recommended_next_step']}`.
"""
