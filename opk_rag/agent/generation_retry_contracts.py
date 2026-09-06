from __future__ import annotations

from pathlib import Path
from typing import Any

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.generation_retry import GENERATION_RETRY_POLICY_ID, RETRY_OUTCOMES
from opk_rag.evaluation.core_rag_benchmark import file_digest, read_json, write_json
from opk_rag.evaluation.stable_generation_retry_eligibility import build_retry_eligibility_proposal

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = GENERATION_RETRY_POLICY_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0076_governed_generation_retry_contract.json"
RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0076-governed-generation-retry"
BENCHMARK_DIR = ROOT / "evaluation-data" / "core-rag-benchmark-v1"

AUTHORITATIVE_INPUTS = {
    "task0071_contract": ROOT / "evaluation-data" / "contracts" / "task0071_governed_agent_recovery_contract.json",
    "task0072_contract": ROOT / "evaluation-data" / "contracts" / "task0072_recovery_eligibility_diagnosis_contract.json",
    "task0073_contract": ROOT / "evaluation-data" / "contracts" / "task0073_post_generation_observability_contract.json",
    "task0074_contract": ROOT / "evaluation-data" / "contracts" / "task0074_reference_runtime_stability_contract.json",
    "task0075_contract": ROOT / "evaluation-data" / "contracts" / "task0075_post_generation_instability_diagnosis_contract.json",
    "task0075_proposal": ROOT / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "stable_generation_retry_eligibility_proposal.json",
    "task0075_readiness": ROOT / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "task0076_readiness_gates.json",
    "benchmark_manifest": BENCHMARK_DIR / "benchmark_manifest.json",
    "benchmark_questions": BENCHMARK_DIR / "question_set.jsonl",
    "benchmark_annotations": BENCHMARK_DIR / "annotations.jsonl",
}


def build_generation_retry_contract(*, branch: str, head: str, input_identities: dict[str, Any] | None = None) -> dict[str, Any]:
    proposal = build_retry_eligibility_proposal()
    contract: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "schema_version": "opk-rag.governed-single-attempt-generation-retry-contract.v1",
        "task_id": "TASK-0076",
        "code_identity": {"branch": branch, "head": head},
        "authoritative_input_identities": input_identities or input_identity_manifest(),
        "task0075_proposal_identity": {
            "proposal_id": proposal["proposal_id"],
            "proposal_schema_version": proposal["schema_version"],
            "proposal_status_required": "proposal_only",
            "proposal_digest": stable_digest(proposal),
        },
        "reference_runtime_identity": {
            "generation_prompt_identity": "answer-prompt-v4",
            "generation_schema_identity": "answer-response-v3",
            "runtime_retry_default_enabled": False,
        },
        "retry_eligibility_schema": "opk-rag.generation-retry-eligibility-input.v1",
        "retry_rule_order": proposal["deterministic_rule_order"],
        "retry_disqualifiers": [row["signal"] for row in proposal["disqualifying_rules"]]
        + [
            "provider_server_failure",
            "retrieval_failure",
            "database_failure",
            "answer_draft_present",
            "valid_grounded_answer_already_exists",
        ],
        "retry_request_identity_requirements": {
            "same_logical_request_digest_required": True,
            "attempt_metadata_excluded_from_digest": True,
        },
        "maximum_generation_retries": 1,
        "maximum_total_generation_calls": 2,
        "retrieval_retry_allowed": False,
        "query_reformulation_allowed": False,
        "recursive_retry_allowed": False,
        "provider_substitution_allowed": False,
        "model_substitution_allowed": False,
        "prompt_change_allowed": False,
        "provider_parameter_change_allowed": False,
        "evidence_bundle_change_allowed": False,
        "provider_identity": "frozen_reference_runtime_provider",
        "model_identity": "frozen_reference_runtime_model",
        "generation_prompt_identity": "answer-prompt-v4",
        "generation_schema_identity": "answer-response-v3",
        "provider_parameter_identity": "frozen_reference_runtime_answer_config",
        "evidence_identity_requirements": {
            "selected_evidence_identities_same": True,
            "selected_evidence_order_same": True,
            "selected_evidence_content_same": True,
        },
        "retry_outcome_taxonomy": sorted(RETRY_OUTCOMES),
        "state_transitions": {
            "GENERATION_ATTEMPT_1": ["CITATION_VALIDATION", "FINAL_ABSTENTION", "GENERATION_RETRY_ELIGIBILITY", "TERMINAL_ERROR"],
            "GENERATION_RETRY_ELIGIBILITY": ["GENERATION_RETRY", "FINAL_ABSTENTION", "TERMINAL_ERROR"],
            "GENERATION_RETRY": ["CITATION_VALIDATION", "FINAL_ABSTENTION", "TERMINAL_ERROR"],
            "RETRY_CITATION_VALIDATION": ["RETRY_GROUNDING_VALIDATION", "FINAL_ABSTENTION"],
            "RETRY_GROUNDING_VALIDATION": ["FINAL_ANSWER", "FINAL_ABSTENTION"],
        },
        "trace_schema": "opk-rag.generation-retry-trace.v1",
        "experiment_design": {
            "variant_a1": "existing_agent_path_without_generation_retry",
            "variant_a2": "governed_single_attempt_generation_retry",
            "paired_execution_preferred": True,
            "development_replicate_count": 2,
        },
        "timeout_rules": {"per_sample_timeout_seconds": 300, "hidden_sample_retry_allowed": False},
        "evaluation_metrics": [
            "end_to_end_accuracy",
            "safe_action_accuracy",
            "retry_eligible_count",
            "retry_attempted_count",
            "retry_grounded_answer_count",
            "over_abstention_count",
            "provider_call_count",
            "latency_p50_p95",
        ],
        "safety_gates": {
            "citation_validation_required": True,
            "grounding_validation_required": True,
            "unsupported_claim_bypass_allowed": False,
            "safety_terminal_retry_count_required": 0,
            "correct_unanswerable_retry_count_required": 0,
        },
        "promotion_eligibility_gates": {
            "minimum_recovered_grounded_answers_per_replicate": 3,
            "minimum_total_recovered_grounded_answers": 6,
            "eligible_sample_set_agreement_rate_minimum": 0.8,
            "retry_outcome_exact_agreement_rate_minimum": 0.75,
            "a2_p95_latency_multiplier_maximum": 2.0,
        },
        "privacy_policy": "privacy_safe_digests_counts_and_bounded_metadata_only",
        "artifact_schema": "opk-rag.task0076-artifacts.v1",
        "verifier_rules": [
            "contract_digest_match",
            "replicate_terminal_rows_complete",
            "maximum_generation_retries_observed_lte_1",
            "maximum_total_generation_calls_observed_lte_2",
            "no_retrieval_retry",
            "no_query_reformulation_retry",
            "no_recursive_retry",
            "request_identity_match_for_attempt_2",
            "aggregate_recomputed",
            "privacy_scan_pass",
        ],
    }
    contract["contract_digest"] = stable_digest({key: value for key, value in contract.items() if key != "contract_digest"})
    return contract


def input_identity_manifest() -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for name, path in AUTHORITATIVE_INPUTS.items():
        identities[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "sha256": file_digest(path) if path.exists() else None,
        }
    return {"schema_version": "opk-rag.task0076-input-identity.v1", "inputs": identities}


def validate_authoritative_inputs() -> dict[str, Any]:
    manifest = input_identity_manifest()
    inputs = manifest["inputs"]
    findings: list[dict[str, Any]] = []
    for name, identity in inputs.items():
        if not identity["exists"]:
            findings.append({"code": "missing_authoritative_input", "name": name})
    proposal_path = AUTHORITATIVE_INPUTS["task0075_proposal"]
    readiness_path = AUTHORITATIVE_INPUTS["task0075_readiness"]
    if proposal_path.exists():
        proposal = read_json(proposal_path)
        if proposal.get("status") != "proposal_only":
            findings.append({"code": "task0075_proposal_status_not_proposal_only", "actual": proposal.get("status")})
    if readiness_path.exists():
        readiness = read_json(readiness_path)
        if readiness.get("task0076_readiness") != "ready":
            findings.append({"code": "task0076_readiness_not_ready", "actual": readiness.get("task0076_readiness")})
    return manifest | {"validation_status": "pass" if not findings else "fail", "findings": findings}


def write_generation_retry_contract(path: Path = CONTRACT_PATH, *, branch: str, head: str) -> dict[str, Any]:
    contract = build_generation_retry_contract(branch=branch, head=head, input_identities=validate_authoritative_inputs())
    write_json(path, contract)
    return contract


def load_generation_retry_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    return read_json(path)
