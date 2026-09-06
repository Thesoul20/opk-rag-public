from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.evidence_comparison import EVIDENCE_COMPARISON_CONTRACT_VERSION
from opk_rag.agent.query_plan import query_plan_contract_manifest
from opk_rag.agent.query_reformulation import query_reformulation_prompt_manifest
from opk_rag.agent.recovery_state import AGENT_RECOVERY_STATE_CONTRACT_VERSION
from opk_rag.agent.recovery_trace import AGENT_RECOVERY_TRACE_CONTRACT_VERSION

AGENT_RECOVERY_CONTRACT_ID = "opk-rag.governed-single-agent-recovery-loop.v1"
AGENT_RECOVERY_CONTRACT_SCHEMA_VERSION = "opk-rag.governed-single-agent-recovery-loop-contract.v1"
AGENT_RECOVERY_TASK_ID = "TASK-0071"

ALLOWED_RECOVERY_STATES = (
    "START",
    "INITIAL_RETRIEVAL",
    "INITIAL_EVIDENCE_INSPECTION",
    "RECOVERY_ELIGIBILITY",
    "QUERY_REFORMULATION",
    "RECOVERY_RETRIEVAL",
    "EVIDENCE_COMPARISON",
    "FINAL_ANSWERABILITY",
    "GENERATION",
    "GENERATION_RETRY_ELIGIBILITY",
    "GENERATION_RETRY",
    "RETRY_CITATION_VALIDATION",
    "RETRY_GROUNDING_VALIDATION",
    "CITATION_VALIDATION",
    "GROUNDING_VALIDATION",
    "FINAL_ANSWER",
    "FINAL_ABSTENTION",
    "TERMINAL_ERROR",
)

ALLOWED_RECOVERY_TRANSITIONS = (
    ("START", "INITIAL_RETRIEVAL"),
    ("INITIAL_RETRIEVAL", "INITIAL_EVIDENCE_INSPECTION"),
    ("INITIAL_RETRIEVAL", "TERMINAL_ERROR"),
    ("INITIAL_EVIDENCE_INSPECTION", "GENERATION"),
    ("INITIAL_EVIDENCE_INSPECTION", "RECOVERY_ELIGIBILITY"),
    ("INITIAL_EVIDENCE_INSPECTION", "FINAL_ABSTENTION"),
    ("INITIAL_EVIDENCE_INSPECTION", "TERMINAL_ERROR"),
    ("RECOVERY_ELIGIBILITY", "QUERY_REFORMULATION"),
    ("RECOVERY_ELIGIBILITY", "FINAL_ABSTENTION"),
    ("QUERY_REFORMULATION", "RECOVERY_RETRIEVAL"),
    ("QUERY_REFORMULATION", "FINAL_ABSTENTION"),
    ("QUERY_REFORMULATION", "TERMINAL_ERROR"),
    ("RECOVERY_RETRIEVAL", "EVIDENCE_COMPARISON"),
    ("RECOVERY_RETRIEVAL", "TERMINAL_ERROR"),
    ("EVIDENCE_COMPARISON", "FINAL_ANSWERABILITY"),
    ("EVIDENCE_COMPARISON", "FINAL_ABSTENTION"),
    ("FINAL_ANSWERABILITY", "GENERATION"),
    ("FINAL_ANSWERABILITY", "FINAL_ABSTENTION"),
    ("FINAL_ANSWERABILITY", "TERMINAL_ERROR"),
    ("GENERATION", "CITATION_VALIDATION"),
    ("GENERATION", "GENERATION_RETRY_ELIGIBILITY"),
    ("GENERATION", "FINAL_ABSTENTION"),
    ("GENERATION", "TERMINAL_ERROR"),
    ("GENERATION_RETRY_ELIGIBILITY", "GENERATION_RETRY"),
    ("GENERATION_RETRY_ELIGIBILITY", "FINAL_ABSTENTION"),
    ("GENERATION_RETRY_ELIGIBILITY", "TERMINAL_ERROR"),
    ("GENERATION_RETRY", "RETRY_CITATION_VALIDATION"),
    ("GENERATION_RETRY", "FINAL_ABSTENTION"),
    ("GENERATION_RETRY", "TERMINAL_ERROR"),
    ("RETRY_CITATION_VALIDATION", "RETRY_GROUNDING_VALIDATION"),
    ("RETRY_CITATION_VALIDATION", "FINAL_ABSTENTION"),
    ("RETRY_GROUNDING_VALIDATION", "FINAL_ANSWER"),
    ("RETRY_GROUNDING_VALIDATION", "FINAL_ABSTENTION"),
    ("CITATION_VALIDATION", "GROUNDING_VALIDATION"),
    ("CITATION_VALIDATION", "FINAL_ABSTENTION"),
    ("CITATION_VALIDATION", "TERMINAL_ERROR"),
    ("GROUNDING_VALIDATION", "FINAL_ANSWER"),
    ("GROUNDING_VALIDATION", "FINAL_ABSTENTION"),
    ("GROUNDING_VALIDATION", "TERMINAL_ERROR"),
)


@dataclass(frozen=True)
class AgentRecoveryContract:
    benchmark_id: str
    benchmark_version: str
    benchmark_file_hashes: dict[str, str]
    corpus_snapshot_identity: str
    canonical_evidence_identity: str
    core_tool_layer_identity: str
    retrieval_configuration_identity: str
    answerability_contract_identity: str
    generation_contract_identity: str
    citation_contract_identity: str
    grounding_contract_identity: str
    provider_runtime_identity: str
    model_identity: str
    code_git_head: str
    artifact_contract_path: str = "evaluation-data/contracts/task0071_governed_agent_recovery_contract.json"
    artifact_result_root: str = "evaluation-data/results/task0071-governed-agent-recovery"
    artifact_report_path: str = "docs/TASK0071_GOVERNED_SINGLE_AGENT_RAG_RECOVERY_REPORT.md"
    contract_id: str = AGENT_RECOVERY_CONTRACT_ID
    schema_version: str = AGENT_RECOVERY_CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "task_id": AGENT_RECOVERY_TASK_ID,
            "artifact_paths": {
                "contract_path": self.artifact_contract_path,
                "result_root": self.artifact_result_root,
                "report_path": self.artifact_report_path,
                "runtime_identity_path": f"{self.artifact_result_root}/runtime_identity.json",
                "blocked_status_path": f"{self.artifact_result_root}/blocked_status.json",
            },
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_file_hashes": self.benchmark_file_hashes,
            "corpus_snapshot_identity": self.corpus_snapshot_identity,
            "canonical_evidence_identity": self.canonical_evidence_identity,
            "core_tool_layer_identity": self.core_tool_layer_identity,
            "retrieval_configuration_identity": self.retrieval_configuration_identity,
            "answerability_contract_identity": self.answerability_contract_identity,
            "generation_contract_identity": self.generation_contract_identity,
            "citation_contract_identity": self.citation_contract_identity,
            "grounding_contract_identity": self.grounding_contract_identity,
            "provider_runtime_identity": self.provider_runtime_identity,
            "model_identity": self.model_identity,
            "agent_state_schema": AGENT_RECOVERY_STATE_CONTRACT_VERSION,
            "allowed_states": list(ALLOWED_RECOVERY_STATES),
            "allowed_transitions": [list(item) for item in ALLOWED_RECOVERY_TRANSITIONS],
            "transition_priority": [
                "answer_when_answerability_and_validation_pass",
                "recover_once_when_insufficient_and_permitted",
                "abstain_on_safety_or_no_gain",
                "typed_error_on_infrastructure_or_contract_failure",
            ],
            "tool_call_budget": {
                "max_retrieval_calls": 2,
                "max_reformulation_calls": 1,
                "max_generation_calls": 1,
                "max_agent_transitions": 14,
                "max_total_tool_calls": 6,
            },
            "retry_budget": {"recursive_retries": 0, "query_reformulation_repairs": 1},
            "query_reformulation_prompt_identity": query_reformulation_prompt_manifest(),
            "query_reformulation_output_schema": query_plan_contract_manifest(),
            "query_reformulation_validation_rules": [
                "non_empty",
                "not_identical_to_original",
                "closed_enum_strategy",
                "no_forbidden_or_gold_content",
                "max_query_length_256",
            ],
            "evidence_comparison_rules": {"contract_version": EVIDENCE_COMPARISON_CONTRACT_VERSION, "gold_annotations_allowed": False},
            "termination_rules": ["FINAL_ANSWER", "FINAL_ABSTENTION", "TERMINAL_ERROR"],
            "timeout_rules": {
                "retrieval_timeout": "TERMINAL_ERROR",
                "reformulation_timeout": "TERMINAL_ERROR",
                "generation_timeout": "TERMINAL_ERROR",
            },
            "experiment_variants": ["C0", "A1"],
            "evaluation_metrics": ["end_to_end_accuracy", "safe_action_accuracy", "recovery_attempted_count", "recovered_grounded_answer_count"],
            "promotion_gates": ["runtime_reproducibility", "safety_preservation", "recovery_value", "bounded_execution", "privacy_governance"],
            "privacy_policy": {"raw_private_content_in_trace": False, "gold_annotations_in_runtime": False, "secrets_in_artifacts": False},
            "trace_schema": AGENT_RECOVERY_TRACE_CONTRACT_VERSION,
            "output_schema": "opk-rag.governed-agent-recovery-result.v1",
            "code_git_head": self.code_git_head,
        }
        payload["contract_digest"] = stable_digest({key: value for key, value in payload.items() if key != "contract_digest"})
        return payload

    @property
    def contract_digest(self) -> str:
        return self.to_dict()["contract_digest"]


def validate_recovery_transition(from_state: str, to_state: str) -> None:
    if (from_state, to_state) not in ALLOWED_RECOVERY_TRANSITIONS:
        raise ValueError(f"unknown recovery transition: {from_state}->{to_state}")
