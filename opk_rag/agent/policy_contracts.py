from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from opk_rag.agent.contracts import AgentActionName

AGENT_POLICY_DECISION_CONTRACT_VERSION = "opk-rag.agent-policy-decision.v1"
AGENT_POLICY_STATE_VIEW_CONTRACT_VERSION = "opk-rag.agent-policy-state-view.v1"
AGENT_POLICY_PROMPT_VERSION = "opk-rag.model-agent-policy-prompt.v3"

PolicyReasonCode = Literal[
    "initial_retrieval_required",
    "evidence_available_requires_assessment",
    "insufficient_evidence_can_expand",
    "expansion_budget_exhausted",
    "answerable_generation_required",
    "generated_answer_requires_verification",
    "verified_answer_can_finish",
    "unanswerable_must_abstain",
    "grounding_rejected_must_abstain",
    "execution_budget_exhausted",
    "invalid_state_must_fail",
]

ALLOWED_POLICY_REASON_CODES = {
    "initial_retrieval_required",
    "evidence_available_requires_assessment",
    "insufficient_evidence_can_expand",
    "expansion_budget_exhausted",
    "answerable_generation_required",
    "generated_answer_requires_verification",
    "verified_answer_can_finish",
    "unanswerable_must_abstain",
    "grounding_rejected_must_abstain",
    "execution_budget_exhausted",
    "invalid_state_must_fail",
}

EXPECTED_TRANSITION_BY_ACTION = {
    "search": "evidence_available",
    "expand_evidence": "evidence_expanded",
    "evaluate_answerability": "answerability_decided",
    "generate_grounded_answer": "generation_available",
    "verify_grounding": "grounding_verified",
    "finish_answer": "terminal_answer",
    "finish_abstain": "terminal_abstain",
    "finish_failure": "terminal_failure",
}

POLICY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "contract_version",
        "action",
        "reason_code",
        "arguments",
        "decision_summary",
        "expected_state_transition",
    ],
    "properties": {
        "contract_version": {"const": AGENT_POLICY_DECISION_CONTRACT_VERSION},
        "action": {"enum": list(EXPECTED_TRANSITION_BY_ACTION)},
        "reason_code": {"enum": sorted(ALLOWED_POLICY_REASON_CODES)},
        "arguments": {"type": "object"},
        "decision_summary": {"type": "string", "maxLength": 200},
        "expected_state_transition": {"enum": sorted(set(EXPECTED_TRANSITION_BY_ACTION.values()))},
    },
}

ACTION_ARGUMENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "search": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "用户原始问题，必须是非空字符串，必须来自 governed state view 的 question 字段。",
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    },
    "expand_evidence": {
        "type": "object",
        "additionalProperties": False,
        "required": ["chunk_id"],
        "properties": {
            "chunk_id": {"type": "string", "minLength": 1},
            "max_items": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    },
    "evaluate_answerability": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
    "generate_grounded_answer": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
    "verify_grounding": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
    "finish_answer": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
    "finish_abstain": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
    "finish_failure": {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
}


@dataclass(frozen=True)
class PolicyDecision:
    contract_version: str
    action: AgentActionName
    reason_code: str
    arguments: dict[str, Any] = field(default_factory=dict)
    decision_summary: str = ""
    expected_state_transition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "action": self.action,
            "reason_code": self.reason_code,
            "arguments": self.arguments,
            "decision_summary": self.decision_summary,
            "expected_state_transition": self.expected_state_transition,
        }


def policy_contract_manifest() -> dict[str, Any]:
    return {
        "schema_version": AGENT_POLICY_DECISION_CONTRACT_VERSION,
        "state_view_contract_version": AGENT_POLICY_STATE_VIEW_CONTRACT_VERSION,
        "prompt_version": AGENT_POLICY_PROMPT_VERSION,
        "decision_schema": POLICY_DECISION_SCHEMA,
        "action_argument_schemas": ACTION_ARGUMENT_SCHEMAS,
        "allowed_reason_codes": sorted(ALLOWED_POLICY_REASON_CODES),
        "expected_transition_by_action": EXPECTED_TRANSITION_BY_ACTION,
        "governance": {
            "model_generates_final_answer": False,
            "answerability_required": True,
            "grounding_verification_required": True,
            "registered_actions_only": True,
            "max_structural_repairs": 1,
        },
    }
