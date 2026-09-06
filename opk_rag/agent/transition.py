from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.agent.contracts import AgentActionName, AgentRuntimeConfig, AgentState, stable_digest

GENERATION_ELIGIBILITY_CONTRACT_VERSION = "opk-rag.generation-eligibility.v1"
RUNTIME_TRANSITION_TRACE_VERSION = "opk-rag.runtime-transition-trace.v1"
RUNTIME_TRANSITION_CONTRACT_VERSION = "opk-rag.runtime-transition-contract.v1"

GENERATION_ELIGIBLE_STATUSES = {"answerable", "partially_answerable"}
EXPLICIT_ELIGIBLE_ABSTAIN_REASONS = {
    "budget_exhausted",
    "provider_unavailable",
    "runtime_contract_failure",
    "explicit_safety_block",
}


@dataclass(frozen=True)
class GenerationEligibility:
    answerability_status: str | None
    generation_eligible: bool
    eligibility_reason_code: str
    evidence_bundle_digest: str | None
    answerability_result_digest: str | None
    allowed_next_actions: tuple[AgentActionName, ...]
    disallowed_next_actions: tuple[AgentActionName, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": GENERATION_ELIGIBILITY_CONTRACT_VERSION,
            "answerability_status": self.answerability_status,
            "generation_eligible": self.generation_eligible,
            "eligibility_reason_code": self.eligibility_reason_code,
            "evidence_bundle_digest": self.evidence_bundle_digest,
            "answerability_result_digest": self.answerability_result_digest,
            "allowed_next_actions": list(self.allowed_next_actions),
            "disallowed_next_actions": list(self.disallowed_next_actions),
        }


def derive_generation_eligibility(state: AgentState, config: AgentRuntimeConfig) -> GenerationEligibility:
    status = state.answerability_state.get("status")
    evidence_digest = state.evidence_state.get("trusted_evidence_digest")
    answerability_digest = state.answerability_state.get("answerability_result_digest")
    disallowed: tuple[AgentActionName, ...] = ("finish_answer",)

    reason = _non_eligible_reason(state, config)
    if reason is not None:
        return GenerationEligibility(
            answerability_status=status,
            generation_eligible=False,
            eligibility_reason_code=reason,
            evidence_bundle_digest=evidence_digest,
            answerability_result_digest=answerability_digest,
            allowed_next_actions=(),
            disallowed_next_actions=("generate_grounded_answer", "finish_answer"),
        )
    return GenerationEligibility(
        answerability_status=status,
        generation_eligible=True,
        eligibility_reason_code="sufficient_grounded_evidence",
        evidence_bundle_digest=evidence_digest,
        answerability_result_digest=answerability_digest,
        allowed_next_actions=("generate_grounded_answer",),
        disallowed_next_actions=disallowed,
    )


def resolve_available_actions(state: AgentState, config: AgentRuntimeConfig) -> tuple[AgentActionName, ...]:
    if state.is_terminal:
        return ()
    if state.step_index >= config.max_steps:
        return ("finish_failure",)
    if not state.evidence_state.get("searched"):
        if state.tool_call_count >= config.max_tool_calls:
            return ("finish_abstain",)
        return ("search",)
    status = state.answerability_state.get("status")
    if status is None:
        if state.tool_call_count >= config.max_tool_calls:
            return ("finish_abstain",)
        return ("evaluate_answerability",)
    if status in {"insufficient_evidence", "partial_evidence"}:
        actions: list[AgentActionName] = []
        if state.tool_call_count < config.max_tool_calls and state.expansion_count < config.max_expansions:
            actions.append("expand_evidence")
        if config.model_planning_enabled and state.tool_call_count < config.max_tool_calls and state.search_round_count < config.max_search_rounds:
            actions.append("search")
        actions.append("finish_abstain")
        return tuple(actions)
    if status in GENERATION_ELIGIBLE_STATUSES:
        if state.generation_state.get("attempted"):
            if not state.verification_state.get("attempted"):
                if state.tool_call_count >= config.max_tool_calls:
                    return ("finish_abstain",)
                return ("verify_grounding",)
            if state.verification_state.get("valid") is True and state.generation_state.get("status") == "answered":
                return ("finish_answer",)
            return ("finish_abstain",)
        if not derive_generation_eligibility(state, config).generation_eligible:
            return ("finish_abstain",)
        if state.tool_call_count >= config.max_tool_calls:
            return ("finish_abstain",)
        if not state.generation_state.get("attempted"):
            return ("generate_grounded_answer",)
    return ("finish_abstain",)


def current_phase(state: AgentState, config: AgentRuntimeConfig) -> str:
    if state.is_terminal:
        return "terminal"
    if state.step_index >= config.max_steps:
        return "budget_exhausted"
    if not state.evidence_state.get("searched"):
        return "search_pending"
    if state.answerability_state.get("status") is None:
        return "answerability_pending"
    if derive_generation_eligibility(state, config).generation_eligible and not state.generation_state.get("attempted"):
        return "generation_pending"
    if state.generation_state.get("attempted") and not state.verification_state.get("attempted"):
        return "verification_pending"
    return "finish_pending"


def transition_trace(
    *,
    step_index: int,
    state_before: AgentState,
    state_after: AgentState,
    selected_action: str,
    action_source: str,
    runtime_validation: str,
    state_before_digest: str,
    state_after_digest: str,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    before_eligibility = derive_generation_eligibility(state_before, config)
    return {
        "contract_version": RUNTIME_TRANSITION_TRACE_VERSION,
        "step_index": step_index,
        "phase_before": current_phase(state_before, config),
        "answerability_status": state_before.answerability_state.get("status"),
        "generation_eligible": before_eligibility.generation_eligible,
        "eligibility_reason_code": before_eligibility.eligibility_reason_code,
        "available_actions": list(resolve_available_actions(state_before, config)),
        "selected_action": selected_action,
        "action_source": action_source,
        "runtime_validation": runtime_validation,
        "phase_after": current_phase(state_after, config),
        "state_before_digest": state_before_digest,
        "state_after_digest": state_after_digest,
    }


def generation_eligibility_contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": GENERATION_ELIGIBILITY_CONTRACT_VERSION,
        "eligible_answerability_statuses": sorted(GENERATION_ELIGIBLE_STATUSES),
        "requires": [
            "latest_answerability_result_digest",
            "valid_evidence_bundle_digest",
            "non_empty_evidence_bundle",
            "no_forbidden_scope",
            "budget_remaining",
            "runtime_not_terminal",
            "generation_not_attempted",
        ],
        "eligible_allowed_next_actions": ["generate_grounded_answer"],
        "eligible_disallowed_next_actions": ["finish_answer"],
    }


def runtime_transition_contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": RUNTIME_TRANSITION_CONTRACT_VERSION,
        "legal_transitions": [
            {"phase": "answerability_evaluated", "generation_eligible": True, "next_phase": "generation_pending"},
            {"phase": "generation_pending", "action": "generate_grounded_answer"},
            {"phase": "generation_completed", "next_phase": "verification_pending"},
            {"phase": "verification_pending", "action": "verify_grounding"},
            {"phase": "verification_passed", "action": "finish_answer"},
            {"phase": "verification_failed", "action": "finish_abstain"},
        ],
        "illegal_transitions": [
            {"phase": "answerability_evaluated", "generation_eligible": True, "action": "finish_answer"},
            {"phase": "answerability_evaluated", "generation_eligible": True, "action": "finish_abstain", "unless_reason_code_in": sorted(EXPLICIT_ELIGIBLE_ABSTAIN_REASONS)},
        ],
    }


def answerability_result_digest(payload: dict[str, Any], evidence_digest: str | None) -> str:
    return stable_digest(
        {
            "answerability": {
                "status": payload.get("status"),
                "reason_code": payload.get("reason_code"),
                "evidence_count": payload.get("evidence_count"),
                "evidence_chunk_ids": payload.get("evidence_chunk_ids", []),
            },
            "evidence_bundle_digest": evidence_digest,
        }
    )


def _non_eligible_reason(state: AgentState, config: AgentRuntimeConfig) -> str | None:
    status = state.answerability_state.get("status")
    if state.is_terminal:
        return "runtime_terminal_state"
    if state.step_index >= config.max_steps or (not state.verification_state.get("attempted") and state.tool_call_count >= config.max_tool_calls):
        return "budget_exhausted"
    if state.generation_state.get("attempted"):
        return "generation_already_attempted"
    if status not in GENERATION_ELIGIBLE_STATUSES:
        return "answerability_not_generation_eligible"
    if not state.evidence_state.get("searched"):
        return "missing_evidence_bundle"
    evidence_digest = state.evidence_state.get("trusted_evidence_digest")
    if not evidence_digest:
        return "missing_evidence_bundle_digest"
    evidence_count = state.answerability_state.get("evidence_count")
    chunk_ids = state.evidence_state.get("chunk_ids") or []
    citation_ids = state.evidence_state.get("citation_ids") or []
    if evidence_count == 0 or (not chunk_ids and not citation_ids):
        return "empty_evidence_bundle"
    answerability_digest = state.answerability_state.get("answerability_result_digest")
    if not answerability_digest:
        return "missing_answerability_result_digest"
    recorded_evidence_digest = state.answerability_state.get("evidence_bundle_digest")
    if recorded_evidence_digest != evidence_digest:
        return "stale_answerability_result"
    if state.answerability_state.get("forbidden_scope_present") is True:
        return "forbidden_scope_present"
    return None
