from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from opk_rag.agentic_v2.action import validate_agent_action
from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.decision import validate_agent_decision
from opk_rag.agentic_v2.guard import AgentGuard
from opk_rag.agentic_v2.observation import build_agent_observation
from opk_rag.agentic_v2.policy_provider import FakePolicyProvider
from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime
from opk_rag.agentic_v2.schemas import contract_digest
from opk_rag.agentic_v2.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0242"
SCHEMA = "opk-rag.task0242.deterministic-agent-guard-and-decision-to-action-validation.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0242-deterministic-agent-guard-and-decision-to-action-validation"
CONTRACT = ROOT / "evaluation-data/contracts/task0242_deterministic_agent_guard_and_decision_to_action_validation.json"
TASK0240 = ROOT / "evaluation-data/results/task0240-llm-agentic-rag-typed-contracts-and-pydantic-foundation/summary.json"
TASK0241 = ROOT / "evaluation-data/results/task0241-llm-agent-policy-provider-and-structured-decision-runtime/summary.json"
REGRESSION = RESULT_DIR / "regression.json"

PRODUCTION_PREFIXES = (
    "opk_rag/search/", "opk_rag/answer/", "opk_rag/retrieval/", "opk_rag/graph/", "opk_rag/reranking/",
    "opk_rag/embedding/", "opk_rag/indexing/", "opk_rag/core_tools/", "opk_rag/agent/", "opk_rag/cli.py",
)
FORBIDDEN_TRACE_TERMS = {
    "api_key", "database_url", "authorization", "gold_evidence", "gold_citation", "expected_action",
    "required_claims", "forbidden_claims", "benchmark_score", "hidden_reasoning", "chain_of_thought",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_changed_paths() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return sorted(paths)


def _state(**updates: Any) -> AgentState:
    base: dict[str, Any] = {
        "run_id": "task0242-eval",
        "original_query": "How does rollback consistency work?",
        "current_query": "How does rollback consistency work?",
        "candidate_count": 0,
        "evidence_count": 0,
        "source_count": 0,
        "answerability_status": "unknown",
    }
    base.update(updates)
    return AgentState(**base)


def _obs(state: AgentState):
    return build_agent_observation(state)


def _decision(action: str, arguments: dict[str, Any], reason: str = "initial_retrieval_needed"):
    return validate_agent_decision({
        "contract_version": "opk-rag.agentic-v2.decision.v1",
        "proposed_action": action,
        "arguments": arguments,
        "reason_code": reason,
        "confidence": 0.8,
        "short_reason": "Public summary only.",
    })


def _validate_guard_matrix() -> dict[str, Any]:
    guard = AgentGuard()
    fresh = _state()
    allow = guard.validate(state=fresh, observation=_obs(fresh), decision=_decision("hybrid_search", {"query": fresh.current_query, "top_k": 10}))
    modify = guard.validate(state=fresh, observation=_obs(fresh), decision=_decision("hybrid_search", {"query": fresh.current_query, "top_k": 20}))
    exhausted = _state(retrieval_call_count=2)
    reject = guard.validate(state=exhausted, observation=_obs(exhausted), decision=_decision("hybrid_search", {"query": exhausted.current_query}))
    terminal = _state(status="finished", termination_reason="done")
    terminate = guard.validate(state=terminal, observation=_obs(terminal), decision=_decision("hybrid_search", {"query": terminal.current_query}))
    return {
        "allow_supported": allow.guard_decision.decision == "allow",
        "modify_supported": modify.guard_decision.decision == "modify" and modify.validated_action.arguments.top_k == 10,
        "reject_supported": reject.guard_decision.decision == "reject",
        "terminate_supported": terminate.guard_decision.decision == "terminate" and terminate.termination is not None,
        "metrics": guard.metrics.summary(),
    }


def _budget_matrix() -> dict[str, bool]:
    guard = AgentGuard()
    step = _state(step_index=3)
    retrieval = _state(retrieval_call_count=2)
    rewrite = _state(rewrite_count=1)
    graph_call = _state(graph_call_count=1)
    graph_hop = _state(graph_hop_count=1)
    return {
        "step_budget_enforced": guard.validate(state=step, observation=_obs(step), decision=_decision("hybrid_search", {"query": step.current_query})).guard_decision.reason_code == "agent_step_budget_exhausted",
        "retrieval_budget_enforced": guard.validate(state=retrieval, observation=_obs(retrieval), decision=_decision("hybrid_search", {"query": retrieval.current_query})).guard_decision.reason_code == "retrieval_budget_exhausted",
        "rewrite_budget_enforced": guard.validate(state=rewrite, observation=_obs(rewrite), decision=_decision("rewrite_query", {"query": "rewritten"}, "query_ambiguity")).guard_decision.reason_code == "query_rewrite_budget_exhausted",
        "graph_call_budget_enforced": guard.validate(state=graph_call, observation=_obs(graph_call), decision=_decision("graph_search", {"query": graph_call.current_query, "hop_limit": 1}, "cross_document_relation_needed")).guard_decision.reason_code == "graph_call_budget_exhausted",
        "graph_hop_budget_enforced": guard.validate(state=graph_hop, observation=_obs(graph_hop), decision=_decision("graph_search", {"query": graph_hop.current_query, "hop_limit": 1}, "cross_document_relation_needed")).guard_decision.reason_code == "graph_hop_budget_exhausted",
    }


def _decision_action_matrix() -> dict[str, bool]:
    guard = AgentGuard()
    duplicate_graph = _state(previous_actions=("graph_search",))
    duplicate_hybrid = _state(previous_actions=("hybrid_search",), retrieval_call_count=1)
    query_change = _state()
    finish_bad = _state(evidence_count=1, source_count=1, answerability_status="insufficient_evidence")
    abstain = _state(answerability_status="unanswerable")
    return {
        "duplicate_graph_recovery_rejected": guard.validate(state=duplicate_graph, observation=_obs(duplicate_graph), decision=_decision("graph_search", {"query": duplicate_graph.current_query, "hop_limit": 1}, "cross_document_relation_needed")).guard_decision.reason_code == "graph_recovery_already_used",
        "duplicate_retrieval_without_query_change_rejected": guard.validate(state=duplicate_hybrid, observation=_obs(duplicate_hybrid), decision=_decision("hybrid_search", {"query": duplicate_hybrid.current_query})).guard_decision.reason_code == "duplicate_retrieval_without_query_change",
        "unauthorized_query_change_rejected": guard.validate(state=query_change, observation=_obs(query_change), decision=_decision("hybrid_search", {"query": "different query"})).guard_decision.reason_code == "unauthorized_query_change",
        "finish_answerability_gate_valid": guard.validate(state=finish_bad, observation=_obs(finish_bad), decision=_decision("finish", {"reason_code": "sufficient_evidence"}, "sufficient_evidence")).guard_decision.reason_code == "finish_not_authorized_by_answerability",
        "abstain_policy_valid": guard.validate(state=abstain, observation=_obs(abstain), decision=_decision("abstain", {"reason_code": "insufficient_evidence"}, "insufficient_evidence")).guard_decision.decision == "allow",
    }


def _deterministic_replay() -> dict[str, Any]:
    state = _state()
    obs = _obs(state)
    decision = _decision("hybrid_search", {"query": state.current_query, "top_k": 20})
    guard = AgentGuard()
    first = guard.validate(state=state, observation=obs, decision=decision)
    second = guard.validate(state=state, observation=obs, decision=decision)
    canonical_first = {
        "guard": first.guard_decision.model_dump(mode="json"),
        "termination": None if first.termination is None else first.termination.model_dump(mode="json"),
        "allowed_actions": first.trace.allowed_actions,
        "parameter_modifications": first.trace.parameter_modifications,
        "decision_id": first.trace.decision_id,
    }
    canonical_second = {
        "guard": second.guard_decision.model_dump(mode="json"),
        "termination": None if second.termination is None else second.termination.model_dump(mode="json"),
        "allowed_actions": second.trace.allowed_actions,
        "parameter_modifications": second.trace.parameter_modifications,
        "decision_id": second.trace.decision_id,
    }
    return {
        "deterministic_replay_valid": canonical_first == canonical_second,
        "result_digest": stable_digest(canonical_first),
    }


def _pydantic_guard_separation() -> dict[str, bool]:
    pydantic_rejected = False
    try:
        _decision("graph_search", {"query": "q", "hop_limit": 5}, "cross_document_relation_needed")
    except ValidationError:
        pydantic_rejected = True
    state = _state(graph_call_count=1)
    valid = _decision("graph_search", {"query": state.current_query, "hop_limit": 1}, "cross_document_relation_needed")
    guard_rejected = AgentGuard().validate(state=state, observation=_obs(state), decision=valid).guard_decision.reason_code == "graph_call_budget_exhausted"
    return {
        "pydantic_structural_rejection_valid": pydantic_rejected,
        "guard_runtime_rejection_valid": guard_rejected,
        "pydantic_guard_separation_valid": pydantic_rejected and guard_rejected,
    }


def _fail_closed() -> bool:
    class ExplodingResolver(AllowedActionResolver):
        def resolve(self, **kwargs):
            raise RuntimeError("forced evaluator failure")
    state = _state()
    result = AgentGuard(resolver=ExplodingResolver()).validate(state=state, observation=_obs(state), decision=_decision("hybrid_search", {"query": state.current_query}))
    return result.guard_decision.decision == "reject" and result.guard_decision.reason_code == "guard_internal_error" and result.validated_action is None


def _policy_guard_integration() -> dict[str, Any]:
    state = _state()
    obs = _obs(state)
    resolver = AllowedActionResolver()
    allowed = resolver.resolve(state=state, observation=obs)
    output = {
        "contract_version": "opk-rag.agentic-v2.decision.v1",
        "proposed_action": "hybrid_search",
        "arguments": {"query": state.current_query, "top_k": 10},
        "reason_code": "initial_retrieval_needed",
        "confidence": 0.9,
        "short_reason": "Need governed retrieval.",
    }
    policy = LLMAgentPolicyRuntime(provider=FakePolicyProvider(outputs=[output]))
    policy_result = policy.decide(observation=obs, allowed_actions=allowed)
    guard_result = AgentGuard(resolver=resolver).validate(state=state, observation=obs, decision=policy_result.decision)
    return {
        "policy_guard_integration_valid": guard_result.guard_decision.decision == "allow" and guard_result.validated_action.action == "hybrid_search",
        "allowed_action_agreement_valid": guard_result.trace.allowed_actions == allowed,
        "tool_result_present": hasattr(guard_result, "tool_result"),
    }


def _security() -> dict[str, Any]:
    state = _state(current_query="postgresql://user:secret@host/db")
    result = AgentGuard().validate(state=state, observation=_obs(state), decision=_decision("hybrid_search", {"query": state.current_query}))
    text = json.dumps(result.trace.model_dump(mode="json"), ensure_ascii=False).lower()
    return {
        "benchmark_gold_exposure_count": sum(1 for term in FORBIDDEN_TRACE_TERMS if term in text),
        "secret_exposure_count": 1 if "postgresql://user:secret@host/db" in text else 0,
        "hidden_reasoning_exposure_count": 1 if "chain_of_thought" in text or "hidden_reasoning" in text else 0,
        "guard_trace_query_redacted": "postgresql://user:secret@host/db" not in text,
    }


def _framework_flags() -> dict[str, bool]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    source = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in (ROOT / "opk_rag/agentic_v2").glob("*.py"))
    return {
        "pydantic_ai_agent_runtime_used": "pydantic_ai" in source or "pydantic-ai" in pyproject,
        "langgraph_runtime_used": "langgraph" in source or "langgraph" in pyproject,
        "crewai_runtime_used": "crewai" in source or "crewai" in pyproject,
        "autogen_runtime_used": "autogen" in source or "autogen" in pyproject,
    }


def build_summary(*, write: bool = True) -> dict[str, Any]:
    task0240 = _read_json(TASK0240)
    task0241 = _read_json(TASK0241)
    current_digest = contract_digest()
    guard_matrix = _validate_guard_matrix()
    budget = _budget_matrix()
    decision_action = _decision_action_matrix()
    deterministic = _deterministic_replay()
    separation = _pydantic_guard_separation()
    integration = _policy_guard_integration()
    security = _security()
    changed = _git_changed_paths()
    production_changed = [path for path in changed if path.startswith(PRODUCTION_PREFIXES)]
    flags = _framework_flags()
    regression = _read_json(REGRESSION) if REGRESSION.is_file() else {}
    guard_source = (ROOT / "opk_rag/agentic_v2/guard.py").read_text(encoding="utf-8").lower()
    trace_source = (ROOT / "opk_rag/agentic_v2/guard_trace.py").read_text(encoding="utf-8").lower()

    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete",
        "current_stage": "llm_agentic_rag_development",
        "task0240_contract_digest": task0240.get("agentic_v2_contract_digest"),
        "current_task0240_contract_digest": current_digest,
        "task0240_contract_digest_match": task0240.get("agentic_v2_contract_digest") == current_digest,
        "task0241_policy_runtime_valid": task0241.get("task_status") == "complete" and task0241.get("structured_decision_runtime_valid") is True,
        "deterministic_agent_guard_implemented": (ROOT / "opk_rag/agentic_v2/guard.py").is_file(),
        "allowed_action_resolver_implemented": (ROOT / "opk_rag/agentic_v2/allowed_actions.py").is_file(),
        "decision_to_action_validation_valid": all(guard_matrix[key] for key in ("allow_supported", "modify_supported", "reject_supported", "terminate_supported")),
        "guard_allow_supported": guard_matrix["allow_supported"],
        "guard_modify_supported": guard_matrix["modify_supported"],
        "guard_reject_supported": guard_matrix["reject_supported"],
        "guard_terminate_supported": guard_matrix["terminate_supported"],
        **budget,
        **decision_action,
        **deterministic,
        **separation,
        "fail_closed_behavior_valid": _fail_closed(),
        **integration,
        **security,
        "guard_trace_valid": "state_digest" in trace_source and "decision_digest" in trace_source and security["guard_trace_query_redacted"],
        "guard_metrics_valid": all(key in guard_matrix["metrics"] for key in ("allow_count", "modify_count", "reject_count", "terminate_count", "guard_p50_latency_ms", "guard_p95_latency_ms")),
        "guard_metrics": guard_matrix["metrics"],
        "agent_guard_is_side_effect_free": all(token not in guard_source for token in ("executor.execute", "search_knowledge_base", "qdrantclient", "psycopg.connect", "requests.")),
        **flags,
        "agent_framework_runtime_used": any(flags.values()),
        "tool_execution_active": False,
        "state_transition_runtime_active": False,
        "agent_loop_active": False,
        "production_agentic_v2_active": False,
        "changed_paths": changed,
        "production_runtime_changed_paths": production_changed,
        "production_runtime_behavior_changed": bool(production_changed),
        "blocking_failure_count": 0,
        "next_task": "TASK-0243_agentic_v2_tool_registry_and_guarded_tool_executor",
        "git_commit_created": False,
        "focused_tests_passed": regression.get("focused_tests_passed"),
        "related_agent_regression_passed": regression.get("related_agent_regression_passed"),
        "task0242_verifier_passed": regression.get("task0242_verifier_passed"),
        "git_diff_check_passed": regression.get("git_diff_check_passed"),
    }
    gates = [
        summary["task0240_contract_digest_match"], summary["task0241_policy_runtime_valid"],
        summary["deterministic_agent_guard_implemented"], summary["allowed_action_resolver_implemented"],
        summary["decision_to_action_validation_valid"], summary["step_budget_enforced"],
        summary["retrieval_budget_enforced"], summary["rewrite_budget_enforced"],
        summary["graph_call_budget_enforced"], summary["graph_hop_budget_enforced"],
        summary["duplicate_graph_recovery_rejected"], summary["duplicate_retrieval_without_query_change_rejected"],
        summary["unauthorized_query_change_rejected"], summary["finish_answerability_gate_valid"],
        summary["abstain_policy_valid"], summary["pydantic_guard_separation_valid"],
        summary["deterministic_replay_valid"], summary["fail_closed_behavior_valid"],
        summary["policy_guard_integration_valid"], summary["allowed_action_agreement_valid"],
        not summary["tool_result_present"], summary["benchmark_gold_exposure_count"] == 0,
        summary["secret_exposure_count"] == 0, summary["hidden_reasoning_exposure_count"] == 0,
        summary["guard_trace_valid"], summary["guard_metrics_valid"], summary["agent_guard_is_side_effect_free"],
        not summary["agent_framework_runtime_used"], not summary["production_runtime_behavior_changed"],
    ]
    summary["blocking_failure_count"] = sum(1 for value in gates if not value)
    summary["task_status"] = "complete" if summary["blocking_failure_count"] == 0 else "partial"

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        CONTRACT.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT.write_text(json.dumps({
            "schema_version": "opk-rag.task0242.contract.v1",
            "task_id": TASK_ID,
            "task0240_contract_digest": task0240.get("agentic_v2_contract_digest"),
            "task0241_required": True,
            "guard_role": "deterministic_runtime_action_authorization",
            "outcomes": ["allow", "modify", "reject", "terminate"],
            "tool_execution_allowed": False,
            "state_mutation_allowed": False,
            "production_integration_allowed": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "guard_policy.json").write_text(json.dumps({
            "authority": "deterministic_guard",
            "llm_authority": "proposal_only",
            "pydantic_authority": "structural_validation",
            "guard_authority": "runtime_action_authorization",
            "hybrid_top_k_runtime_max": 10,
            "query_mutation_only_via": "rewrite_query",
            "finish_answerability": ["answerable", "partially_answerable"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resolver = AllowedActionResolver()
        matrix = {
            "fresh": resolver.resolve(state=_state(), observation=_obs(_state())),
            "evidence_answerable": resolver.resolve(state=_state(evidence_count=1, source_count=1, answerability_status="answerable"), observation=_obs(_state(evidence_count=1, source_count=1, answerability_status="answerable"))),
            "retrieval_exhausted": resolver.resolve(state=_state(retrieval_call_count=2), observation=_obs(_state(retrieval_call_count=2))),
            "terminal": resolver.resolve(state=_state(status="finished", termination_reason="done"), observation=_obs(_state(status="finished", termination_reason="done"))),
        }
        (RESULT_DIR / "allowed_action_matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "budget_validation.json").write_text(json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "decision_action_validation.json").write_text(json.dumps({**guard_matrix, **decision_action, **separation, **integration}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "deterministic_replay.json").write_text(json.dumps(deterministic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "security.json").write_text(json.dumps(security | {"fail_closed_behavior_valid": summary["fail_closed_behavior_valid"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "metrics.json").write_text(json.dumps(guard_matrix["metrics"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def verify() -> dict[str, Any]:
    summary = build_summary(write=False)
    required = {
        "task_status": "complete",
        "current_stage": "llm_agentic_rag_development",
        "task0240_contract_digest_match": True,
        "task0241_policy_runtime_valid": True,
        "deterministic_agent_guard_implemented": True,
        "allowed_action_resolver_implemented": True,
        "decision_to_action_validation_valid": True,
        "guard_allow_supported": True,
        "guard_modify_supported": True,
        "guard_reject_supported": True,
        "guard_terminate_supported": True,
        "step_budget_enforced": True,
        "retrieval_budget_enforced": True,
        "rewrite_budget_enforced": True,
        "graph_call_budget_enforced": True,
        "graph_hop_budget_enforced": True,
        "duplicate_graph_recovery_rejected": True,
        "duplicate_retrieval_without_query_change_rejected": True,
        "unauthorized_query_change_rejected": True,
        "finish_answerability_gate_valid": True,
        "abstain_policy_valid": True,
        "pydantic_guard_separation_valid": True,
        "deterministic_replay_valid": True,
        "fail_closed_behavior_valid": True,
        "benchmark_gold_exposure_count": 0,
        "secret_exposure_count": 0,
        "hidden_reasoning_exposure_count": 0,
        "guard_trace_valid": True,
        "guard_metrics_valid": True,
        "tool_execution_active": False,
        "state_transition_runtime_active": False,
        "agent_loop_active": False,
        "production_agentic_v2_active": False,
        "agent_framework_runtime_used": False,
        "production_runtime_behavior_changed": False,
        "blocking_failure_count": 0,
    }
    mismatches = {key: {"expected": value, "actual": summary.get(key)} for key, value in required.items() if summary.get(key) != value}
    files = [
        ROOT / "tasks/TASK-0242_deterministic_agent_guard_and_decision_to_action_validation.md",
        ROOT / "docs/LLM_AGENTIC_RAG_GUARD_RUNTIME.md",
        ROOT / "docs/TASK0242_DETERMINISTIC_AGENT_GUARD_AND_DECISION_TO_ACTION_VALIDATION_REPORT.md",
        CONTRACT,
        RESULT_DIR / "summary.json",
        RESULT_DIR / "guard_policy.json",
        RESULT_DIR / "allowed_action_matrix.json",
        RESULT_DIR / "budget_validation.json",
        RESULT_DIR / "decision_action_validation.json",
        RESULT_DIR / "deterministic_replay.json",
        RESULT_DIR / "security.json",
        RESULT_DIR / "metrics.json",
        RESULT_DIR / "regression.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    return {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": not mismatches and not missing, "mismatches": mismatches, "missing_files": missing}


if __name__ == "__main__":
    print(json.dumps(build_summary(), ensure_ascii=False, indent=2))
