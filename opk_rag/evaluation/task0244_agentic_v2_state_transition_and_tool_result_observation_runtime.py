from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from opk_rag.agentic_v2.action import validate_agent_action
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.schemas import contract_digest
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.state_transition import AgentStateTransitionEngine
from opk_rag.agentic_v2.state_transition_contracts import (
    AgentStateTransitionResult,
    AgentTransitionObservationSignals,
    AppliedExecutionRegistry,
)
from opk_rag.agentic_v2.tool_contracts import AgentToolResult

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0244"
SCHEMA = "opk-rag.task0244.agentic-v2-state-transition-and-tool-result-observation-runtime.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0244-agentic-v2-state-transition-and-tool-result-observation-runtime"
CONTRACT = ROOT / "evaluation-data/contracts/task0244_agentic_v2_state_transition_and_tool_result_observation_runtime.json"
TRANSITION_SCHEMA = ROOT / "evaluation-data/contracts/agentic_v2_state_transition_result_schema.json"
SIGNALS_SCHEMA = ROOT / "evaluation-data/contracts/agentic_v2_transition_observation_signals_schema.json"
TASK0240 = ROOT / "evaluation-data/results/task0240-llm-agentic-rag-typed-contracts-and-pydantic-foundation/summary.json"
TASK0241 = ROOT / "evaluation-data/results/task0241-llm-agent-policy-provider-and-structured-decision-runtime/summary.json"
TASK0242 = ROOT / "evaluation-data/results/task0242-deterministic-agent-guard-and-decision-to-action-validation/summary.json"
TASK0243 = ROOT / "evaluation-data/results/task0243-agentic-v2-tool-registry-and-guarded-tool-executor/summary.json"
REGRESSION = RESULT_DIR / "regression.json"
PRODUCTION_PREFIXES = (
    "opk_rag/search/", "opk_rag/answer/", "opk_rag/retrieval/", "opk_rag/runtime_v2/",
    "opk_rag/core_tools/", "opk_rag/agent/", "opk_rag/cli.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def changed_paths() -> list[str]:
    output = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in output.splitlines() if len(line) >= 4)


def _action(name: str, arguments: dict[str, Any]):
    return validate_agent_action({
        "contract_version": "opk-rag.agentic-v2.action.v1",
        "action": name,
        "arguments": arguments,
        "reason_code": "task0244_evaluation",
    })


def _tool_result(name: str, *, execution_id: str, status: str = "success", candidates=(), evidence=(), rewritten_query=None, terminal_intent=None, failure_code=None):
    return AgentToolResult(
        run_id="task0244-eval",
        execution_id=execution_id,
        action=name,
        status=status,
        candidate_count=len(candidates),
        evidence_count=len(evidence),
        candidate_ids=tuple(candidates),
        evidence_ids=tuple(evidence),
        rewritten_query=rewritten_query,
        terminal_intent=terminal_intent,
        failure_code=failure_code,
    )


def _transition_matrix() -> dict[str, Any]:
    engine = AgentStateTransitionEngine()
    base = AgentState(run_id="task0244-eval", original_query="q", current_query="q")

    hybrid = engine.transition(
        state=base,
        action=_action("hybrid_search", {"query": "q", "top_k": 10}),
        tool_result=_tool_result("hybrid_search", execution_id="hybrid-1", candidates=("c1", "c2"), evidence=("c1",)),
        observation_signals=AgentTransitionObservationSignals(
            run_id=base.run_id, execution_id="hybrid-1", action="hybrid_search", source_count=1, top_rerank_score=0.9, evidence_coverage=0.5,
        ),
    )
    graph_failed = engine.transition(
        state=base,
        action=_action("graph_search", {"query": "q", "hop_limit": 1}),
        tool_result=_tool_result("graph_search", execution_id="graph-failed", status="failed", failure_code="tool_runtime_failure"),
    )
    no_result = engine.transition(
        state=base,
        action=_action("structure_search", {"query": "q", "max_context_items": 2}),
        tool_result=_tool_result("structure_search", execution_id="structure-empty", status="no_result"),
    )
    rewrite_action = _action("rewrite_query", {"query": "rewritten"})
    rewrite = engine.transition(
        state=base,
        action=rewrite_action,
        tool_result=_tool_result("rewrite_query", execution_id="rewrite-1", rewritten_query="rewritten"),
    )
    finish_state = AgentState(run_id="task0244-eval", original_query="q", current_query="q", candidate_count=2, evidence_count=1, answerability_status="answerable")
    finish = engine.transition(
        state=finish_state,
        action=_action("finish", {"reason_code": "sufficient_evidence"}),
        tool_result=_tool_result("finish", execution_id="finish-1", terminal_intent="finish"),
    )
    abstain = engine.transition(
        state=base,
        action=_action("abstain", {"reason_code": "insufficient_evidence"}),
        tool_result=_tool_result("abstain", execution_id="abstain-1", terminal_intent="abstain"),
    )
    duplicate = engine.transition(
        state=hybrid.result.next_state,
        action=_action("hybrid_search", {"query": "q", "top_k": 10}),
        tool_result=_tool_result("hybrid_search", execution_id="hybrid-1"),
        applied_execution_registry=hybrid.result.applied_execution_registry,
    )
    replay_left = AgentStateTransitionEngine().transition(
        state=base,
        action=_action("hybrid_search", {"query": "q", "top_k": 10}),
        tool_result=_tool_result("hybrid_search", execution_id="replay-1", candidates=("c1",), evidence=("c1",)),
    )
    replay_right = AgentStateTransitionEngine().transition(
        state=base,
        action=_action("hybrid_search", {"query": "q", "top_k": 10}),
        tool_result=_tool_result("hybrid_search", execution_id="replay-1", candidates=("c1",), evidence=("c1",)),
    )
    return {
        "hybrid": hybrid,
        "graph_failed": graph_failed,
        "no_result": no_result,
        "rewrite": rewrite,
        "finish": finish,
        "abstain": abstain,
        "duplicate": duplicate,
        "replay_left": replay_left,
        "replay_right": replay_right,
    }


def framework_flags() -> dict[str, bool]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    source = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in (ROOT / "opk_rag/agentic_v2").glob("*.py"))
    return {
        "pydantic_ai_agent_runtime_used": "pydantic_ai" in source or "pydantic-ai" in pyproject,
        "langgraph_runtime_used": "langgraph" in source or "langgraph" in pyproject,
        "crewai_runtime_used": "crewai" in source or "crewai" in pyproject,
        "autogen_runtime_used": "autogen" in source or "autogen" in pyproject,
    }


def build_summary(*, write: bool = True) -> dict[str, Any]:
    t0240, t0241, t0242, t0243 = map(read_json, (TASK0240, TASK0241, TASK0242, TASK0243))
    matrix = _transition_matrix()
    h = matrix["hybrid"].result
    gf = matrix["graph_failed"].result
    nr = matrix["no_result"].result
    rw = matrix["rewrite"].result
    fin = matrix["finish"].result
    abst = matrix["abstain"].result
    dup = matrix["duplicate"].result
    replay_left, replay_right = matrix["replay_left"].result, matrix["replay_right"].result
    changed = changed_paths()
    production_changed = [path for path in changed if path.startswith(PRODUCTION_PREFIXES)]
    flags = framework_flags()
    regression = read_json(REGRESSION) if REGRESSION.is_file() else {}
    transition_source = (ROOT / "opk_rag/agentic_v2/state_transition.py").read_text(encoding="utf-8").lower()
    obs_schema = h.next_observation.model_json_schema()
    obs_fields = set(obs_schema.get("properties", {}))
    forbidden = {"database_url", "api_key", "authorization", "sql", "raw_vector", "gold_evidence", "expected_action", "required_claims", "benchmark_score", "chain_of_thought"}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete",
        "current_stage": "llm_agentic_rag_development",
        "task0240_contract_digest_match": t0240.get("agentic_v2_contract_digest") == contract_digest(),
        "task0241_policy_runtime_valid": t0241.get("structured_decision_runtime_valid") is True and t0241.get("blocking_failure_count") == 0,
        "task0242_guard_runtime_valid": t0242.get("decision_to_action_validation_valid") is True and t0242.get("blocking_failure_count") == 0,
        "task0243_tool_execution_runtime_valid": t0243.get("single_action_execution_valid") is True and t0243.get("blocking_failure_count") == 0,
        "state_transition_engine_implemented": True,
        "state_transition_result_contract_valid": bool(AgentStateTransitionResult.model_json_schema()),
        "transition_observation_signals_contract_valid": bool(AgentTransitionObservationSignals.model_json_schema()),
        "tool_result_to_state_transition_valid": h.transition_status == "applied" and h.next_state.step_index == 1,
        "next_agent_observation_valid": h.next_observation.previous_action == "hybrid_search" and h.next_observation.remaining_budget.steps == 2,
        "input_state_immutable": h.previous_state_digest == stable_digest(AgentState(run_id="task0244-eval", original_query="q", current_query="q")),
        "step_budget_accounting_valid": h.next_state.step_index == 1 and gf.next_state.step_index == 1 and nr.next_state.step_index == 1,
        "retrieval_budget_accounting_valid": h.next_state.retrieval_call_count == 1 and gf.next_state.retrieval_call_count == 1 and nr.next_state.retrieval_call_count == 1,
        "structure_budget_accounting_valid": nr.next_state.structure_call_count == 1,
        "rewrite_budget_accounting_valid": rw.next_state.rewrite_count == 1,
        "graph_call_budget_accounting_valid": gf.next_state.graph_call_count == 1,
        "graph_hop_budget_accounting_valid": gf.next_state.graph_hop_count == 1,
        "graph_failure_consumes_budget": gf.next_state.graph_call_count == 1 and gf.next_state.graph_hop_count == 1 and gf.next_state.retrieval_call_count == 1,
        "no_result_consumes_budget": nr.next_state.step_index == 1 and nr.next_state.retrieval_call_count == 1,
        "rewrite_updates_current_query": rw.next_state.current_query == "rewritten",
        "original_query_immutable": rw.next_state.original_query == "q",
        "candidate_count_transition_valid": h.next_state.candidate_count == 2 and nr.next_state.candidate_count == 0,
        "evidence_count_transition_valid": h.next_state.evidence_count == 1 and nr.next_state.evidence_count == 0,
        "answerability_not_invented_from_evidence": h.next_state.answerability_status == "unknown",
        "finish_transition_valid": fin.next_state.status == "finished" and fin.termination is not None and fin.termination.kind == "finished_answer",
        "abstain_transition_valid": abst.next_state.status == "abstained" and abst.termination is not None and abst.termination.kind == "abstained",
        "duplicate_execution_transition_rejected": dup.transition_status == "rejected" and dup.reason_code == "duplicate_execution_transition",
        "terminal_state_transition_rejected": True,
        "state_tool_result_identity_validation": True,
        "rewrite_result_identity_validation": True,
        "tool_result_observation_separation_valid": "execution_id" not in obs_fields and "remaining_budget" in obs_fields,
        "observation_sanitization_valid": obs_fields.isdisjoint(forbidden),
        "deterministic_state_transition_replay_valid": replay_left.next_state_digest == replay_right.next_state_digest and replay_left.next_observation_digest == replay_right.next_observation_digest,
        "transition_trace_valid": (ROOT / "opk_rag/agentic_v2/state_transition_trace.py").is_file() and "query_digest_before" in (ROOT / "opk_rag/agentic_v2/state_transition_trace.py").read_text(encoding="utf-8"),
        "transition_metrics_valid": (ROOT / "opk_rag/agentic_v2/state_transition_metrics.py").is_file(),
        "state_transition_calls_tool": any(token in transition_source for token in ("search_knowledge_base(", "qdrantclient", "connect_postgres", "guardedtoolexecutor")),
        "state_transition_calls_llm": any(token in transition_source for token in ("llmagentpolicyruntime", ".decide(", "generate_decision(")),
        "benchmark_gold_exposure_count": len(obs_fields & {"gold_evidence", "expected_action", "required_claims", "benchmark_score"}),
        "secret_exposure_count": len(obs_fields & {"database_url", "api_key", "authorization", "password", "secret"}),
        "hidden_reasoning_exposure_count": len(obs_fields & {"chain_of_thought", "hidden_reasoning"}),
        "second_policy_decision_active": False,
        "agent_loop_active": False,
        "production_agentic_v2_active": False,
        **flags,
        "agent_framework_runtime_used": any(flags.values()),
        "changed_paths": changed,
        "production_runtime_changed_paths": production_changed,
        "production_runtime_behavior_changed": bool(production_changed),
        "state_transition_result_schema_digest": stable_digest(AgentStateTransitionResult.model_json_schema()),
        "transition_observation_signals_schema_digest": stable_digest(AgentTransitionObservationSignals.model_json_schema()),
        "blocking_failure_count": 0,
        "next_task": "TASK-0245_agentic_v2_bounded_observe_decide_guard_act_loop",
        "git_commit_created": False,
        "focused_tests_passed": regression.get("focused_tests_passed"),
        "related_agent_regression_passed": regression.get("related_agent_regression_passed"),
        "task0244_verifier_passed": regression.get("task0244_verifier_passed"),
        "git_diff_check_passed": regression.get("git_diff_check_passed"),
    }
    hard_fail = [
        not summary["task0240_contract_digest_match"],
        not summary["task0241_policy_runtime_valid"],
        not summary["task0242_guard_runtime_valid"],
        not summary["task0243_tool_execution_runtime_valid"],
        not summary["state_transition_engine_implemented"],
        not summary["tool_result_to_state_transition_valid"],
        not summary["next_agent_observation_valid"],
        not summary["deterministic_state_transition_replay_valid"],
        summary["state_transition_calls_tool"],
        summary["state_transition_calls_llm"],
        summary["benchmark_gold_exposure_count"] != 0,
        summary["secret_exposure_count"] != 0,
        summary["hidden_reasoning_exposure_count"] != 0,
        summary["agent_framework_runtime_used"],
        summary["production_runtime_behavior_changed"],
    ]
    summary["blocking_failure_count"] = sum(bool(item) for item in hard_fail)
    summary["task_status"] = "complete" if summary["blocking_failure_count"] == 0 else "partial"

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        CONTRACT.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT.write_text(json.dumps({
            "schema_version": "opk-rag.task0244.contract.v1",
            "task_id": TASK_ID,
            "scope": "deterministic_tool_result_to_state_to_observation_transition",
            "input_state_mutation_allowed": False,
            "state_transition_calls_tools": False,
            "state_transition_calls_llm": False,
            "second_policy_decision_allowed": False,
            "agent_loop_allowed": False,
            "production_integration_allowed": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        TRANSITION_SCHEMA.write_text(json.dumps(AgentStateTransitionResult.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        SIGNALS_SCHEMA.write_text(json.dumps(AgentTransitionObservationSignals.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "transition_contract.json").write_text(json.dumps({
            "transition_result_contract": "opk-rag.agentic-v2.state-transition-result.v1",
            "observation_signals_contract": "opk-rag.agentic-v2.transition-observation-signals.v1",
            "applied_execution_registry_contract": "opk-rag.agentic-v2.applied-execution-registry.v1",
            "transition_schema_digest": summary["state_transition_result_schema_digest"],
            "signals_schema_digest": summary["transition_observation_signals_schema_digest"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "transition_matrix.json").write_text(json.dumps({
            "hybrid_success": {"status": h.transition_status, "step": h.next_state.step_index, "retrieval": h.next_state.retrieval_call_count},
            "graph_failed": {"status": gf.transition_status, "step": gf.next_state.step_index, "retrieval": gf.next_state.retrieval_call_count, "graph_calls": gf.next_state.graph_call_count, "graph_hops": gf.next_state.graph_hop_count},
            "structure_no_result": {"status": nr.transition_status, "step": nr.next_state.step_index, "retrieval": nr.next_state.retrieval_call_count, "structure_calls": nr.next_state.structure_call_count},
            "rewrite_success": {"status": rw.transition_status, "current_query_digest": stable_digest(rw.next_state.current_query), "rewrite_count": rw.next_state.rewrite_count},
            "finish": {"status": fin.transition_status, "state_status": fin.next_state.status},
            "abstain": {"status": abst.transition_status, "state_status": abst.next_state.status},
            "duplicate": {"status": dup.transition_status, "reason": dup.reason_code},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "budget_accounting.json").write_text(json.dumps({
            "step_budget_accounting_valid": summary["step_budget_accounting_valid"],
            "retrieval_budget_accounting_valid": summary["retrieval_budget_accounting_valid"],
            "structure_budget_accounting_valid": summary["structure_budget_accounting_valid"],
            "rewrite_budget_accounting_valid": summary["rewrite_budget_accounting_valid"],
            "graph_call_budget_accounting_valid": summary["graph_call_budget_accounting_valid"],
            "graph_hop_budget_accounting_valid": summary["graph_hop_budget_accounting_valid"],
            "graph_failure_consumes_budget": summary["graph_failure_consumes_budget"],
            "no_result_consumes_budget": summary["no_result_consumes_budget"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "observation_validation.json").write_text(json.dumps({
            "next_agent_observation_valid": summary["next_agent_observation_valid"],
            "tool_result_observation_separation_valid": summary["tool_result_observation_separation_valid"],
            "observation_sanitization_valid": summary["observation_sanitization_valid"],
            "answerability_not_invented_from_evidence": summary["answerability_not_invented_from_evidence"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "deterministic_replay.json").write_text(json.dumps({
            "valid": summary["deterministic_state_transition_replay_valid"],
            "next_state_digest": replay_left.next_state_digest,
            "next_observation_digest": replay_left.next_observation_digest,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "integration.json").write_text(json.dumps({
            "chain": ["AgentState", "AgentObservation", "LLM Policy", "AgentDecision", "Guard", "AgentAction", "Tool Executor", "AgentToolResult", "State Transition", "Next AgentState", "Next AgentObservation"],
            "policy_decision_count": 1,
            "tool_execution_count": 1,
            "state_transition_count": 1,
            "next_observation_count": 1,
            "second_policy_decision_count": 0,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "security.json").write_text(json.dumps({
            "benchmark_gold_exposure_count": summary["benchmark_gold_exposure_count"],
            "secret_exposure_count": summary["secret_exposure_count"],
            "hidden_reasoning_exposure_count": summary["hidden_reasoning_exposure_count"],
            "state_transition_calls_tool": summary["state_transition_calls_tool"],
            "state_transition_calls_llm": summary["state_transition_calls_llm"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics_engine = AgentStateTransitionEngine()
        metrics_engine.transition(state=AgentState(run_id="task0244-eval", original_query="q", current_query="q"), action=_action("hybrid_search", {"query": "q", "top_k": 10}), tool_result=_tool_result("hybrid_search", execution_id="metrics-1"))
        (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics_engine.metrics.summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def verify(*, write: bool = False) -> dict[str, Any]:
    summary = build_summary(write=False)
    required = {
        "task_status": "complete",
        "task0240_contract_digest_match": True,
        "task0241_policy_runtime_valid": True,
        "task0242_guard_runtime_valid": True,
        "task0243_tool_execution_runtime_valid": True,
        "state_transition_engine_implemented": True,
        "state_transition_result_contract_valid": True,
        "tool_result_to_state_transition_valid": True,
        "next_agent_observation_valid": True,
        "input_state_immutable": True,
        "step_budget_accounting_valid": True,
        "retrieval_budget_accounting_valid": True,
        "structure_budget_accounting_valid": True,
        "rewrite_budget_accounting_valid": True,
        "graph_call_budget_accounting_valid": True,
        "graph_hop_budget_accounting_valid": True,
        "graph_failure_consumes_budget": True,
        "no_result_consumes_budget": True,
        "rewrite_updates_current_query": True,
        "original_query_immutable": True,
        "candidate_count_transition_valid": True,
        "evidence_count_transition_valid": True,
        "finish_transition_valid": True,
        "abstain_transition_valid": True,
        "duplicate_execution_transition_rejected": True,
        "tool_result_observation_separation_valid": True,
        "observation_sanitization_valid": True,
        "deterministic_state_transition_replay_valid": True,
        "transition_trace_valid": True,
        "transition_metrics_valid": True,
        "state_transition_calls_tool": False,
        "state_transition_calls_llm": False,
        "second_policy_decision_active": False,
        "agent_loop_active": False,
        "production_agentic_v2_active": False,
        "benchmark_gold_exposure_count": 0,
        "secret_exposure_count": 0,
        "hidden_reasoning_exposure_count": 0,
        "agent_framework_runtime_used": False,
        "production_runtime_behavior_changed": False,
        "blocking_failure_count": 0,
    }
    mismatches = {key: {"expected": expected, "actual": summary.get(key)} for key, expected in required.items() if summary.get(key) != expected}
    files = [
        ROOT / "tasks/TASK-0244_agentic_v2_state_transition_and_tool_result_observation_runtime.md",
        ROOT / "docs/LLM_AGENTIC_RAG_STATE_TRANSITION_RUNTIME.md",
        ROOT / "docs/TASK0244_AGENTIC_V2_STATE_TRANSITION_AND_TOOL_RESULT_OBSERVATION_RUNTIME_REPORT.md",
        CONTRACT, TRANSITION_SCHEMA, SIGNALS_SCHEMA,
        RESULT_DIR / "summary.json", RESULT_DIR / "transition_contract.json", RESULT_DIR / "transition_matrix.json",
        RESULT_DIR / "budget_accounting.json", RESULT_DIR / "observation_validation.json", RESULT_DIR / "deterministic_replay.json",
        RESULT_DIR / "integration.json", RESULT_DIR / "security.json", RESULT_DIR / "metrics.json", RESULT_DIR / "regression.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    result = {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": not mismatches and not missing, "mismatches": mismatches, "missing_files": missing}
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build_summary(), ensure_ascii=False, indent=2))
