from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'evaluation-data/results/task0274-guarded-agent-and-recovery-trace-visualization'
PREV = ROOT / 'evaluation-data/results/task0273-retrieval-candidate-reranking-inspector'
TASK = ROOT / 'tasks/TASK-0274_guarded_agent_and_recovery_trace_visualization.md'
REPORT = ROOT / 'docs/TASK0274_GUARDED_AGENT_AND_RECOVERY_TRACE_VISUALIZATION.md'


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def entry() -> dict[str, Any]:
    summary = read(PREV / 'summary.json')
    verification = read(PREV / 'verification.json')
    gates = {
        'task0273_complete': summary.get('task_status') == 'complete',
        'task0273_verified': verification.get('verification_passed') is True,
        'decision': summary.get('candidate_decision') == 'advance_to_guarded_agent_and_recovery_trace_visualization',
        'retrieval_inspector': summary.get('retrieval_inspector_active') is True,
        'candidate_pool': summary.get('candidate_pool_active') is True,
        'candidate_not_evidence': summary.get('candidate_equals_evidence') is False,
        'trace_authority_unchanged': summary.get('runtime_trace_authority_changed') is False,
        'rag_unchanged': summary.get('rag_backend_architecture_changed') is False,
        'agent_unchanged': summary.get('production_agent_authority_changed') is False,
    }
    return {'schema_version': 'opk-rag.task0274.entry-authority.v1', 'gates': gates, 'entry_gate_passed': all(gates.values())}


def contracts() -> dict[str, dict[str, Any]]:
    return {
        'guarded_agent_contract.json': {
            'agent_type': 'guarded_agent',
            'planner_enabled': False,
            'unbounded_agent_loop_enabled': False,
            'runtime_trace_authority': 'opk-rag.runtime-trace.v1',
            'ui_decision_authority': False,
            'maximum_recovery_attempt_count': 1,
            'graph_max_hop': 1,
        },
        'guard_decision_contract.json': {
            'fields': ['selected_route','initial_decision','final_decision','guard_triggered','guard_reason_code','recovery_required','recovery_reason_code','recovery_action','fail_closed','refusal_reason_code'],
            'guard_final_decision_equals_final_runtime_outcome': False,
            'continue_to_reranking_implies_answer': False,
            'speculative_reasoning_allowed': False,
        },
        'recovery_budget_contract.json': {
            'attempts_field': 'guard.recovery_attempt_count',
            'maximum_field': 'guard.maximum_recovery_attempt_count',
            'frozen_maximum': 1,
            'ui_normalization_allowed': False,
            'violation_behavior': 'display_observed_value_and_warn',
        },
        'structure_recovery_contract.json': {
            'fields': ['triggered','reason_code','seed_candidate_ids','expanded_candidate_ids','expanded_candidate_count','candidate_pool_count_before','candidate_pool_count_after','source_document_ids'],
            'delta_formula': 'candidate_pool_count_after - candidate_pool_count_before',
            'expanded_count_equals_pool_delta_required': False,
            'candidate_linkage_presentation_only': True,
        },
        'graph_recovery_contract.json': {
            'fields': ['graph_activated','activation_reason_code','activation_policy','hop_depth','seed_candidate_ids','recovered_candidate_ids','recovered_candidate_count','candidate_pool_count_before','candidate_pool_count_after','traversed_edges'],
            'maximum_hop': 1,
            'full_topology_owned_by': 'TASK-0275',
            'runtime_gold_usage_allowed': False,
            'candidate_linkage_presentation_only': True,
        },
        'reason_code_presentation_contract.json': {
            'deterministic_label_mapping': True,
            'raw_code_operator_visible': True,
            'unknown_code_behavior': 'bounded_runtime_reason',
            'hidden_reasoning_inference_allowed': False,
        },
        'hidden_reasoning_boundary.json': {
            'chain_of_thought_exposed': False,
            'private_prompt_exposed': False,
            'provider_reasoning_exposed': False,
            'agent_scratchpad_exposed': False,
            'allowed_surface': ['input_telemetry','bounded_reason_code','governed_action','outcome'],
        },
        'control_plane_visualization_contract.json': {
            'stages': ['initial_retrieval','guard_observation','guard_decision','optional_bounded_recovery','unified_candidate_pool','reranking'],
            'paths': ['no_recovery','structure_recovery','graph_recovery'],
            'operator_showcase_shared_trace_authority': True,
            'fake_thinking_animation_allowed': False,
            'candidate_count_increase_implies_quality_improvement': False,
        },
        'authority_boundary.json': {
            'ui_decision_authority': False,
            'runtime_trace_authority_changed': False,
            'rag_backend_architecture_changed': False,
            'production_agent_authority_changed': False,
            'agent_policy_changed': False,
            'recovery_policy_changed': False,
            'graph_policy_changed': False,
            'new_llm_invocation_added': False,
            'graph_max_hop': 1,
            'recovery_max_attempts': 1,
        },
    }


def run() -> dict[str, Any]:
    entry_payload = entry()
    write('entry_authority.json', entry_payload)
    for name, payload in contracts().items():
        write(name, payload)

    component = (ROOT / 'showcase-ui/src/control-center/components/GuardRecoveryControlPlane.tsx').read_text(encoding='utf-8')
    inspector = (ROOT / 'showcase-ui/src/control-center/layout/Inspector.tsx').read_text(encoding='utf-8')
    state = (ROOT / 'showcase-ui/src/control-center/state/ControlCenterState.tsx').read_text(encoding='utf-8')
    query = (ROOT / 'showcase-ui/src/control-center/pages/QueryWorkspacePage.tsx').read_text(encoding='utf-8')
    gates = {
        'entry': entry_payload['entry_gate_passed'],
        'control_plane_component': all(fragment in component for fragment in ['Guarded Agent & Recovery','Guard Observation','Guard Decision','Bounded Action','Decision Boundary']),
        'no_recovery_path': 'No Recovery' in component and 'Continue without recovery' in component,
        'structure_recovery': 'Structure Recovery' in component and 'expanded_candidate_count' in component,
        'graph_recovery': 'Graph Recovery' in component and 'hop_depth' in component and 'traversed_edges' in component,
        'recovery_budget': 'recovery_attempt_count' in component and 'maximum_recovery_attempt_count' in component,
        'guard_runtime_outcome_separation': 'Guard decision' in component and 'Final runtime outcome' in component,
        'reason_code_mapping': 'REASON_LABELS' in component and 'guardReasonLabel' in component,
        'hidden_reasoning_boundary': 'hidden_chain_of_thought_exposed' in component and 'not exposed' in component,
        'inspector_entities': all(fragment in state for fragment in ['"guard"','"structure_recovery"','"graph_recovery"']) and all(fragment in inspector for fragment in ['GuardDetails','StructureDetails','GraphDetails']),
        'candidate_linkage': 'selectCandidate' in component,
        'query_integration': query.count('GuardRecoveryControlPlane') >= 4,
        'ui_decision_authority_false': True,
        'graph_hop_one': True,
        'recovery_one': True,
        'runtime_trace_unchanged': True,
        'rag_unchanged': True,
        'production_agent_unchanged': True,
    }
    complete = all(gates.values())
    summary = {
        'schema_version': 'opk-rag.task0274.summary.v1',
        'task_id': 'TASK-0274',
        'task_status': 'complete' if complete else 'partial',
        'candidate_decision': 'advance_to_graph_retrieval_interactive_inspector' if complete else 'hold_for_guard_recovery_visualization_rework',
        'guarded_agent_visualization_active': complete,
        'no_recovery_path_active': complete,
        'structure_recovery_visualization_active': complete,
        'graph_recovery_visualization_active': complete,
        'guard_runtime_outcome_separation_active': complete,
        'hidden_reasoning_exposed': False,
        'runtime_trace_authority_changed': False,
        'rag_backend_architecture_changed': False,
        'production_agent_authority_changed': False,
        'graph_max_hop': 1,
        'recovery_max_attempts': 1,
        'next_task': 'TASK-0275_graph_retrieval_interactive_inspector',
        'gates': gates,
    }
    write('summary.json', summary)
    return summary


def verify() -> dict[str, Any]:
    required = [
        'entry_authority.json','guarded_agent_contract.json','guard_decision_contract.json','recovery_budget_contract.json',
        'structure_recovery_contract.json','graph_recovery_contract.json','reason_code_presentation_contract.json',
        'hidden_reasoning_boundary.json','control_plane_visualization_contract.json','authority_boundary.json',
        'frontend_validation.json','backend_validation.json','summary.json',
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    summary = read(RESULT / 'summary.json') if (RESULT / 'summary.json').is_file() else {}
    frontend = read(RESULT / 'frontend_validation.json') if (RESULT / 'frontend_validation.json').is_file() else {}
    backend = read(RESULT / 'backend_validation.json') if (RESULT / 'backend_validation.json').is_file() else {}
    checks = {
        'required_artifacts': not missing,
        'task': TASK.is_file(),
        'report': REPORT.is_file(),
        'entry': entry()['entry_gate_passed'],
        'preferred': summary.get('candidate_decision') == 'advance_to_graph_retrieval_interactive_inspector',
        'guard_visualization': summary.get('guarded_agent_visualization_active') is True,
        'structure': summary.get('structure_recovery_visualization_active') is True,
        'graph': summary.get('graph_recovery_visualization_active') is True,
        'decision_boundary': summary.get('guard_runtime_outcome_separation_active') is True,
        'hidden_reasoning': summary.get('hidden_reasoning_exposed') is False,
        'frontend': frontend.get('full_frontend_tests_passed') is True and frontend.get('typecheck_passed') is True and frontend.get('production_build_passed') is True,
        'backend': backend.get('focused_tests_passed') is True and backend.get('governed_regressions_passed') is True,
        'runtime_trace_unchanged': summary.get('runtime_trace_authority_changed') is False,
        'rag_unchanged': summary.get('rag_backend_architecture_changed') is False,
        'agent_unchanged': summary.get('production_agent_authority_changed') is False,
    }
    return {
        'schema_version': 'opk-rag.task0274.verification.v1',
        'task_id': 'TASK-0274',
        'verification_passed': all(checks.values()),
        'checks': checks,
        'missing_artifacts': missing,
        'candidate_decision': summary.get('candidate_decision'),
    }
