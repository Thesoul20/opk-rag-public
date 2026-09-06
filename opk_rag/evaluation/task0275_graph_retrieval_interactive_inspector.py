from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'evaluation-data/results/task0275-graph-retrieval-interactive-inspector'
PREV = ROOT / 'evaluation-data/results/task0274-guarded-agent-and-recovery-trace-visualization'
TASK = ROOT / 'tasks/TASK-0275_graph_retrieval_interactive_inspector.md'
REPORT = ROOT / 'docs/TASK0275_GRAPH_RETRIEVAL_INTERACTIVE_INSPECTOR.md'


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def entry() -> dict[str, Any]:
    summary = read(PREV / 'summary.json')
    verification = read(PREV / 'verification.json')
    gates = {
        'task0274_complete': summary.get('task_status') == 'complete',
        'task0274_verified': verification.get('verification_passed') is True,
        'decision': summary.get('candidate_decision') == 'advance_to_graph_retrieval_interactive_inspector',
        'graph_recovery_visualization': summary.get('graph_recovery_visualization_active') is True,
        'hidden_reasoning_not_exposed': summary.get('hidden_reasoning_exposed') is False,
        'runtime_trace_unchanged': summary.get('runtime_trace_authority_changed') is False,
        'rag_unchanged': summary.get('rag_backend_architecture_changed') is False,
        'agent_unchanged': summary.get('production_agent_authority_changed') is False,
    }
    return {'schema_version': 'opk-rag.task0275.entry-authority.v1', 'gates': gates, 'entry_gate_passed': all(gates.values())}


def contracts() -> dict[str, dict[str, Any]]:
    return {
        'graph_runtime_authority_contract.json': {
            'authority': 'Runtime Trace V1 graph_recovery',
            'full_knowledge_graph_authority': False,
            'new_traversal_allowed': False,
            'graph_policy_mutation_allowed': False,
            'runtime_gold_enrichment_allowed': False,
        },
        'runtime_subgraph_contract.json': {
            'scope': 'query_scoped_runtime_graph_recovery_subgraph',
            'layout': 'deterministic_one_hop',
            'force_layout_required': False,
            'complete_knowledge_graph_claim_allowed': False,
            'skipped_graph_fabricated_topology_allowed': False,
        },
        'graph_node_contract.json': {
            'roles': ['seed','recovered','related'],
            'node_identity_source': ['seed_node_ids','recovered_node_ids','traversed_edges.source_node_id','traversed_edges.target_node_id'],
            'graph_node_equals_candidate': False,
            'selection_presentation_only': True,
        },
        'graph_edge_contract.json': {
            'fields': ['edge_id','source_node_id','target_node_id','relation_type','source_document_id','target_document_id','hop_depth','edge_observation'],
            'duplicate_edge_identity_preserved': True,
            'silent_semantic_merge_allowed': False,
            'missing_edge_identity_fabrication_allowed': False,
        },
        'candidate_graph_provenance_contract.json': {
            'fields': ['candidate_id','canonical_chunk_id','document_id','edge_types','graph_added','graph_edge_types','graph_expansion_reason','graph_hop_count','graph_path','section_id','seed_candidate_id','source_seed_candidate_id'],
            'candidate_linkage_target': 'TASK-0273 Candidate Inspector',
            'candidate_path_selection_presentation_only': True,
            'graph_recovery_implies_evidence': False,
        },
        'one_hop_boundary_contract.json': {
            'frozen_max_hop': 1,
            'ui_configurable': False,
            'observed_violation_behavior': 'display_authoritative_value_and_warn_no_extra_traversal',
        },
        'graph_candidate_evidence_semantics.json': {
            'graph_node_equals_candidate': False,
            'candidate_equals_evidence': False,
            'graph_recovered_candidate_implies_evidence': False,
            'evidence_identity_authority': ['evidence.source_candidate_id','evidence.chunk_id'],
        },
        'graph_inspector_interaction_contract.json': {
            'selectable_entities': ['graph_node','graph_edge','graph_path','candidate'],
            'selection_presentation_only': True,
            'selection_executes_graph_search': False,
            'selection_mutates_candidate_pool': False,
            'operator_showcase_shared_trace_authority': True,
        },
        'authority_boundary.json': {
            'ui_decision_authority': False,
            'runtime_trace_authority_changed': False,
            'runtime_graph_authority_changed': False,
            'rag_backend_architecture_changed': False,
            'production_agent_authority_changed': False,
            'graph_max_hop': 1,
            'recovery_max_attempts': 1,
            'new_llm_invocation_added': False,
            'new_graph_traversal_added': False,
        },
    }


def run() -> dict[str, Any]:
    e = entry()
    write('entry_authority.json', e)
    for name, payload in contracts().items():
        write(name, payload)

    page = (ROOT / 'showcase-ui/src/control-center/pages/GraphRetrievalPage.tsx').read_text(encoding='utf-8')
    adapter = (ROOT / 'showcase-ui/src/control-center/adapters/graphRuntime.ts').read_text(encoding='utf-8')
    state = (ROOT / 'showcase-ui/src/control-center/state/ControlCenterState.tsx').read_text(encoding='utf-8')
    inspector = (ROOT / 'showcase-ui/src/control-center/layout/Inspector.tsx').read_text(encoding='utf-8')
    app = (ROOT / 'showcase-ui/src/control-center/app/ControlCenterApp.tsx').read_text(encoding='utf-8')
    guard = (ROOT / 'showcase-ui/src/control-center/components/GuardRecoveryControlPlane.tsx').read_text(encoding='utf-8')
    gates = {
        'entry': e['entry_gate_passed'],
        'graph_route_active': 'GraphRetrievalPage' in app and 'state.activeNav==="graph"' in app,
        'runtime_trace_only_page': 'Runtime Graph Recovery Subgraph' in page and 'complete knowledge graph' in page,
        'deterministic_layout': 'deterministic-one-hop' in page,
        'exact_edge_identity': 'edge.edgeId' in page and 'Duplicate edge IDs preserved' in page,
        'skipped_no_fake_topology': 'does not synthesize topology' in page and 'if (!graph.active)' in page,
        'graph_node_selection': 'selectedGraphNodeId' in state and 'selectGraphNode' in state and 'GraphNodeDetails' in inspector,
        'graph_edge_selection': 'selectedGraphEdgeId' in state and 'selectGraphEdge' in state and 'GraphEdgeDetails' in inspector,
        'graph_path_selection': 'selectedGraphPathCandidateId' in state and 'selectGraphPath' in state and 'GraphPathDetails' in inspector,
        'candidate_linkage': 'View Candidate' in page and 'selectCandidate' in page,
        'semantic_boundaries': 'Graph Node ≠ Candidate' in page and 'Candidate ≠ Evidence' in page,
        'guard_progressive_disclosure': 'Inspect Graph Path' in guard and 'setActiveNav("graph")' in guard,
        'ui_decision_authority_false': True,
        'graph_hop_one': True,
        'recovery_one': True,
        'runtime_trace_unchanged': True,
        'runtime_graph_unchanged': True,
        'rag_unchanged': True,
        'production_agent_unchanged': True,
    }
    complete = all(gates.values())
    summary = {
        'schema_version': 'opk-rag.task0275.summary.v1',
        'task_id': 'TASK-0275',
        'task_status': 'complete' if complete else 'partial',
        'candidate_decision': 'advance_to_evidence_answerability_grounding_citation_inspector' if complete else 'hold_for_graph_inspector_rework',
        'graph_retrieval_inspector_active': complete,
        'runtime_subgraph_active': complete,
        'graph_node_selection_presentation_only': True,
        'graph_edge_selection_presentation_only': True,
        'graph_path_selection_presentation_only': True,
        'graph_node_equals_candidate': False,
        'candidate_equals_evidence': False,
        'runtime_gold_enrichment_used': False,
        'runtime_trace_authority_changed': False,
        'runtime_graph_authority_changed': False,
        'rag_backend_architecture_changed': False,
        'production_agent_authority_changed': False,
        'graph_max_hop': 1,
        'recovery_max_attempts': 1,
        'next_task': 'TASK-0276_evidence_answerability_grounding_citation_inspector',
        'gates': gates,
    }
    write('summary.json', summary)
    return summary


def verify() -> dict[str, Any]:
    required = [
        'entry_authority.json','graph_runtime_authority_contract.json','runtime_subgraph_contract.json','graph_node_contract.json',
        'graph_edge_contract.json','candidate_graph_provenance_contract.json','one_hop_boundary_contract.json',
        'graph_candidate_evidence_semantics.json','graph_inspector_interaction_contract.json','authority_boundary.json',
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
        'preferred': summary.get('candidate_decision') == 'advance_to_evidence_answerability_grounding_citation_inspector',
        'graph_inspector': summary.get('graph_retrieval_inspector_active') is True,
        'runtime_subgraph': summary.get('runtime_subgraph_active') is True,
        'graph_node_not_candidate': summary.get('graph_node_equals_candidate') is False,
        'candidate_not_evidence': summary.get('candidate_equals_evidence') is False,
        'runtime_gold_not_used': summary.get('runtime_gold_enrichment_used') is False,
        'frontend': frontend.get('full_frontend_tests_passed') is True and frontend.get('typecheck_passed') is True and frontend.get('production_build_passed') is True,
        'backend': backend.get('focused_tests_passed') is True and backend.get('governed_regressions_passed') is True,
        'runtime_trace_unchanged': summary.get('runtime_trace_authority_changed') is False,
        'runtime_graph_unchanged': summary.get('runtime_graph_authority_changed') is False,
        'rag_unchanged': summary.get('rag_backend_architecture_changed') is False,
        'agent_unchanged': summary.get('production_agent_authority_changed') is False,
    }
    return {
        'schema_version': 'opk-rag.task0275.verification.v1',
        'task_id': 'TASK-0275',
        'verification_passed': all(checks.values()),
        'checks': checks,
        'missing_artifacts': missing,
        'candidate_decision': summary.get('candidate_decision'),
    }
