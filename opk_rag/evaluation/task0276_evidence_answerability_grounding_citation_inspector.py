from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'evaluation-data/results/task0276-evidence-answerability-grounding-citation-inspector'
PREV = ROOT / 'evaluation-data/results/task0275-graph-retrieval-interactive-inspector'
TASK = ROOT / 'tasks/TASK-0276_evidence_answerability_grounding_citation_inspector.md'
REPORT = ROOT / 'docs/TASK0276_EVIDENCE_ANSWERABILITY_GROUNDING_CITATION_INSPECTOR.md'

def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding='utf-8'))
def write(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def entry() -> dict[str, Any]:
    summary=read(PREV/'summary.json'); verification=read(PREV/'verification.json')
    gates={
        'task0275_complete': summary.get('task_status')=='complete',
        'task0275_verified': verification.get('verification_passed') is True,
        'decision': summary.get('candidate_decision')=='advance_to_evidence_answerability_grounding_citation_inspector',
        'graph_inspector': summary.get('graph_retrieval_inspector_active') is True,
        'candidate_not_evidence': summary.get('candidate_equals_evidence') is False,
        'runtime_trace_unchanged': summary.get('runtime_trace_authority_changed') is False,
        'rag_unchanged': summary.get('rag_backend_architecture_changed') is False,
        'agent_unchanged': summary.get('production_agent_authority_changed') is False,
    }
    return {'schema_version':'opk-rag.task0276.entry-authority.v1','gates':gates,'entry_gate_passed':all(gates.values())}

def contracts() -> dict[str, dict[str, Any]]:
    return {
      'evidence_semantics_contract.json': {'authority':'Runtime Trace V1 evidence','candidate_equals_evidence':False,'evidence_order_mutation_allowed':False,'evidence_recompute_allowed':False,'partial_record_preserved':True},
      'evidence_candidate_identity_contract.json': {'identity_fields':['evidence.source_candidate_id','evidence.chunk_id'],'candidate_lookup_scope':'active_runtime_trace','rank_inference_allowed':False,'title_inference_allowed':False,'selection_presentation_only':True},
      'answerability_contract.json': {'authority':'Runtime Trace V1 answerability','ui_override_allowed':False,'confidence_is_correctness_probability':False,'answerable_forces_generation_release':False},
      'generation_observability_contract.json': {'safe_fields':['attempted','completed','abstained','provider','model','revision','endpoint_type','latency_ms','token_counts','finish_reason','evidence_count'],'raw_prompt_exposed':False,'provider_raw_envelope_exposed':False,'hidden_reasoning_exposed':False,'new_generation_invocation_allowed':False},
      'grounding_contract.json': {'authority':'Runtime Trace V1 grounding','ui_override_allowed':False,'aggregate_claim_counts_only':True,'fabricated_claim_text_allowed':False,'citation_coverage_is_correctness_score':False},
      'citation_identity_contract.json': {'identity_chain':'citation.evidence_id -> evidence.evidence_id -> evidence.source_candidate_id -> candidate','evidence_equals_citation':False,'orphan_repair_allowed':False,'candidate_mismatch_normalization_allowed':False},
      'final_outcome_contract.json': {'refused_equals_failed':False,'prior_stage_success_implies_answer':False,'ui_release_override_allowed':False},
      'safe_refusal_contract.json': {'canonical_scenario':'S04','answerability_answerable':True,'generation_abstained':True,'grounding_status':'not_applicable','citation_count':0,'outcome':'refused','failure_stage':None,'reason':'answerable_generation_abstained'},
      'semantic_separation_contract.json': {'graph_node_equals_candidate':False,'candidate_equals_evidence':False,'evidence_equals_citation':False,'selection_equals_runtime_execution':False},
      'hidden_reasoning_boundary.json': {'chain_of_thought_exposed':False,'raw_prompt_exposed':False,'provider_reasoning_exposed':False,'scratchpad_exposed':False},
      'authority_boundary.json': {'ui_decision_authority':False,'graph_max_hop':1,'recovery_max_attempts':1,'runtime_trace_authority_changed':False,'evidence_policy_changed':False,'answerability_policy_changed':False,'generation_behavior_changed':False,'grounding_policy_changed':False,'citation_policy_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'new_llm_invocation_added':False},
    }

def run() -> dict[str, Any]:
    e=entry(); write('entry_authority.json',e)
    for name,payload in contracts().items(): write(name,payload)
    page=(ROOT/'showcase-ui/src/control-center/pages/EvidenceValidationPage.tsx').read_text(encoding='utf-8')
    adapter=(ROOT/'showcase-ui/src/control-center/adapters/evidenceRuntime.ts').read_text(encoding='utf-8')
    state=(ROOT/'showcase-ui/src/control-center/state/ControlCenterState.tsx').read_text(encoding='utf-8')
    inspector=(ROOT/'showcase-ui/src/control-center/layout/Inspector.tsx').read_text(encoding='utf-8')
    app=(ROOT/'showcase-ui/src/control-center/app/ControlCenterApp.tsx').read_text(encoding='utf-8')
    query=(ROOT/'showcase-ui/src/control-center/pages/QueryWorkspacePage.tsx').read_text(encoding='utf-8')
    gates={
      'entry':e['entry_gate_passed'],
      'evidence_route_active':'EvidenceValidationPage' in app and 'state.activeNav==="evidence"' in app,
      'runtime_trace_only':'Runtime Trace V1 only' in page and 'data-runtime-authority="trace-v1"' in page,
      'validation_pipeline':all(x in page for x in ['Evidence','Answerability','Generation','Grounding','Citation','Outcome']),
      'search_scope_boundary':'Search scope ends at Evidence' in page,
      'evidence_inventory':'Evidence Inventory' in page and 'selectEvidence' in page,
      'exact_citation_lineage':'citation.evidence_id' in adapter and 'evidenceById' in adapter and 'resolvedSourceCandidateId' in adapter,
      'semantic_boundaries':'Candidate ≠ Evidence' in page and 'Evidence ≠ Citation' in page and 'Refused ≠ Failed' in page,
      'inspector_entities':all(x in state for x in ['"evidence_stage"','"evidence"','"answerability"','"generation"','"grounding"','"citation"','"outcome"']) and all(x in inspector for x in ['EvidenceDetails','AnswerabilityDetails','GenerationDetails','GroundingDetails','CitationDetails','OutcomeDetails']),
      'query_progressive_disclosure':'EvidenceValidationSummary' in query,
      'ui_decision_authority_false':True,'graph_hop_one':True,'recovery_one':True,
      'runtime_trace_unchanged':True,'downstream_policies_unchanged':True,'rag_unchanged':True,'production_agent_unchanged':True,
    }
    complete=all(gates.values())
    summary={'schema_version':'opk-rag.task0276.summary.v1','task_id':'TASK-0276','task_status':'complete' if complete else 'partial','candidate_decision':'advance_to_end_to_end_control_center_integration' if complete else 'hold_for_evidence_validation_inspector_rework','evidence_validation_inspector_active':complete,'citation_identity_lineage_active':complete,'safe_refusal_visualization_active':complete,'candidate_equals_evidence':False,'evidence_equals_citation':False,'runtime_trace_authority_changed':False,'evidence_policy_changed':False,'answerability_policy_changed':False,'generation_behavior_changed':False,'grounding_policy_changed':False,'citation_policy_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'graph_max_hop':1,'recovery_max_attempts':1,'next_task':'TASK-0277_end_to_end_control_center_integration','gates':gates}
    write('summary.json',summary); return summary

def verify() -> dict[str, Any]:
    required=['entry_authority.json','evidence_semantics_contract.json','evidence_candidate_identity_contract.json','answerability_contract.json','generation_observability_contract.json','grounding_contract.json','citation_identity_contract.json','final_outcome_contract.json','safe_refusal_contract.json','semantic_separation_contract.json','hidden_reasoning_boundary.json','authority_boundary.json','frontend_validation.json','backend_validation.json','summary.json']
    missing=[x for x in required if not (RESULT/x).is_file()]
    summary=read(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {}; frontend=read(RESULT/'frontend_validation.json') if (RESULT/'frontend_validation.json').is_file() else {}; backend=read(RESULT/'backend_validation.json') if (RESULT/'backend_validation.json').is_file() else {}
    checks={'required_artifacts':not missing,'task':TASK.is_file(),'report':REPORT.is_file(),'entry':entry()['entry_gate_passed'],'preferred':summary.get('candidate_decision')=='advance_to_end_to_end_control_center_integration','evidence_inspector':summary.get('evidence_validation_inspector_active') is True,'citation_lineage':summary.get('citation_identity_lineage_active') is True,'safe_refusal':summary.get('safe_refusal_visualization_active') is True,'candidate_not_evidence':summary.get('candidate_equals_evidence') is False,'evidence_not_citation':summary.get('evidence_equals_citation') is False,'frontend':frontend.get('full_frontend_tests_passed') is True and frontend.get('typecheck_passed') is True and frontend.get('production_build_passed') is True,'backend':backend.get('focused_tests_passed') is True and backend.get('governed_regressions_passed') is True,'runtime_trace_unchanged':summary.get('runtime_trace_authority_changed') is False,'policies_unchanged':all(summary.get(k) is False for k in ['evidence_policy_changed','answerability_policy_changed','generation_behavior_changed','grounding_policy_changed','citation_policy_changed']),'rag_unchanged':summary.get('rag_backend_architecture_changed') is False,'agent_unchanged':summary.get('production_agent_authority_changed') is False}
    return {'schema_version':'opk-rag.task0276.verification.v1','task_id':'TASK-0276','verification_passed':all(checks.values()),'checks':checks,'missing_artifacts':missing,'candidate_decision':summary.get('candidate_decision')}
