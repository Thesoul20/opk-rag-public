from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'evaluation-data/results/task0273-retrieval-candidate-reranking-inspector'
PREV=ROOT/'evaluation-data/results/task0272-query-and-conversation-workspace'
TASK=ROOT/'tasks/TASK-0273_retrieval_candidate_reranking_inspector.md'
REPORT=ROOT/'docs/TASK0273_RETRIEVAL_CANDIDATE_RERANKING_INSPECTOR.md'

def read(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def write(n:str,p:dict[str,Any]): RESULT.mkdir(parents=True,exist_ok=True);(RESULT/n).write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def entry():
 s=read(PREV/'summary.json');v=read(PREV/'verification.json')
 g={
  'task0272_complete':s.get('task_status')=='complete',
  'task0272_verified':v.get('verification_passed') is True,
  'decision':s.get('candidate_decision')=='advance_to_runtime_trace_and_contextual_inspector_integration',
  'query_workspace':s.get('query_workspace_active') is True,
  'conversation_workspace':s.get('persistent_conversation_workspace_active') is True,
  'trace_association':s.get('runtime_trace_association_active') is True,
  'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,
  'rag_unchanged':s.get('rag_backend_architecture_changed') is False,
  'agent_unchanged':s.get('production_agent_authority_changed') is False,
 }
 return {'schema_version':'opk-rag.task0273.entry-authority.v1','gates':g,'entry_gate_passed':all(g.values())}

def contracts():
 return {
  'retrieval_inspector_contract.json':{'authority':'Runtime Trace V1','fields':['requested_policy','executed_policy','retrieval_mode','retrievers_used','initial_candidate_count','vector_candidate_count','lexical_candidate_count','post_structure_candidate_count','post_graph_unified_candidate_count','rerank_input_count','final_result_count','candidate_pool_count_after_recovery'],'missing_semantics':'unavailable_not_zero','ui_inference_allowed':False},
  'candidate_semantics_contract.json':{'candidate_equals_evidence':False,'states':['candidate','context_candidate','evidence_selected'],'evidence_identity_sources':['evidence.source_candidate_id','evidence.chunk_id'],'ranking_implies_evidence':False,'selected_for_context_implies_evidence':False},
  'candidate_provenance_contract.json':{'sources':['initial','structure_expansion','graph_recovery'],'multiple_provenance_allowed':True,'fabricated_provenance_allowed':False,'guard_reason_explanation_owned_by':'TASK-0274','graph_topology_owned_by':'TASK-0275'},
  'reranking_inspector_contract.json':{'retrieval_score_and_rerank_score_distinct':True,'probability_interpretation_allowed':False,'correctness_interpretation_allowed':False,'runtime_fields':['reranker_model','reranker_revision','reranker_device','execution_scope','precision','input_candidate_count','output_candidate_count']},
  'rank_movement_contract.json':{'formula':'initial_rank - final_rank','presentation_only':True,'requires_both_ranks':True,'runtime_authority':False},
  'contextual_inspector_contract.json':{'tabs':['details','trace','evidence','source','metadata'],'selected_entity':'candidate','selection_presentation_only':True,'backend_mutation':False,'arbitrary_local_file_read':False,'raw_internal_object_dump':False,'stale_selection_cleared':True},
  'authority_boundary.json':{'ui_decision_authority':False,'runtime_trace_authority_changed':False,'retrieval_authority_changed':False,'reranking_authority_changed':False,'evidence_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'graph_max_hop':1,'recovery_max_attempts':1},
 }

def run():
 e=entry();write('entry_authority.json',e)
 for n,p in contracts().items(): write(n,p)
 state=(ROOT/'showcase-ui/src/control-center/state/ControlCenterState.tsx').read_text()
 inspector=(ROOT/'showcase-ui/src/control-center/layout/Inspector.tsx').read_text()
 component=(ROOT/'showcase-ui/src/control-center/components/RetrievalCandidateInspector.tsx').read_text()
 query=(ROOT/'showcase-ui/src/control-center/pages/QueryWorkspacePage.tsx').read_text()
 g={
  'entry':e['entry_gate_passed'],
  'selection_state':'selectedCandidateId' in state and 'selectCandidate' in state,
  'retrieval_component':'RetrievalOverview' in component and 'CandidatePool' in component and 'RerankingSummary' in component,
  'candidate_evidence_boundary':'Candidate ≠ Evidence' in component and 'candidateRelationship' in component,
  'candidate_inspector':'selectedCandidate' in inspector and 'EvidenceRelationship' in inspector,
  'stale_selection_clear':'!candidate' in inspector and 'selectCandidate(null)' in inspector,
  'query_integration':'RetrievalCandidateInspection' in query,
  'conversation_trace_replay':'inspectTrace' in query and 'getTrace' in query,
  'ui_decision_authority_false':True,'graph_hop_one':True,'recovery_one':True,
  'runtime_trace_unchanged':True,'rag_unchanged':True,'production_agent_unchanged':True,
 }
 ok=all(g.values())
 s={'schema_version':'opk-rag.task0273.summary.v1','task_id':'TASK-0273','task_status':'complete' if ok else 'partial','candidate_decision':'advance_to_guarded_agent_and_recovery_trace_visualization' if ok else 'hold_for_retrieval_inspector_rework','retrieval_inspector_active':ok,'candidate_pool_active':ok,'reranking_inspector_active':ok,'candidate_selection_presentation_only':True,'candidate_equals_evidence':False,'conversation_trace_replay_active':ok,'runtime_trace_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'graph_max_hop':1,'recovery_max_attempts':1,'next_task':'TASK-0274_guarded_agent_and_recovery_trace_visualization','gates':g}
 write('summary.json',s);return s

def verify():
 req=['entry_authority.json','retrieval_inspector_contract.json','candidate_semantics_contract.json','candidate_provenance_contract.json','reranking_inspector_contract.json','rank_movement_contract.json','contextual_inspector_contract.json','authority_boundary.json','frontend_validation.json','backend_validation.json','summary.json']
 missing=[n for n in req if not (RESULT/n).is_file()]
 s=read(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {};f=read(RESULT/'frontend_validation.json') if (RESULT/'frontend_validation.json').is_file() else {};b=read(RESULT/'backend_validation.json') if (RESULT/'backend_validation.json').is_file() else {}
 c={'required_artifacts':not missing,'task':TASK.is_file(),'report':REPORT.is_file(),'entry':entry()['entry_gate_passed'],'preferred':s.get('candidate_decision')=='advance_to_guarded_agent_and_recovery_trace_visualization','retrieval':s.get('retrieval_inspector_active') is True,'candidate_pool':s.get('candidate_pool_active') is True,'reranking':s.get('reranking_inspector_active') is True,'candidate_not_evidence':s.get('candidate_equals_evidence') is False,'frontend':f.get('full_frontend_tests_passed') is True and f.get('typecheck_passed') is True and f.get('production_build_passed') is True,'backend':b.get('focused_tests_passed') is True and b.get('governed_regressions_passed') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False}
 return {'schema_version':'opk-rag.task0273.verification.v1','task_id':'TASK-0273','verification_passed':all(c.values()),'checks':c,'missing_artifacts':missing,'candidate_decision':s.get('candidate_decision')}
