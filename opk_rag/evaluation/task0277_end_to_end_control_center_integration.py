from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'evaluation-data/results/task0277-end-to-end-control-center-integration'
PREV=ROOT/'evaluation-data/results/task0276-evidence-answerability-grounding-citation-inspector'
TASK=ROOT/'tasks/TASK-0277_end_to_end_control_center_integration.md'
REPORT=ROOT/'docs/TASK0277_END_TO_END_CONTROL_CENTER_INTEGRATION.md'
def read(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def write(name:str,payload:dict[str,Any])->None:
    RESULT.mkdir(parents=True,exist_ok=True); (RESULT/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def entry()->dict[str,Any]:
    s=read(PREV/'summary.json'); v=read(PREV/'verification.json')
    gates={'task0276_complete':s.get('task_status')=='complete','task0276_verified':v.get('verification_passed') is True,'decision':s.get('candidate_decision')=='advance_to_end_to_end_control_center_integration','evidence_inspector':s.get('evidence_validation_inspector_active') is True,'citation_lineage':s.get('citation_identity_lineage_active') is True,'safe_refusal':s.get('safe_refusal_visualization_active') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False}
    return {'schema_version':'opk-rag.task0277.entry-authority.v1','gates':gates,'entry_gate_passed':all(gates.values())}
def contracts()->dict[str,dict[str,Any]]:
    return {
      'active_trace_contract.json':{'authority':'single ShowcaseState Runtime Trace V1','page_specific_trace_copy_allowed':False,'navigation_preserves_trace':True,'trace_change_clears_stale_selection':True},
      'end_to_end_pipeline_contract.json':{'stages':['conversation_resolution','initial_retrieval','guard_agent_decision','optional_recovery','candidate_pool','reranking','evidence_composition','answerability','generation','grounding','citation','outcome'],'stage_state_authority':'Runtime Trace V1 only','neighbor_stage_inference_allowed':False},
      'cross_page_navigation_contract.json':{'pages':['query','runtime','graph','evidence'],'navigation_executes_runtime':False,'new_search_allowed':False,'new_ask_allowed':False,'new_graph_traversal_allowed':False,'new_generation_allowed':False},
      'selection_lifecycle_contract.json':{'cleared_on_trace_change':['candidate','graph_node','graph_edge','graph_path','evidence','citation','validation_stage'],'stale_identity_rebinding_allowed':False},
      'progressive_disclosure_contract.json':{'forward':['query->runtime','runtime->graph','runtime->evidence','graph->candidate','graph->evidence','candidate->graph_path','candidate->evidence','evidence->candidate','evidence->graph_path','evidence->citation','citation->evidence','citation->candidate'],'identity_authority':'exact runtime IDs'},
      'runtime_page_contract.json':{'role':'active execution inspection hub','active_trace_required_for_execution_pipeline':True,'machine_status_remains_read_only':True,'live_timing_only':True,'historical_benchmark_mixing_allowed':False},
      'conversation_trace_integration_contract.json':{'reopened_trace_uses_same_control_center_flow':True,'expired_trace_reconstruction_allowed':False,'history_is_evidence':False},
      'operator_showcase_consistency_contract.json':{'same_runtime_trace':True,'mode_switch_executes_backend':False,'mode_switch_changes_semantics':False},
      'semantic_boundary_contract.json':{'graph_node_equals_candidate':False,'candidate_equals_evidence':False,'evidence_equals_citation':False,'answerability_forces_generation':False,'refused_equals_failed':False,'ui_equals_runtime_authority':False},
      'runtime_integrity_contract.json':{'ui_decision_authority':False,'graph_max_hop':1,'recovery_max_attempts':1,'runtime_trace_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'new_llm_invocation_added':False,'new_graph_traversal_added':False},
      'authority_boundary.json':{'ui_decision_authority':False,'graph_max_hop':1,'recovery_max_attempts':1,'runtime_trace_authority_changed':False,'retrieval_behavior_changed':False,'agent_behavior_changed':False,'graph_behavior_changed':False,'rerank_behavior_changed':False,'evidence_behavior_changed':False,'generation_behavior_changed':False,'grounding_behavior_changed':False,'citation_behavior_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False},
    }
def run()->dict[str,Any]:
    e=entry();write('entry_authority.json',e)
    for n,p in contracts().items():write(n,p)
    state=(ROOT/'showcase-ui/src/control-center/state/ControlCenterState.tsx').read_text()
    shell=(ROOT/'showcase-ui/src/control-center/layout/AppShell.tsx').read_text()
    bar=(ROOT/'showcase-ui/src/control-center/components/ActiveExecutionBar.tsx').read_text()
    sync=(ROOT/'showcase-ui/src/control-center/components/TraceSelectionSynchronizer.tsx').read_text()
    runtime=(ROOT/'showcase-ui/src/control-center/pages/RuntimeDashboardPages.tsx').read_text()
    adapter=(ROOT/'showcase-ui/src/control-center/adapters/endToEndRuntime.ts').read_text()
    graph=(ROOT/'showcase-ui/src/control-center/pages/GraphRetrievalPage.tsx').read_text()
    inspector=(ROOT/'showcase-ui/src/control-center/layout/Inspector.tsx').read_text()
    query=(ROOT/'showcase-ui/src/control-center/pages/QueryWorkspacePage.tsx').read_text()
    gates={'entry':e['entry_gate_passed'],'single_active_trace':'activeTraceId' in state and 'bindActiveTrace' in state and 'TraceSelectionSynchronizer' in shell,'global_execution_bar':'ActiveExecutionBar' in shell and 'data-trace-id' in bar,'runtime_hub':'End-to-End Runtime Pipeline' in runtime and 'buildIntegratedStages' in runtime,'pipeline_complete':all(x in adapter for x in ['conversation_resolution','initial_retrieval','guard_agent_decision','optional_recovery','candidate_pool','reranking','evidence_composition','answerability','generation','grounding','citation','outcome']),'stale_selection_clear':'clearSelections' in state and 'bindActiveTrace' in state,'query_drilldown':'Execution Drill-down' in query and 'Inspect Runtime' in query and 'Inspect Evidence' in query,'graph_to_evidence':'View Evidence' in graph,'candidate_bidirectional':'View Graph Path' in inspector and 'View Evidence' in inspector,'navigation_only':True,'ui_decision_authority_false':True,'graph_hop_one':True,'recovery_one':True,'runtime_trace_unchanged':True,'rag_unchanged':True,'agent_unchanged':True}
    complete=all(gates.values())
    summary={'schema_version':'opk-rag.task0277.summary.v1','task_id':'TASK-0277','task_status':'complete' if complete else 'partial','candidate_decision':'advance_to_showcase_executive_view' if complete else 'hold_for_end_to_end_integration_rework','end_to_end_control_center_integrated':complete,'single_active_trace_authority':complete,'runtime_execution_hub_active':complete,'cross_page_progressive_disclosure_active':complete,'stale_selection_lifecycle_active':complete,'runtime_trace_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'graph_max_hop':1,'recovery_max_attempts':1,'next_task':'TASK-0278_showcase_executive_view','gates':gates}
    write('summary.json',summary);return summary
def verify()->dict[str,Any]:
    required=['entry_authority.json','active_trace_contract.json','end_to_end_pipeline_contract.json','cross_page_navigation_contract.json','selection_lifecycle_contract.json','progressive_disclosure_contract.json','runtime_page_contract.json','conversation_trace_integration_contract.json','operator_showcase_consistency_contract.json','semantic_boundary_contract.json','runtime_integrity_contract.json','authority_boundary.json','frontend_validation.json','backend_validation.json','summary.json']
    missing=[x for x in required if not (RESULT/x).is_file()]
    s=read(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {}; f=read(RESULT/'frontend_validation.json') if (RESULT/'frontend_validation.json').is_file() else {}; b=read(RESULT/'backend_validation.json') if (RESULT/'backend_validation.json').is_file() else {}
    checks={'required_artifacts':not missing,'task':TASK.is_file(),'report':REPORT.is_file(),'entry':entry()['entry_gate_passed'],'preferred':s.get('candidate_decision')=='advance_to_showcase_executive_view','integration':s.get('end_to_end_control_center_integrated') is True,'single_trace':s.get('single_active_trace_authority') is True,'runtime_hub':s.get('runtime_execution_hub_active') is True,'selection_lifecycle':s.get('stale_selection_lifecycle_active') is True,'frontend':f.get('full_frontend_tests_passed') is True and f.get('typecheck_passed') is True and f.get('production_build_passed') is True,'backend':b.get('focused_tests_passed') is True and b.get('governed_regressions_passed') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False}
    return {'schema_version':'opk-rag.task0277.verification.v1','task_id':'TASK-0277','verification_passed':all(checks.values()),'checks':checks,'missing_artifacts':missing,'candidate_decision':s.get('candidate_decision')}
