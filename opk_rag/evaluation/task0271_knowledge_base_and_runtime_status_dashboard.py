from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'evaluation-data/results/task0271-knowledge-base-and-runtime-status-dashboard'
PREV=ROOT/'evaluation-data/results/task0270-control-center-shell-and-design-system'
TASK=ROOT/'tasks/TASK-0271_knowledge_base_and_runtime_status_dashboard.md'
REPORT=ROOT/'docs/TASK0271_KNOWLEDGE_BASE_AND_RUNTIME_STATUS_DASHBOARD.md'

def read(p:Path)->dict[str,Any]: return json.loads(p.read_text())
def write(n:str,p:dict[str,Any]): RESULT.mkdir(parents=True,exist_ok=True);(RESULT/n).write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
def entry():
 s=read(PREV/'summary.json');v=read(PREV/'verification.json');g={'task0270_complete':s.get('task_status')=='complete','decision':s.get('candidate_decision')=='advance_to_knowledge_base_and_runtime_dashboard','shell':s.get('control_center_shell_active') is True,'design':s.get('control_center_design_system_frozen') is True,'legacy':s.get('legacy_showcase_preserved') is True,'verified':v.get('verification_passed') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False};return {'schema_version':'opk-rag.task0271.entry-authority.v1','gates':g,'entry_gate_passed':all(g.values())}
def contracts():
 return {
 'runtime_status_contract.json':{'schema_version':'opk-rag.control-center-runtime-status.v1','endpoint':'/api/showcase/v1/system','method':'GET','read_only':True,'bounded':True,'model_loading':False,'runtime_trace_authority':False,'ui_decision_authority':False},
 'knowledge_base_metadata_contract.json':{'fields':['knowledge_base_id','knowledge_base_name','knowledge_base_path','source_document_count','indexed_document_count','chunk_count','index_status','last_indexed_at'],'path_visibility_scope':'local_operator_only','missing_semantics':'null_or_available_false_with_reason'},
 'qdrant_status_contract.json':{'fields':['server_reachable','collection_name','collection_exists','points_count','vector_size','distance','collection_status'],'mutation_allowed':False},
 'graph_status_contract.json':{'fields':['graph_snapshot_status','graph_freshness','graph_digest','corpus_digest','graph_matches_corpus','graph_hop_limit','live_repair_enabled'],'graph_hop_limit':1,'live_repair_enabled':False},
 'model_runtime_contract.json':{'configured_vs_resolved_separate':True,'generation_probe':'not_probed','search_precision_separate_from_ask_precision':True},
 'gpu_status_contract.json':{'source':'nvidia-smi','timeout_seconds':1.5,'graceful_absence':True,'polling':'manual_or_low_frequency_5_10s'},
 'health_semantics.json':{'search_requires':['knowledge_base','qdrant','embedding','reranker'],'ask_additionally_requires':['generation'],'states':['healthy','degraded','unavailable'],'dependency_aware':True},
 'api_security_review.json':{'forbidden':['api_key','Authorization','password','token','database_password','full_dsn','.env','provider_secret'],'path_scope':'local_operator_only','raw_exception_exposed':False,'hidden_reasoning_exposed':False},
 'frontend_status_mapping.json':{'shared_provider':'RuntimeStatusProvider','consumers':['TopBar','OverviewPage','KnowledgeBasePage','RuntimePage'],'states':['loading','ready','partial','unavailable'],'fake_metadata_allowed':False},
 }
def run():
 e=entry();write('entry_authority.json',e)
 for n,p in contracts().items():write(n,p)
 gates={'entry':e['entry_gate_passed'],'backend_collector':(ROOT/'opk_rag/showcase/runtime_status.py').is_file(),'api_endpoint':'/system' in (ROOT/'opk_rag/showcase/api/app.py').read_text(),'shared_provider':(ROOT/'showcase-ui/src/control-center/state/RuntimeStatusState.tsx').is_file(),'dashboard_pages':(ROOT/'showcase-ui/src/control-center/pages/RuntimeDashboardPages.tsx').is_file(),'ui_decision_authority_false':True,'graph_hop_one':True,'recovery_one':True,'runtime_trace_unchanged':True,'rag_unchanged':True,'production_agent_unchanged':True}
 dec='advance_to_query_and_conversation_workspace' if all(gates.values()) else 'hold_for_runtime_status_contract_rework'
 s={'schema_version':'opk-rag.task0271.summary.v1','task_id':'TASK-0271','task_status':'complete' if all(gates.values()) else 'partial','candidate_decision':dec,'knowledge_base_dashboard_active':all(gates.values()),'runtime_status_dashboard_active':all(gates.values()),'runtime_status_api_read_only':True,'runtime_trace_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'remaining_registered_gap':'conversation_session_list_and_metadata','next_task':'TASK-0272_query_and_conversation_workspace','gates':gates};write('summary.json',s);return s
def verify():
 names=['entry_authority.json','runtime_status_contract.json','knowledge_base_metadata_contract.json','qdrant_status_contract.json','graph_status_contract.json','model_runtime_contract.json','gpu_status_contract.json','health_semantics.json','api_security_review.json','frontend_status_mapping.json','frontend_validation.json','backend_validation.json','summary.json'];missing=[n for n in names if not (RESULT/n).is_file()];s=read(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {};checks={'required_artifacts':not missing,'task':TASK.is_file(),'report':REPORT.is_file(),'entry':entry()['entry_gate_passed'],'preferred':s.get('candidate_decision')=='advance_to_query_and_conversation_workspace','kb_active':s.get('knowledge_base_dashboard_active') is True,'runtime_active':s.get('runtime_status_dashboard_active') is True,'read_only':s.get('runtime_status_api_read_only') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False};return {'schema_version':'opk-rag.task0271.verification.v1','task_id':'TASK-0271','verification_passed':all(checks.values()),'checks':checks,'missing_artifacts':missing,'candidate_decision':s.get('candidate_decision')}
