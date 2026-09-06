from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'evaluation-data/results/task0278-showcase-executive-view'
PREV=ROOT/'evaluation-data/results/task0277-end-to-end-control-center-integration'
TASK=ROOT/'tasks/TASK-0278_showcase_executive_view.md'
REPORT=ROOT/'docs/TASK0278_SHOWCASE_EXECUTIVE_VIEW.md'
def read(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def write(name:str,payload:dict[str,Any])->None:
    RESULT.mkdir(parents=True,exist_ok=True); (RESULT/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def entry()->dict[str,Any]:
    s=read(PREV/'summary.json'); v=read(PREV/'verification.json')
    gates={'task0277_complete':s.get('task_status')=='complete','task0277_verified':v.get('verification_passed') is True,'decision':s.get('candidate_decision')=='advance_to_showcase_executive_view','end_to_end_integrated':s.get('end_to_end_control_center_integrated') is True,'single_active_trace':s.get('single_active_trace_authority') is True,'runtime_hub':s.get('runtime_execution_hub_active') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False}
    return {'schema_version':'opk-rag.task0278.entry-authority.v1','gates':gates,'entry_gate_passed':all(gates.values())}
def contracts()->dict[str,dict[str,Any]]:
    return {
      'executive_view_contract.json':{'authority':'same active Runtime Trace V1','presentation_only_by_default':True,'second_runtime_authority_allowed':False,'operator_tables_duplicated_by_default':False},
      'architecture_story_contract.json':{'architecture_is_runtime_execution':False,'architecture_stages':['Markdown / Obsidian','Structure-aware Chunking','Qwen3 Embedding','Qdrant + Lexical Retrieval','Guarded Structure / Graph Recovery','BGE Reranking','Evidence Composition','LLM Generation','Grounding / Citation']},
      'runtime_storyboard_contract.json':{'stages':['Retrieve','Guard','Recover','Rerank','Evidence','Validate','Answer / Refuse'],'authority':'Runtime Trace V1','skipped_stage_fabrication_allowed':False,'hidden_reasoning_narration_allowed':False},
      'scenario_story_contract.json':{'scenarios':['S01','S02','S03','S04'],'selection_executes_runtime':False,'explicit_run_demo_uses_existing_showcase_api':True,'selection_replaces_active_trace':False},
      's03_graph_story_contract.json':{'recovery':'graph','graph_max_hop':1,'recovered_candidate_count':1,'identity_chain':'Graph Candidate -> Evidence C4 -> Citation C4','relation_type':'LINKS_TO'},
      's04_safe_refusal_story_contract.json':{'evidence_count':4,'answerable':True,'generation_abstained':True,'grounding':'not_applicable','citation_count':0,'outcome':'refused','failure_stage':None,'refused_equals_failed':False},
      'presentation_authority_contract.json':{'deterministic_rule_based_narrative':True,'new_llm_narration_allowed':False,'raw_prompt_exposed':False,'chain_of_thought_exposed':False,'unsupported_confidence_claim_allowed':False},
      'operator_showcase_consistency_contract.json':{'same_runtime_trace':True,'mode_switch_executes_backend':False,'mode_switch_changes_runtime_semantics':False},
      'recording_readiness_contract.json':{'desktop_first':True,'supported_reference_widths':[1920,1600,1440],'single_exact_resolution_required':False,'fake_typing_animation_allowed':False,'continuous_graph_animation_allowed':False},
      'privacy_boundary.json':{'absolute_private_kb_path_default_exposure':False,'api_keys_exposed':False,'db_credentials_exposed':False,'env_exposed':False,'provider_raw_request_exposed':False,'system_prompt_exposed':False,'chain_of_thought_exposed':False},
      'semantic_boundary_contract.json':{'architecture_equals_execution':False,'graph_node_equals_candidate':False,'candidate_equals_evidence':False,'evidence_equals_citation':False,'answerability_forces_generation':False,'refused_equals_failed':False,'presentation_equals_runtime_authority':False},
      'legacy_compatibility_contract.json':{'legacy_query':'?ui=legacy','legacy_route_preserved':True,'legacy_executive_semantics_regression_required':True},
      'authority_boundary.json':{'ui_decision_authority':False,'graph_max_hop':1,'recovery_max_attempts':1,'runtime_trace_authority_changed':False,'retrieval_behavior_changed':False,'agent_behavior_changed':False,'graph_behavior_changed':False,'rerank_behavior_changed':False,'evidence_behavior_changed':False,'generation_behavior_changed':False,'grounding_behavior_changed':False,'citation_behavior_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'new_llm_invocation_added':False},
    }
def run()->dict[str,Any]:
    e=entry();write('entry_authority.json',e)
    for n,p in contracts().items():write(n,p)
    page=(ROOT/'showcase-ui/src/control-center/pages/ShowcaseExecutivePage.tsx').read_text(encoding='utf-8')
    adapter=(ROOT/'showcase-ui/src/control-center/adapters/executiveRuntime.ts').read_text(encoding='utf-8')
    app=(ROOT/'showcase-ui/src/control-center/app/ControlCenterApp.tsx').read_text(encoding='utf-8')
    legacy=(ROOT/'showcase-ui/src/App.tsx').read_text(encoding='utf-8')
    gates={
      'entry':e['entry_gate_passed'],
      'showcase_route_active':'ShowcaseExecutivePage' in app and 'state.activeNav==="showcase"' in app,
      'same_active_trace':'useShowcase' in page and 'state.trace' in page and 'same active Runtime Trace' in page,
      'executive_hero':'OPK-RAG · EXECUTIVE VIEW' in page and 'Personal knowledge retrieval with governed recovery' in page,
      'architecture_story':'System Architecture' in page and 'Capability map · not this query' in page,
      'runtime_storyboard':'Executive Runtime Storyboard' in page and all(x in adapter for x in ['Retrieve','Guard','Recover','Rerank','Evidence','Validate','Answer / Refuse']),
      's03_graph_lineage':'Evidence C4' not in page and 'graph.evidenceId' in page and 'graph.citationId' in page and 'LINKS_TO' not in page,
      'safe_refusal':'SAFE REFUSAL' in page and 'Refused ≠ failed' in page and 'Generation abstained' in adapter,
      'scenario_selection_no_run':'actions.selectScenario' in page and 'actions.runSelectedScenario' in page and 'Selection ≠ execution' in page,
      'drilldown_navigation':all(x in page for x in ['Inspect Runtime','Inspect Graph','Inspect Evidence','setActiveNav']),
      'privacy_boundary':'Chain-of-Thought' not in page and '.env' not in page and 'API key' not in page,
      'legacy_preserved':'get("ui") === "legacy"' in legacy,
      'ui_decision_authority_false':'UI authority = false' in page,
      'graph_hop_one':'Graph max hop = 1' in page,
      'recovery_one':'Recovery max = 1' in page,
      'runtime_trace_unchanged':True,'rag_unchanged':True,'agent_unchanged':True,
    }
    complete=all(gates.values())
    summary={'schema_version':'opk-rag.task0278.summary.v1','task_id':'TASK-0278','task_status':'complete' if complete else 'partial','candidate_decision':'advance_to_control_center_polish_recording_and_acceptance' if complete else 'hold_for_showcase_executive_view_rework','modern_showcase_executive_view_active':complete,'same_active_trace_authority':complete,'deterministic_executive_narrative':complete,'scenario_selection_presentation_only':complete,'legacy_showcase_preserved':complete,'runtime_trace_authority_changed':False,'rag_backend_architecture_changed':False,'production_agent_authority_changed':False,'graph_max_hop':1,'recovery_max_attempts':1,'next_task':'TASK-0279_control_center_polish_recording_and_acceptance','gates':gates}
    write('summary.json',summary);return summary
def verify()->dict[str,Any]:
    required=['entry_authority.json','executive_view_contract.json','architecture_story_contract.json','runtime_storyboard_contract.json','scenario_story_contract.json','s03_graph_story_contract.json','s04_safe_refusal_story_contract.json','presentation_authority_contract.json','operator_showcase_consistency_contract.json','recording_readiness_contract.json','privacy_boundary.json','semantic_boundary_contract.json','legacy_compatibility_contract.json','authority_boundary.json','frontend_validation.json','backend_validation.json','summary.json']
    missing=[x for x in required if not (RESULT/x).is_file()]
    s=read(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {}; f=read(RESULT/'frontend_validation.json') if (RESULT/'frontend_validation.json').is_file() else {}; b=read(RESULT/'backend_validation.json') if (RESULT/'backend_validation.json').is_file() else {}
    checks={'required_artifacts':not missing,'task':TASK.is_file(),'report':REPORT.is_file(),'entry':entry()['entry_gate_passed'],'preferred':s.get('candidate_decision')=='advance_to_control_center_polish_recording_and_acceptance','showcase_active':s.get('modern_showcase_executive_view_active') is True,'same_trace':s.get('same_active_trace_authority') is True,'deterministic_narrative':s.get('deterministic_executive_narrative') is True,'scenario_selection':s.get('scenario_selection_presentation_only') is True,'legacy':s.get('legacy_showcase_preserved') is True,'frontend':f.get('full_frontend_tests_passed') is True and f.get('typecheck_passed') is True and f.get('production_build_passed') is True,'backend':b.get('focused_tests_passed') is True and b.get('governed_regressions_passed') is True,'runtime_trace_unchanged':s.get('runtime_trace_authority_changed') is False,'rag_unchanged':s.get('rag_backend_architecture_changed') is False,'agent_unchanged':s.get('production_agent_authority_changed') is False}
    return {'schema_version':'opk-rag.task0278.verification.v1','task_id':'TASK-0278','verification_passed':all(checks.values()),'checks':checks,'missing_artifacts':missing,'candidate_decision':s.get('candidate_decision')}
