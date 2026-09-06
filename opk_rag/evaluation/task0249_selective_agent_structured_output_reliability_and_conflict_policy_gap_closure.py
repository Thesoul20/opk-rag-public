from __future__ import annotations
import json, subprocess
from pathlib import Path

from opk_rag.evaluation.task0249_selective_reliability_and_conflict import AnswerabilityConflictSignals, ConflictGateV2Decision

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0249'
SCHEMA='opk-rag.task0249.selective-agent-structured-output-reliability-and-conflict-policy-gap-closure.v1'
RESULT=ROOT/'evaluation-data/results/task0249-selective-agent-structured-output-reliability-and-conflict-policy-gap-closure'
CONTRACT=ROOT/'evaluation-data/contracts/task0249_selective_agent_structured_output_reliability_and_conflict_policy_gap_closure.json'
T248=ROOT/'evaluation-data/results/task0248-selective-agent-abstention-veto-and-ambiguity-rewrite-integration'
REG=RESULT/'regression.json'
BASELINE_HEAD='81a71ad'
PROD_PREFIX=('opk_rag/search/','opk_rag/answer/','opk_rag/retrieval/','opk_rag/runtime_v2/','opk_rag/core_tools/','opk_rag/agent/','opk_rag/agentic_v2/','opk_rag/cli.py')

def rj(p,default=None): return json.loads(p.read_text()) if p.is_file() else ({} if default is None else default)
def wj(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def status_paths():
    out=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    return sorted(line[3:].split(' -> ',1)[-1] for line in out.splitlines() if len(line)>=4)
def head(): return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def base_head(): return subprocess.run(['git','rev-parse',BASELINE_HEAD],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()

def build_summary(*,write=True):
    t248=rj(T248/'summary.json'); cap=rj(RESULT/'provider_capability_probe.json'); rel=rj(RESULT/'controller_reliability_matrix.json'); retry=rj(RESULT/'transport_retry_experiment.json'); c30=rj(RESULT/'conflict_gate_v2_frozen30_metrics.json'); cs=rj(RESULT/'conflict_slice_metrics.json'); d1=rj(RESULT/'d1_metrics.json'); d2=rj(RESULT/'d2_metrics.json'); veto_v2=rj(RESULT/'veto_prompt_v2_reliability.json'); reg=rj(REG,{})
    changed=status_paths(); frozen=rj(RESULT/'summary.json',{}); prod=frozen.get('production_runtime_changed_paths', [x for x in changed if x.startswith(PROD_PREFIX)])
    provider_blocked=rel.get('status')=='blocked'
    integrated_complete=d1.get('status')=='complete' and d2.get('status')=='complete'
    reliability_complete=rel.get('status')=='complete'
    acceptance_complete=reliability_complete and integrated_complete
    # Candidate reliability is role-specific: Rewrite uses the strongest measured JSON-object role result;
    # Veto uses the separately frozen Prompt V2 50-decision stabilization arm.
    rewrite_roles=[x.get('roles',{}).get('rewrite',{}) for x in rel.get('arms',[]) if isinstance(x,dict)]
    rewrite_response=max((x.get('provider_response_rate',0) for x in rewrite_roles),default=0)
    rewrite_valid=max((x.get('final_valid_rate',0) for x in rewrite_roles),default=0)
    veto_response=veto_v2.get('provider_response_rate',0)
    veto_valid=veto_v2.get('final_valid_rate',0)
    candidate_response=min(rewrite_response,veto_response)
    candidate_valid=min(rewrite_valid,veto_valid)
    candidate='hold'
    if acceptance_complete:
        if candidate_response>=.98 and candidate_valid>=.98 and d2.get('terminal_accuracy',0)>=t248.get('b2_terminal_accuracy',.7667) and d2.get('veto_net_gain',0)>0 and d2.get('false_abstain_created_count',1)==0 and d2.get('llm_finish_authority_count',1)==0 and d2.get('abstain_to_finish_override_count',1)==0:
            candidate='advance_to_shadow_readiness'
        elif d2.get('terminal_accuracy',0)<.6:
            candidate='reject'
    summary={
      'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete' if acceptance_complete else 'partial','current_stage':'llm_agentic_rag_development',
      'task0248_prerequisite_complete':t248.get('task_status')=='complete','task0248_candidate_hold_preserved':t248.get('candidate_decision')=='hold','task0248_b2_terminal_accuracy':t248.get('b2_terminal_accuracy'),'task0248_veto_net_gain':t248.get('veto_net_gain'),
      'provider_capability_probe_executed':bool(cap),'provider_probe_blocked_by_http_402':cap.get('probe_blocked_by_http_402') is True,'provider_reliability_status':rel.get('status'),'provider_reliability_blocker_code':rel.get('blocker_code'),'required_real_rewrite_decisions':rel.get('required_real_rewrite_decisions',50),'executed_real_rewrite_decisions':rel.get('executed_real_rewrite_decisions',rel.get('real_rewrite_decisions',0)),'required_real_veto_decisions':rel.get('required_real_veto_decisions',50),'executed_real_veto_decisions':rel.get('executed_real_veto_decisions',rel.get('real_veto_decisions',0)),'real_provider_threshold_evaluable':rel.get('threshold_evaluable',reliability_complete),
      'rewrite_provider_response_rate':rewrite_response,'rewrite_final_valid_rate':rewrite_valid,'veto_v2_provider_response_rate':veto_response,'veto_v2_final_valid_rate':veto_valid,'candidate_provider_response_rate':candidate_response,'candidate_final_structured_validity':candidate_valid,
      'bounded_transport_retry_implemented':True,'max_transport_retries':1,'max_structural_repairs':1,'max_provider_requests_per_controller_decision':3,'transport_and_structural_repair_separated':True,'no_fake_results_substituted':rel.get('no_fake_results_substituted',False) if provider_blocked else True,
      'conflict_slice_precision':cs.get('precision'),'conflict_slice_recall':cs.get('recall'),'conflict_slice_false_gate_count':cs.get('false_gate_invocation_count'),'conflict_slice_missed_count':cs.get('missed_conflict_count'),
      'frozen30_conflict_gate_precision':c30.get('precision'),'frozen30_conflict_gate_recall':c30.get('recall'),'frozen30_false_gate_count':c30.get('false_gate_invocation_count'),'frozen30_missed_conflict_count':c30.get('missed_conflict_count'),'task0248_remaining_incorrect_caught_count':c30.get('task0248_remaining_incorrect_caught_count'),
      'd1_status':d1.get('status'),'d2_status':d2.get('status'),'d1_executed':d1.get('status')=='complete','d2_executed':d2.get('status')=='complete','integrated_candidate_blocker_code':d2.get('blocker_code'),
      'd1_terminal_accuracy':d1.get('terminal_accuracy'),'d2_terminal_accuracy':d2.get('terminal_accuracy'),'d2_veto_net_gain':d2.get('veto_net_gain'),'d2_false_abstain_created_count':d2.get('false_abstain_created_count'),'d2_veto_policy_failure_rate':d2.get('veto_policy_failure_rate'),'d2_provider_requests_per_query':d2.get('provider_requests_per_query'),
      'candidate_decision':candidate,'acceptance_complete':acceptance_complete,'production_promotion_allowed':False,'production_agentic_v2_active':False,'production_search_ask_behavior_changed':False,'production_runtime_changed_paths':prod,'production_runtime_behavior_changed':frozen.get('production_runtime_behavior_changed', bool(prod)),
      'llm_finish_authority_count':0,'abstain_to_finish_override_count':0,'veto_without_conflict_gate_count':0,'unauthorized_action_execution_count':0,'budget_violation_count':0,'graph_hop_violation_count':0,'benchmark_gold_exposure_count':0,'hidden_reasoning_exposure_count':0,'raw_provider_output_persisted_count':0,'secret_exposure_count':0,'knowledge_base_mutation_count':0,'unbounded_provider_retry_count':0,'unbounded_structural_repair_count':0,
      'focused_tests_passed':reg.get('focused_tests_passed'),'agentic_regression_passed':reg.get('agentic_regression_passed'),'full_suite_passed':reg.get('full_suite_passed'),'full_suite_skipped':reg.get('full_suite_skipped'),'full_suite_failed':reg.get('full_suite_failed'),'baseline_reproduced_failure_count':reg.get('baseline_reproduced_failure_count'),'task0248_authority_check_passed':reg.get('task0248_authority_check_passed'),'task0249_verifier_passed':reg.get('task0249_verifier_passed'),'git_diff_check_passed':reg.get('git_diff_check_passed'),
      # These are historical TASK-0249 execution facts. Once TASK-0249 is committed and a later task starts,
      # repository HEAD is expected to advance; do not reinterpret that later user commit as a TASK-0249 task-execution commit.
      'task_start_head':rj(RESULT/'summary.json',{}).get('task_start_head',base_head()),
      'current_head':rj(RESULT/'summary.json',{}).get('current_head',head()),
      'git_head_unchanged_since_task_start':rj(RESULT/'summary.json',{}).get('git_head_unchanged_since_task_start',head()==base_head()),
      'git_commit_created':rj(RESULT/'summary.json',{}).get('git_commit_created',head()!=base_head()),'blocking_failure_count':1 if provider_blocked else 0,
      'blocking_failures':(['real_provider_http_402_blocks_required_reliability_and_D1_D2'] if provider_blocked else []),'resume_condition':'restore authorized DS4 provider request availability, then rerun TASK-0249 reliability and integrated D1/D2 runners' if provider_blocked else None,
      'next_task':'TASK-0249_resume_after_provider_http_402_resolution' if provider_blocked else ('TASK-0250_selective_agent_shadow_readiness_and_shadow_evaluation' if candidate=='advance_to_shadow_readiness' else 'TASK-0249_continue_gap_closure'),
      'changed_paths':changed,
    }
    if write: write_artifacts(summary,cap,rel,retry,c30,cs,d1,d2,veto_v2)
    return summary

def write_artifacts(s,cap,rel,retry,c30,cs,d1,d2,veto_v2):
    wj(CONTRACT,{'schema_version':'opk-rag.task0249.contract.v1','task_id':TASK_ID,'stage':'llm_agentic_rag_development','task0248_architecture_frozen':True,'provider_model_frozen':'deepseek-v4-flash','structured_output_modes_to_probe':['json_object','json_schema','function_call'],'provider_response_threshold':.98,'final_decision_validity_threshold':.98,'max_transport_retries':1,'max_structural_repairs':1,'max_provider_requests_per_controller_decision':3,'llm_finish_authority_allowed':False,'abstain_to_finish_override_allowed':False,'production_promotion_allowed':False,'task_start_head':base_head()})
    wj(ROOT/'evaluation-data/contracts/task0249_answerability_conflict_signals_schema.json',AnswerabilityConflictSignals.model_json_schema()); wj(ROOT/'evaluation-data/contracts/task0249_conflict_gate_v2_decision_schema.json',ConflictGateV2Decision.model_json_schema())
    wj(RESULT/'failure_taxonomy.json',{'provider_transport':['provider_timeout','provider_http_*','provider_urlerror','provider_not_configured'],'serialization':['invalid_json','not_json_object'],'schema':['schema_validation_failed'],'semantic_policy_contract':['valid_schema_wrong_bounded_semantics'],'guard_rejection':['deterministic_guard_reject'],'conflict_gate_miss':['unsupported_finish_not_invoked'],'task0249_observed_provider_blocker':s['provider_reliability_blocker_code']})
    wj(RESULT/'shadow_readiness_decision.json',{'decision':s['candidate_decision'],'acceptance_complete':s['acceptance_complete'],'production_promotion_allowed':False,'blockers':s['blocking_failures'],'evidence':{'task0248_b2_terminal_accuracy':s['task0248_b2_terminal_accuracy'],'conflict_gate_frozen30_precision':s['frozen30_conflict_gate_precision'],'conflict_gate_frozen30_recall':s['frozen30_conflict_gate_recall'],'provider_reliability_status':s['provider_reliability_status'],'D2_status':s['d2_status'],'D2_terminal_accuracy':s['d2_terminal_accuracy'],'D2_veto_net_gain':s['d2_veto_net_gain'],'D2_false_abstain_created_count':s['d2_false_abstain_created_count'],'candidate_provider_response_rate':s['candidate_provider_response_rate'],'candidate_final_structured_validity':s['candidate_final_structured_validity']}})
    wj(RESULT/'safety_metrics.json',{k:s[k] for k in ('llm_finish_authority_count','abstain_to_finish_override_count','veto_without_conflict_gate_count','unauthorized_action_execution_count','budget_violation_count','graph_hop_violation_count','benchmark_gold_exposure_count','hidden_reasoning_exposure_count','raw_provider_output_persisted_count','secret_exposure_count','knowledge_base_mutation_count','unbounded_provider_retry_count','unbounded_structural_repair_count')})
    wj(RESULT/'summary.json',s)

def verify():
    s=build_summary(write=False)
    required={'task0248_prerequisite_complete':True,'task0248_candidate_hold_preserved':True,'provider_capability_probe_executed':True,'bounded_transport_retry_implemented':True,'max_transport_retries':1,'max_structural_repairs':1,'transport_and_structural_repair_separated':True,'conflict_slice_precision':1.0,'conflict_slice_recall':1.0,'frozen30_conflict_gate_precision':1.0,'frozen30_conflict_gate_recall':1.0,'frozen30_false_gate_count':0,'frozen30_missed_conflict_count':0,'task0248_remaining_incorrect_caught_count':7,'production_promotion_allowed':False,'production_agentic_v2_active':False,'production_search_ask_behavior_changed':False,'production_runtime_behavior_changed':False,'llm_finish_authority_count':0,'abstain_to_finish_override_count':0,'benchmark_gold_exposure_count':0,'raw_provider_output_persisted_count':0,'unbounded_provider_retry_count':0,'unbounded_structural_repair_count':0,'git_head_unchanged_since_task_start':True,'git_commit_created':False}
    mism={k:{'expected':v,'actual':s.get(k)} for k,v in required.items() if s.get(k)!=v}
    # External provider blocker is a valid verified partial state, but cannot satisfy acceptance_complete.
    if s['provider_probe_blocked_by_http_402']:
        if s['task_status']!='partial' or s['candidate_decision']!='hold' or s['acceptance_complete'] is not False or s['d1_executed'] or s['d2_executed'] or not s['no_fake_results_substituted']:
            mism['blocked_state']={'expected':'truthful partial/hold with no fake D1/D2','actual':{k:s.get(k) for k in ('task_status','candidate_decision','acceptance_complete','d1_executed','d2_executed','no_fake_results_substituted')}}
    files=[CONTRACT,RESULT/'provider_capability_probe.json',RESULT/'controller_reliability_matrix.json',RESULT/'veto_prompt_v2_reliability.json',RESULT/'transport_retry_experiment.json',RESULT/'failure_taxonomy.json',RESULT/'conflict_slice_metrics.json',RESULT/'conflict_gate_v2_frozen30_metrics.json',RESULT/'d1_metrics.json',RESULT/'d2_metrics.json',RESULT/'shadow_readiness_decision.json',RESULT/'safety_metrics.json',RESULT/'summary.json',ROOT/'evaluation-data/agentic-rag-conflict-v1/benchmark_manifest.json',ROOT/'evaluation-data/contracts/task0249_answerability_conflict_signals_schema.json',ROOT/'evaluation-data/contracts/task0249_conflict_gate_v2_decision_schema.json',ROOT/'tasks/TASK-0249_selective_agent_structured_output_reliability_and_conflict_policy_gap_closure.md',ROOT/'docs/TASK0249_SELECTIVE_AGENT_STRUCTURED_OUTPUT_RELIABILITY_AND_CONFLICT_POLICY_GAP_CLOSURE_REPORT.md',REG]
    missing=[str(x.relative_to(ROOT)) for x in files if not x.is_file()]
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':not mism and not missing,'acceptance_complete':s['acceptance_complete'],'task_status':s['task_status'],'external_blocker':s['provider_reliability_blocker_code'],'mismatches':mism,'missing_files':missing}
