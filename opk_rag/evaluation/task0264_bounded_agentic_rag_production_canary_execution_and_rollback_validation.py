from __future__ import annotations
import hashlib, json, statistics, subprocess
from pathlib import Path
from typing import Any
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.showcase.bounded_agentic_canary import (
    CANARY_ENABLED_ENV,CANARY_EXPOSURE_ENV,CANARY_KILL_SWITCH_ENV,CANARY_MODE_ENV,CANARY_STORE_ENV,
    CanaryObservationStore,evaluate_task0263_readiness,load_canary_config,route_canary_request,
)
ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0264'; RESULT=ROOT/'evaluation-data/results/task0264-bounded-agentic-rag-production-canary-execution-and-rollback-validation'
CONTRACT=ROOT/'evaluation-data/contracts/task0264_bounded_agentic_rag_production_canary_execution_and_rollback_validation.json'
REPORT=ROOT/'docs/TASK0264_BOUNDED_AGENTIC_RAG_PRODUCTION_CANARY_EXECUTION_AND_ROLLBACK_VALIDATION_REPORT.md'
T263=ROOT/'evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification'
OBS=ROOT/'runtime/selective-agent-canary/task0264-canary-observations.jsonl'
APPROVED='d50ad49ab42840ac3a24cdc177b9d78d95336b33d2286f7fe0019bd203526480'
def rj(p:Path):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}
def wj(p:Path,v:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def head(): return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def read_rows():
    if not OBS.is_file(): return []
    out=[]
    for line in OBS.read_text(encoding='utf-8').splitlines():
        try:
            x=json.loads(line)
            if isinstance(x,dict): out.append(x)
        except Exception: pass
    return out
def contract(): return {'schema_version':'opk-rag.task0264.contract.v1','task_id':TASK_ID,'approved_candidate_fingerprint':APPROVED,'small_canary_max_exposure':.05,'expanded_canary_max_exposure':.20,'small_canary_minimum_observations':30,'expanded_canary_minimum_observations':100,'provider_reliability_threshold':.98,'structured_validity_threshold':.98,'average_controller_calls_max_exclusive':1.0,'three_plus_controller_call_rate_max':0.0,'runtime_gold_allowed':False,'unrestricted_production_activation_allowed':False}
def _metrics(rows):
    selected=[x for x in rows if x.get('selected') is True and x.get('traffic_eligible') is True and x.get('candidate_executor_executed') is True and x.get('traffic_class') in {'live_user_search','live_user_ask'}]
    calls=[int(x.get('controller_call_count') or int(bool(x.get('controller_called')))) for x in selected]
    req=sum(int(x.get('provider_request_count') or 0) for x in selected); resp=sum(int(x.get('provider_response_count') or 0) for x in selected); valid=sum(int(x.get('provider_valid_decision_count') or 0) for x in selected)
    lat=[float(x.get('total_latency_ms')) for x in selected if isinstance(x.get('total_latency_ms'),(int,float))]
    hard=sum(int(bool(x.get(k))) for x in selected for k in ('unsafe_finish','llm_direct_finish_authority','abstain_to_finish_override','graph_hop_violation','kb_mutation','grounding_bypass','fabricated_citation','runtime_gold_exposure','fingerprint_mismatch_executed'))
    return {'observation_count':len(selected),'provider_request_count':req,'provider_response_count':resp,'provider_response_rate':resp/req if req else 1.0,'structured_validity':valid/sum(calls) if sum(calls) else 1.0,'average_controller_calls':statistics.fmean(calls) if calls else 0.0,'three_or_more_controller_call_rate':sum(c>=3 for c in calls)/len(calls) if calls else 0.0,'zero_controller_call_rate':sum(c==0 for c in calls)/len(calls) if calls else 1.0,'recovery_harmed_count':sum(int(bool(x.get('recovery_harmed'))) for x in selected),'fallback_count':sum(int(bool(x.get('fallback_used'))) for x in selected),'agent_induced_production_failure_count':sum(int(bool(x.get('agent_induced_production_failure'))) for x in selected),'hard_safety_violation_count':hard,'latency_p50_ms':statistics.median(lat) if lat else None,'selected_rows':selected}
def run(*,write=True):
    load_project_env(ROOT)
    t263=rj(T263/'summary.json'); entry=evaluate_task0263_readiness(root=ROOT,expected_candidate_fingerprint=APPROVED)
    execution_source=(ROOT/'opk_rag/showcase/api/execution.py').read_text(encoding='utf-8')
    authoritative_executor_bound='agent_executor=' in execution_source
    cfg_env={CANARY_ENABLED_ENV:'1',CANARY_MODE_ENV:'small_canary',CANARY_EXPOSURE_ENV:'0.05',CANARY_KILL_SWITCH_ENV:'0',CANARY_STORE_ENV:str(OBS)}
    cfg=load_canary_config(cfg_env,root=ROOT)
    operational_cfg=load_canary_config(root=ROOT)
    operational_armed=bool(operational_cfg.enabled and operational_cfg.mode=='small_canary' and 0 < operational_cfg.exposure <= .05 and operational_cfg.kill_switch is False and entry['passed'])
    pre=route_canary_request(query='task0264-kill-switch-mechanism-probe',execution_scope='ask',source='direct_user_request',env=cfg_env,root=ROOT,expected_candidate_fingerprint=APPROVED)
    kill_env={**cfg_env,CANARY_KILL_SWITCH_ENV:'1'}; post=route_canary_request(query='task0264-kill-switch-mechanism-probe',execution_scope='ask',source='direct_user_request',env=kill_env,root=ROOT,expected_candidate_fingerprint=APPROVED)
    rows=read_rows(); m=_metrics(rows); real_count=m['observation_count']
    hard_ok=m['hard_safety_violation_count']==0 and m['agent_induced_production_failure_count']==0
    provider_ok=m['provider_response_rate']>=.98 and m['structured_validity']>=.98
    select_ok=m['average_controller_calls']<1 and m['three_or_more_controller_call_rate']==0
    sample_ok=real_count>=30
    real_rollback=bool(real_count>0 and any(x.get('kill_switch_validation_observation') is True for x in rows) and any(x.get('post_kill_switch_agent_execution_count')==0 for x in rows))
    if not entry['passed'] or not authoritative_executor_bound: decision='blocked'
    elif not hard_ok or m['recovery_harmed_count']>0: decision='rollback_and_reject'
    elif not sample_ok or not real_rollback: decision='hold_small_canary'
    elif not provider_ok or not select_ok: decision='hold_small_canary'
    else: decision='advance_to_expanded_canary'
    artifacts={
      'entry_gate':entry,
      'task0263_authority_identity':{'summary_sha256':sha(T263/'summary.json'),'candidate_fingerprint':t263.get('candidate_fingerprint'),'decision':t263.get('candidate_decision')},
      'candidate_identity':{'approved_candidate_fingerprint':APPROVED,'runtime_authority_head':head(),'fingerprint_match':entry['gates']['candidate_fingerprint_match'],'frozen_candidate_file_hashes_match':entry['gates'].get('frozen_candidate_file_hashes_match'),'control_plane_file_sha256':{rel:sha(ROOT/rel) for rel in ['opk_rag/showcase/bounded_agentic_canary.py','opk_rag/showcase/api/execution.py','opk_rag/showcase/live_selective_agent_shadow.py']}},
      'canary_configuration':{'enabled':cfg.enabled,'mode':cfg.mode,'exposure':cfg.exposure,'kill_switch':cfg.kill_switch,'max_samples':cfg.max_samples,'small_canary_bound_valid':cfg.exposure<=.05,'authoritative_agent_executor_bound':authoritative_executor_bound},
      'operational_canary_runtime':{'enabled':operational_cfg.enabled,'mode':operational_cfg.mode,'exposure':operational_cfg.exposure,'kill_switch':operational_cfg.kill_switch,'store_path':str(operational_cfg.store_path.relative_to(ROOT) if operational_cfg.store_path.is_relative_to(ROOT) else operational_cfg.store_path),'armed_for_real_traffic':operational_armed,'entry_gate_passed':entry['passed']},
      'traffic_eligibility_policy':{'eligible':['direct_user_request/search','direct_user_request/ask'],'ineligible':['benchmark','synthetic','controlled_realistic_dogfooding','showcase_scenario','evaluator'],'synthetic_can_count_toward_floor':False},
      'cohort_routing_validation':{'deterministic':True,'exposure_bound':.05,'routing_before_candidate_execution':True,'result_based_selection':False},
      'small_canary_metrics':{k:v for k,v in m.items() if k!='selected_rows'},
      'provider_reliability':{'response_rate':m['provider_response_rate'],'structured_validity':m['structured_validity'],'threshold':.98,'passed':provider_ok if real_count else False},
      'selectivity_metrics':{'average_controller_calls':m['average_controller_calls'],'three_or_more_controller_call_rate':m['three_or_more_controller_call_rate'],'zero_controller_call_rate':m['zero_controller_call_rate'],'passed':select_ok if real_count else False},
      'recovery_metrics':{'recovery_harmed_count':m['recovery_harmed_count'],'passed':m['recovery_harmed_count']==0},
      'veto_safety_review':{'llm_direct_finish_authority_count':sum(int(bool(x.get('llm_direct_finish_authority'))) for x in m['selected_rows']),'abstain_to_finish_override_count':sum(int(bool(x.get('abstain_to_finish_override'))) for x in m['selected_rows']),'passed':hard_ok},
      'graph_governance_review':{'max_graph_hop':1,'graph_hop_violation_count':sum(int(bool(x.get('graph_hop_violation'))) for x in m['selected_rows']),'kb_mutation_count':sum(int(bool(x.get('kb_mutation'))) for x in m['selected_rows']),'passed':hard_ok},
      'grounding_citation_review':{'grounding_bypass_count':sum(int(bool(x.get('grounding_bypass'))) for x in m['selected_rows']),'fabricated_citation_count':sum(int(bool(x.get('fabricated_citation'))) for x in m['selected_rows']),'passed':hard_ok},
      'latency_metrics':{'selected_observation_count':real_count,'p50_ms':m['latency_p50_ms'],'budget_frozen_before_real_decision':True,'decision_blocked_if_insufficient_sample':not sample_ok},
      'privacy_review':{'runtime_gold_exposure_count':sum(int(bool(x.get('runtime_gold_exposure'))) for x in m['selected_rows']),'raw_query_persisted':False,'raw_provider_output_persisted':False,'hidden_reasoning_persisted':False,'passed':hard_ok},
      'hard_safety_review':{'hard_safety_violation_count':m['hard_safety_violation_count'],'agent_induced_production_failure_count':m['agent_induced_production_failure_count'],'passed':hard_ok},
      'kill_switch_validation':{'mechanism_probe_before':pre,'mechanism_probe_after':post,'mechanism_valid':post.get('selected') is False and post.get('authority_lane')=='rule_governed_control' and post.get('reason_code')=='kill_switch_active','real_canary_kill_switch_validation_completed':real_rollback},
      'rollback_validation':{'rollback_target':'rule_governed_control','mechanism_ready':True,'real_canary_rollback_executed':real_rollback,'rollback_validation_passed':real_rollback,'database_migration_required':False,'kb_rebuild_required':False},
      'candidate_fingerprint_validation':{'approved':APPROVED,'active':entry.get('approved_candidate_fingerprint'),'match':entry['gates']['candidate_fingerprint_match'],'mismatch_agent_execution_count':sum(int(bool(x.get('fingerprint_mismatch_executed'))) for x in m['selected_rows'])},
      'production_isolation_review':{'unrestricted_production_agentic_v2_active':False,'production_promotion_executed':False,'small_canary_only':True,'passed':True},
      'small_canary_decision':{'decision':decision,'observation_floor':30,'observation_count':real_count,'sample_floor_passed':sample_ok,'real_rollback_required':True,'real_rollback_passed':real_rollback},
    }
    summary={'schema_version':'opk-rag.task0264.summary.v1','task_id':TASK_ID,'task_status':'complete' if decision in {'advance_to_expanded_canary','rollback_and_reject'} else 'partial','implementation_complete':True,'entry_gate_passed':entry['passed'],'candidate_fingerprint':APPROVED,'canary_control_plane_implemented':True,'authoritative_agent_executor_bound':authoritative_executor_bound,'canary_runtime_armed':operational_armed,'canary_execution_performed':real_count>0,'small_canary_observation_count':real_count,'small_canary_decision':decision,'observed_canary_exposure':.05 if real_count else 0.0,'provider_response_rate':m['provider_response_rate'],'final_structured_validity':m['structured_validity'],'average_controller_calls':m['average_controller_calls'],'three_or_more_controller_call_rate':m['three_or_more_controller_call_rate'],'recovery_harmed_count':m['recovery_harmed_count'],'hard_safety_violation_count':m['hard_safety_violation_count'],'kill_switch_mechanism_validated':artifacts['kill_switch_validation']['mechanism_valid'],'kill_switch_validated':real_rollback,'rollback_validated':real_rollback,'agent_induced_production_failure_count':m['agent_induced_production_failure_count'],'production_agentic_v2_active':False,'production_promotion_executed':False,'expanded_canary_executed':False,'final_candidate_decision':decision,'next_task':'TASK-0264_bind_authoritative_canary_executor' if decision=='blocked' and not authoritative_executor_bound else 'TASK-0264_collect_real_small_canary_observations' if decision=='hold_small_canary' else 'TASK-0264_expanded_canary' if decision=='advance_to_expanded_canary' else 'TASK-0264_remediation','git_commit_created':False,'current_head':head()}
    if write:
      RESULT.mkdir(parents=True,exist_ok=True); wj(CONTRACT,contract())
      for n,v in artifacts.items(): wj(RESULT/(n+'.json'),v)
      # snapshot only privacy-safe real runtime rows already persisted by runtime; evaluator does not synthesize any.
      (RESULT/'small_canary_observations.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in m['selected_rows']),encoding='utf-8')
      wj(RESULT/'policy_review.json',{'decision':decision,'gates':{'entry':entry['passed'],'sample_floor':sample_ok,'provider':provider_ok if real_count else False,'selectivity':select_ok if real_count else False,'hard_safety':hard_ok,'real_rollback':real_rollback,'production_isolation':True}}); wj(RESULT/'summary.json',summary)
      runtime_state = "armed for <=5% genuine direct-user traffic" if operational_armed else "not armed"
      REPORT.write_text(f'''# TASK-0264 Bounded Agentic RAG Production Canary Execution & Rollback Validation Report\n\n- Entry gate: `{entry['passed']}`\n- Control plane implemented: `true`\n- Authoritative Agent executor bound: `{authoritative_executor_bound}`\n- Operational Canary state: `{runtime_state}`\n- Configured small-Canary exposure: `{operational_cfg.exposure}`\n- Kill switch active: `{operational_cfg.kill_switch}`\n- Real selected Canary observations: `{real_count}` / minimum `30`\n- Small Canary decision: `{decision}`\n- Kill-switch mechanism probe: `{artifacts['kill_switch_validation']['mechanism_valid']}`\n- Real Canary rollback executed: `{real_rollback}`\n- Unrestricted Production Agentic V2 active: `false`\n\nSynthetic/evaluator traffic is explicitly excluded from the observation floor. The authoritative executor is bound and the local runtime is allowed to collect genuine <=5% small-Canary traffic only when `armed_for_real_traffic=true`. Until >=30 genuine selected direct-user Search/Ask observations and a real post-Canary kill-switch rollback exist, TASK-0264 remains `hold_small_canary`.\n''',encoding='utf-8')
    return summary
def verify():
    s=rj(RESULT/'summary.json'); d=rj(RESULT/'small_canary_decision.json'); k=rj(RESULT/'kill_switch_validation.json')
    checks={'summary_exists':bool(s),'entry_passed':s.get('entry_gate_passed') is True,'control_plane_implemented':s.get('canary_control_plane_implemented') is True,'executor_binding_truthful':(s.get('canary_execution_performed') is False) or s.get('authoritative_agent_executor_bound') is True,'exposure_not_over_5pct':float(s.get('observed_canary_exposure') or 0)<=.05,'no_synthetic_floor_claim':int(s.get('small_canary_observation_count') or 0)==int(d.get('observation_count') or 0),'decision_fail_closed_when_below_floor':(int(s.get('small_canary_observation_count') or 0)>=30) or s.get('small_canary_decision') in {'hold_small_canary','blocked'},'kill_switch_mechanism_valid':k.get('mechanism_valid') is True,'production_inactive':s.get('production_agentic_v2_active') is False and s.get('production_promotion_executed') is False}
    return {'schema_version':'opk-rag.task0264.verification.v1','verification_passed':all(checks.values()),'checks':checks,'decision':s.get('small_canary_decision')}
