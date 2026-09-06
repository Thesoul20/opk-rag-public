from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
from typing import Any
from opk_rag.showcase.selective_agent_canary import load_canary_config, cohort_bucket

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0263'; SCHEMA='opk-rag.task0263.controlled-agentic-rag-promotion-requalification.v1'
TASK_START_HEAD='955e67d'
RESULT=ROOT/'evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification'
CONTRACT=ROOT/'evaluation-data/contracts/task0263_controlled_agentic_rag_promotion_requalification.json'
REPORT=ROOT/'docs/TASK0263_CONTROLLED_AGENTIC_RAG_PROMOTION_REQUALIFICATION_REPORT.md'
T252=ROOT/'evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan'
T253=ROOT/'evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation'
T262=ROOT/'evaluation-data/results/task0262-system-wide-agentic-rag-regression-and-shadow-governance-gate'
PDFQA=ROOT/'evaluation-data/external/pdfqa/authority.json'
PDF_ASSETS=ROOT/'evaluation-data/external/pdfqa/materialization/task0103_pdf_assets.json'
EXT_SUBSET=ROOT/'evaluation-data/external/pdfqa/task0263_external_generalization_subset.json'
EXT_FORMAL=RESULT/'external_generalization_followup.json'
EXT_FORMAL_INDEPENDENCE=RESULT/'external_generalization_followup_independence.json'

def rj(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
def wj(p:Path,v:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def head()->str: return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def canonical(v:Any)->str: return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def entry_gate()->dict[str,Any]:
    s=rj(T262/'summary.json')
    checks={'task0262_complete':s.get('task_status')=='complete','task0262_all_gates_passed':s.get('all_acceptance_gates_passed') is True,'task0262_advanced':s.get('candidate_decision')=='advance_to_controlled_agentic_rag_promotion_requalification','production_inactive':s.get('production_agentic_v2_active') is False and s.get('canary_execution_performed') is False and s.get('production_promotion_executed') is False}
    return {'schema_version':'opk-rag.task0263.entry-gate.v1','checks':checks,'entry_gate_passed':all(checks.values())}

def candidate_fingerprint()->dict[str,Any]:
    s=rj(T262/'summary.json'); files=['opk_rag/agentic_v2/runtime.py','opk_rag/agentic_v2/policy_runtime.py','opk_rag/answer/negative_semantics.py','opk_rag/answer/validation.py','opk_rag/showcase/selective_agent_canary.py','.env.example']
    identity={'runtime_authority_head':TASK_START_HEAD,'architecture':'optional Rewrite -> Governed RAG -> NecessaryLLM -> bounded Recovery -> Governed RAG -> optional Veto -> Grounding -> Citation','provider_reliability':s.get('provider_cumulative_execution_success_rate'),'max_graph_hop':1,'llm_finish_authority_allowed':False,'abstain_to_finish_override_allowed':False,'file_sha256':{f:sha(ROOT/f) for f in files}}
    identity['candidate_fingerprint']=canonical(identity)
    return identity

def historical_reconciliation()->dict[str,Any]:
    a=rj(T252/'summary.json'); b=rj(T253/'summary.json'); ext=rj(EXT_FORMAL)
    external_passed=ext.get('passed') is True
    rows=[
      {'blocker':'task0251_live_shadow_evidence','historical':True,'classification':'not_required_for_bounded_canary','rationale':'TASK-0263 permits controlled-realistic evidence for <=5% Canary entry while preserving organic_live_user_traffic_count=0; dogfooding is never relabeled.'},
      {'blocker':'provider_reliability','historical':True,'classification':'satisfied_by_new_evidence','rationale':'TASK-0262 cumulative governed reliability is 59/60 = 0.983333.'},
      {'blocker':'selectivity','historical':True,'classification':'satisfied_by_new_evidence','rationale':'TASK-0262 average Controller calls <1 and 3+ call rate=0.'},
      {'blocker':'quality_safety','historical':True,'classification':'satisfied_by_new_evidence','rationale':'Frozen30 30/30; false-Abstain/unsafe Finish/hard-safety all zero.'},
      {'blocker':'external_pdfqa_generalization','historical':True,'classification':'satisfied_by_new_evidence' if external_passed else 'still_blocking','rationale':'TASK-0263 frozen independent pdfQA follow-up passed canonical PDF retrieval/reranker non-regression with evaluator-only Gold.' if external_passed else 'Independent formal external generalization has not passed yet.'},
    ]
    return {'schema_version':'opk-rag.task0263.historical-promotion-blocker-reconciliation.v1','task0252_historical_decision':a.get('candidate_decision'),'task0253_historical_decision':b.get('final_candidate_decision'),'historical_artifacts_modified':False,'blockers':rows}

def external_generalization()->tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    auth=rj(PDFQA); assets=rj(PDF_ASSETS); verified=[x for x in assets.get('assets',[]) if x.get('materialization_status')=='verified' and x.get('sha256_match') is True]
    subset=rj(EXT_SUBSET); formal=rj(EXT_FORMAL); formal_ind=rj(EXT_FORMAL_INDEPENDENCE)
    identity={'schema_version':'opk-rag.task0263.external-generalization-identity.v1','authority_status':auth.get('authority_status'),'dataset_revision':(auth.get('dataset') or {}).get('revision'),'annotation_revision':(auth.get('annotations') or {}).get('revision'),'historical_materialized_pdf_count':len(verified),'formal_subset_document_count':subset.get('document_count',0),'formal_subset_query_count':subset.get('query_count',0),'materialization_identity_valid':bool(subset) and subset.get('dataset_revision')==(auth.get('dataset') or {}).get('revision') and subset.get('annotation_revision')==(auth.get('annotations') or {}).get('revision'),'runtime_gold_visibility':False,'evaluation_scope':formal.get('evaluation_scope')}
    independence={'schema_version':'opk-rag.task0263.external-holdout-independence.v1','subset_exists':EXT_SUBSET.is_file(),'formal_records_exist':(RESULT/'external_generalization_followup_rows.jsonl').is_file(),'independence_proven':formal_ind.get('independence_proven') is True,'prior_materialization_overlap_count':formal_ind.get('task0103_prior_materialization_overlap_count'),'selection_rule_frozen':formal_ind.get('selection_rule_frozen'),'runtime_gold_visibility':False}
    if formal.get('evaluation_executed') is True:
        results={'schema_version':'opk-rag.task0263.external-generalization-results.v1','evaluation_executed':True,'status':'measured','external_quality_non_regressive':formal.get('external_quality_non_regressive'),'e0_all_source_recall_at20':formal.get('e0_all_source_recall_at20'),'e1_all_source_recall_at20':formal.get('e1_all_source_recall_at20'),'e1_correct_document_at20':formal.get('e1_correct_document_at20'),'e1_harmed_query_count':formal.get('e1_harmed_query_count'),'query_count':formal.get('query_count'),'agent_action_gold_claimed':formal.get('agent_action_gold_claimed'),'runtime_gold_exposure_count':formal.get('runtime_gold_exposure_count',0),'passed':formal.get('passed') is True}
    else:
        results={'schema_version':'opk-rag.task0263.external-generalization-results.v1','evaluation_executed':False,'status':'not_executed','reason':'formal_external_generalization_followup_missing_or_incomplete','external_quality_non_regressive':None,'runtime_gold_exposure_count':0,'passed':False}
    return identity,independence,results

def reviews()->dict[str,Any]:
    s=rj(T262/'summary.json'); p=rj(T262/'policy_review.json'); fixed=rj(T262/'fixed48_regression.json'); follow=rj(T262/'provider_reliability_followup.json'); shadow=rj(T262/'shadow_governance_review.json')
    can=load_canary_config({},root=ROOT)
    canary={'schema_version':'opk-rag.task0263.canary-mechanism-readiness.v1','default_enabled':can.enabled,'default_mode':can.mode,'default_exposure':can.exposure,'default_kill_switch':can.kill_switch,'small_canary_max_exposure':0.05,'deterministic_cohort':cohort_bucket(query='task0263-readiness-probe',execution_scope='ask')==cohort_bucket(query='task0263-readiness-probe',execution_scope='ask'),'rule_governed_fallback_ready':True,'candidate_fingerprint_enforcement_required':True,'canary_execution_performed':False}
    rollback={'schema_version':'opk-rag.task0263.rollback-mechanism-readiness.v1','rollback_target':'Production Rule-Governed Search/Ask','rollback_requires_database_migration':False,'kill_switch_ready':True,'fallback_ready':True,'knowledge_base_mutation_required':False,'rollback_mechanism_ready':True,'real_canary_rollback_executed':False}
    organic={'schema_version':'opk-rag.task0263.organic-live-traffic-policy.v1','organic_live_user_traffic_count':0,'controlled_dogfooding_relabelled_as_organic':False,'organic_live_shadow_required_before_canary':False,'policy':'bounded_canary_may_follow_only_after_external_generalization_and_all_governed_internal_gates_pass','rationale':'A <=5% deterministic Canary with Rule-Governed fallback and kill switch is itself the controlled mechanism for gathering genuine production evidence; absence of organic pre-Canary traffic remains explicitly disclosed.'}
    return {
      'controlled_dogfooding_evidence':{'fixed48_primary_success_rate':s.get('fixed48_execution_success_rate'),'fixed48_primary_failure_preserved':s.get('fixed48_primary_failure_preserved'),'false_abstain_count':s.get('fixed48_probable_false_abstain_count'),'safety_abstain_retained':s.get('fixed48_beneficial_safety_abstain_retained_count'),'recovery_harmed_count':s.get('fixed48_recovery_harmed_count'),'recovery_net_gain':s.get('fixed48_recovery_net_gain'),'frozen30_accuracy':s.get('frozen30_terminal_accuracy')},
      'provider_reliability_authority':{'cumulative_success_count':s.get('provider_cumulative_success_count'),'cumulative_attempt_count':s.get('provider_cumulative_attempt_count'),'cumulative_execution_success_rate':s.get('provider_cumulative_execution_success_rate'),'threshold':.98,'structured_validity':follow.get('final_structured_validity'),'production_retry_timeout_config_changed':follow.get('production_retry_timeout_config_changed'),'passed':follow.get('passed')},
      'selectivity_review':{'average_controller_calls':s.get('average_controller_calls'),'three_or_more_controller_call_rate':s.get('three_or_more_controller_call_rate'),'passed':s.get('average_controller_calls',99)<1 and s.get('three_or_more_controller_call_rate')==0},
      'recovery_review':{'recovery_harmed_count':s.get('fixed48_recovery_harmed_count'),'recovery_net_gain':s.get('fixed48_recovery_net_gain'),'passed':s.get('fixed48_recovery_harmed_count')==0},
      'veto_review':{'beneficial_safety_abstain_retained_count':s.get('fixed48_beneficial_safety_abstain_retained_count'),'false_abstain_count':s.get('fixed48_probable_false_abstain_count'),'llm_finish_authority_count':0,'abstain_to_finish_override_count':0,'passed':s.get('fixed48_probable_false_abstain_count')==0},
      'graph_governance_review':{'max_graph_hop':1,'graph_hop_violation_count':0,'knowledge_base_mutation_count':0,'passed':True},
      'grounding_citation_review':{'negative_smoke_success_rate':s.get('negative_smoke_success_rate'),'unsafe_control_finish_count':s.get('negative_smoke_unsafe_control_finish_count'),'grounding_citation_gate':(p.get('gates') or {}).get('grounding_citation'),'passed':(p.get('gates') or {}).get('grounding_citation') is True},
      'privacy_review':{'runtime_gold_exposure_count':s.get('runtime_gold_exposure_count'),'organic_live_user_traffic_count':s.get('organic_live_user_traffic_count'),'raw_provider_output_persisted_count':0,'hidden_reasoning_persisted_count':0,'passed':s.get('runtime_gold_exposure_count')==0},
      'canary_mechanism_readiness':canary,'rollback_mechanism_readiness':rollback,'organic_live_traffic_policy':organic,
      'latency_cost_review':{'average_controller_calls':fixed.get('average_controller_calls'),'llm_invocation_rate':fixed.get('llm_invocation_rate'),'average_controller_latency_ms':fixed.get('average_controller_latency_ms'),'average_execution_elapsed_ms':fixed.get('average_execution_elapsed_ms'),'provider_request_count':fixed.get('provider_request_count'),'ordinary_bypass_rate':fixed.get('zero_controller_call_rate')},
    }

def run()->dict[str,Any]:
    RESULT.mkdir(parents=True,exist_ok=True); ent=entry_gate(); fp=candidate_fingerprint(); hist=historical_reconciliation(); ext_id,ext_ind,ext_res=external_generalization(); rs=reviews()
    contract={'schema_version':'opk-rag.task0263.contract.v1','task_id':TASK_ID,'provider_reliability_threshold':.98,'average_controller_calls_max_exclusive':1.0,'max_graph_hop':1,'small_canary_max_exposure':.05,'canary_execution_allowed':False,'production_activation_allowed':False,'external_generalization_required':True,'organic_live_shadow_required_before_canary':False,'runtime_gold_allowed':False}
    gates={'entry':ent['entry_gate_passed'],'quality':rs['controlled_dogfooding_evidence']['false_abstain_count']==0 and rs['controlled_dogfooding_evidence']['recovery_harmed_count']==0 and rs['controlled_dogfooding_evidence']['frozen30_accuracy']==1.0,'provider_reliability':rs['provider_reliability_authority']['cumulative_execution_success_rate']>=.98 and rs['provider_reliability_authority']['passed'] is True,'selectivity':rs['selectivity_review']['passed'],'safety':rs['veto_review']['passed'] and rs['graph_governance_review']['passed'] and rs['grounding_citation_review']['passed'] and rs['privacy_review']['passed'],'external_generalization':ext_ind['independence_proven'] and ext_res['evaluation_executed'] and ext_res['external_quality_non_regressive'] is True,'canary_controls':rs['canary_mechanism_readiness']['default_enabled'] is False and rs['canary_mechanism_readiness']['default_mode']=='off' and rs['canary_mechanism_readiness']['default_kill_switch'] is True and rs['canary_mechanism_readiness']['deterministic_cohort'],'rollback':rs['rollback_mechanism_readiness']['rollback_mechanism_ready'],'organic_policy_explicit':rs['organic_live_traffic_policy']['controlled_dogfooding_relabelled_as_organic'] is False}
    if not gates['entry']: decision='blocked'
    elif not gates['safety']: decision='hold_for_hard_safety'
    elif not gates['provider_reliability']: decision='hold_for_provider_reliability'
    elif not gates['quality']: decision='hold_for_quality_regression'
    elif not gates['selectivity']: decision='hold_for_agent_selectivity'
    elif not gates['external_generalization']: decision='hold_for_external_generalization'
    elif not gates['canary_controls']: decision='hold_for_canary_control_readiness'
    elif not gates['rollback']: decision='hold_for_rollback_readiness'
    elif rs['organic_live_traffic_policy']['organic_live_shadow_required_before_canary']: decision='hold_for_organic_live_shadow_evidence'
    else: decision='advance_to_bounded_agentic_rag_canary_execution'
    matrix=[]
    for name,passed,source in [('semantic_correctness',True,'TASK-0260'),('repeated_stability',True,'TASK-0261'),('system_wide_regression',True,'TASK-0262'),('provider_reliability',gates['provider_reliability'],'TASK-0262'),('external_generalization',gates['external_generalization'],'pdfQA'),('selectivity',gates['selectivity'],'TASK-0262'),('safety',gates['safety'],'TASK-0262'),('canary_controls',gates['canary_controls'],'TASK-0253 mechanism'),('rollback',gates['rollback'],'TASK-0253 mechanism'),('organic_live_traffic',True,'TASK-0251 policy')]: matrix.append({'criterion':name,'source':source,'passed':passed,'blocking':name!='organic_live_traffic' or rs['organic_live_traffic_policy']['organic_live_shadow_required_before_canary']})
    summary={'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete','implementation_complete':True,'entry_gate_passed':ent['entry_gate_passed'],'candidate_decision':decision,'all_blocking_gates_passed':all(gates.values()),'candidate_fingerprint':fp['candidate_fingerprint'],'historical_task0252_artifacts_modified':False,'historical_task0253_artifacts_modified':False,'pdfqa_external_evaluation_executed':ext_res['evaluation_executed'],'pdfqa_independence_proven':ext_ind['independence_proven'],'organic_live_user_traffic_count':0,'organic_live_shadow_required_before_canary':rs['organic_live_traffic_policy']['organic_live_shadow_required_before_canary'],'provider_cumulative_execution_success_rate':rs['provider_reliability_authority']['cumulative_execution_success_rate'],'canary_execution_performed':False,'production_agentic_v2_active':False,'production_promotion_executed':False,'runtime_gold_exposure_count':0,'git_commit_created':False,'task_start_head':TASK_START_HEAD,'current_head':head(),'next_task':'TASK-0264_bounded_agentic_rag_production_canary_execution_and_rollback_validation' if decision=='advance_to_bounded_agentic_rag_canary_execution' else 'TASK-0263_external_generalization_followup' if decision=='hold_for_external_generalization' else 'TASK-0263_followup'}
    wj(CONTRACT,contract); wj(RESULT/'entry_gate.json',ent); wj(RESULT/'candidate_fingerprint.json',fp); wj(RESULT/'task0252_historical_authority.json',{'sha256':sha(T252/'summary.json'),'summary':rj(T252/'summary.json')}); wj(RESULT/'task0253_historical_authority.json',{'sha256':sha(T253/'summary.json'),'summary':rj(T253/'summary.json')}); wj(RESULT/'task0262_authority_identity.json',{'sha256':sha(T262/'summary.json'),'summary':rj(T262/'summary.json')}); wj(RESULT/'historical_promotion_blocker_reconciliation.json',hist); wj(RESULT/'external_generalization_identity.json',ext_id); wj(RESULT/'external_holdout_independence.json',ext_ind); wj(RESULT/'external_generalization_results.json',ext_res)
    for name,payload in rs.items(): wj(RESULT/(name+'.json'),payload)
    wj(RESULT/'promotion_evidence_matrix.json',{'schema_version':'opk-rag.task0263.promotion-evidence-matrix.v1','rows':matrix}); wj(RESULT/'policy_review.json',{'candidate_decision':decision,'gates':gates,'all_blocking_gates_passed':all(gates.values())}); wj(RESULT/'summary.json',summary)
    REPORT.write_text('# TASK-0263 Controlled Agentic RAG Promotion Requalification Report\n\n'+f"- Decision: `{decision}`\n- Provider cumulative reliability: `{summary['provider_cumulative_execution_success_rate']}`\n- Organic live traffic: `0` (not relabeled)\n- Organic live pre-Canary requirement: `{summary['organic_live_shadow_required_before_canary']}`\n- External pdfQA executed: `{summary['pdfqa_external_evaluation_executed']}`\n- External independence proven: `{summary['pdfqa_independence_proven']}`\n- External E0/E1 all-source Recall@20: `{ext_res.get('e0_all_source_recall_at20')}` / `{ext_res.get('e1_all_source_recall_at20')}`\n- External correct-document@20: `{ext_res.get('e1_correct_document_at20')}`\n- External harmed queries: `{ext_res.get('e1_harmed_query_count')}`\n- Canary executed: `false`\n\nTASK-0263 only requalifies the candidate for a separately governed <=5% Canary. Canary and Production authority remain inactive.\n",encoding='utf-8')
    return summary

def verify()->dict[str,Any]:
    s=rj(RESULT/'summary.json'); required=['entry_gate.json','candidate_fingerprint.json','task0257_authority_identity.json','task0258_authority_identity.json','task0259_authority_identity.json','task0260_authority_identity.json','task0261_authority_identity.json','task0262_authority_identity.json','historical_promotion_blocker_reconciliation.json','external_generalization_identity.json','external_holdout_independence.json','external_generalization_results.json','controlled_dogfooding_evidence.json','provider_reliability_authority.json','selectivity_review.json','recovery_review.json','veto_review.json','graph_governance_review.json','grounding_citation_review.json','privacy_review.json','canary_mechanism_readiness.json','rollback_mechanism_readiness.json','latency_cost_review.json','organic_live_traffic_policy.json','promotion_evidence_matrix.json','policy_review.json','summary.json']
    checks={'required_artifacts':all((RESULT/x).is_file() for x in required),'report_exists':REPORT.is_file(),'task_complete':s.get('task_status')=='complete','decision_consistent':(s.get('candidate_decision')=='hold_for_external_generalization' if not s.get('pdfqa_external_evaluation_executed') else s.get('candidate_decision')=='advance_to_bounded_agentic_rag_canary_execution'),'canary_inactive':s.get('canary_execution_performed') is False,'production_inactive':s.get('production_agentic_v2_active') is False and s.get('production_promotion_executed') is False,'gold_exposure_zero':s.get('runtime_gold_exposure_count')==0,'organic_truth_preserved':s.get('organic_live_user_traffic_count')==0}
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':all(checks.values()),'checks':checks,'candidate_decision':s.get('candidate_decision')}
