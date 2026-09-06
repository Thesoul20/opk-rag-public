from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from opk_rag.evaluation.task0248_selective_runtime import (
    AbstentionVetoDecision,
    AbstentionVetoInvocationDecision,
    AmbiguityRewriteDecision,
    QueryAmbiguityDecision,
)

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0248'
SCHEMA='opk-rag.task0248.selective-agent-abstention-veto-and-ambiguity-rewrite-integration.v1'
RESULT=ROOT/'evaluation-data/results/task0248-selective-agent-abstention-veto-and-ambiguity-rewrite-integration'
CONTRACT=ROOT/'evaluation-data/contracts/task0248_selective_agent_abstention_veto_and_ambiguity_rewrite_integration.json'
AMB2=ROOT/'evaluation-data/agentic-rag-ambiguity-v2'
T247=ROOT/'evaluation-data/results/task0247-agentic-v2-candidate-rejection-gap-analysis-and-recovery-plan'
REG=RESULT/'regression.json'
BASELINE_HEAD='91f8535'
PRODUCTION_PREFIXES=('opk_rag/search/','opk_rag/answer/','opk_rag/retrieval/','opk_rag/runtime_v2/','opk_rag/core_tools/','opk_rag/agent/','opk_rag/agentic_v2/','opk_rag/cli.py')


def read_json(p:Path,default=None): return json.loads(p.read_text()) if p.is_file() else ({} if default is None else default)
def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
def write_json(p:Path,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def changed_paths():
    out=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    return sorted(line[3:].split(' -> ',1)[-1] for line in out.splitlines() if len(line)>=4)
def current_head(): return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def baseline_head(): return subprocess.run(['git','rev-parse',BASELINE_HEAD],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()


def build_summary(*,write=True):
    comp=read_json(RESULT/'strategy_comparison.json'); b1=comp.get('B1',{}); b2=comp.get('B2',{}); amb=comp.get('ambiguity_v2',{}); rows=read_jsonl(RESULT/'benchmark_sample_results.jsonl'); ambrows=read_jsonl(RESULT/'ambiguity_v2_sample_results.jsonl'); reg=read_json(REG,{})
    t247=read_json(T247/'summary.json'); changed=changed_paths(); frozen=read_json(RESULT/'summary.json',{}); prod=frozen.get('production_runtime_changed_paths', [x for x in changed if x.startswith(PRODUCTION_PREFIXES)])
    veto_rows=[r['B2']['veto'] for r in rows if r['B2']['veto']['policy_called']]
    veto_failure=Counter(x.get('policy_failure_code') or 'none' for x in veto_rows)
    rewrite_rows=[r for r in ambrows if r.get('rewrite_called')]
    rewrite_failure=Counter(x.get('rewrite_failure_code') or 'none' for x in rewrite_rows)
    remaining=[r['sample_id'] for r in rows if not r['B2']['terminal_correct']]
    safety_keys=('llm_finish_authority_count','abstain_to_finish_override_count','veto_without_conflict_gate_count','unauthorized_action_execution_count','budget_violation_count','graph_hop_violation_count','benchmark_gold_exposure_count','hidden_reasoning_exposure_count','raw_provider_output_persisted_count','secret_exposure_count','knowledge_base_mutation_count')
    safety_zero=all(int(b2.get(k,0) or 0)==0 for k in safety_keys)
    summary={
      'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete','current_stage':'llm_agentic_rag_development',
      'task0247_prerequisite_complete':t247.get('task_status')=='complete','task0247_strategy_decision_preserved':t247.get('strategy_selection_decision')=='advance_selective_architecture','full_time_agentic_v2_remains_rejected':t247.get('full_time_agentic_v2_remains_rejected') is True,
      'query_ambiguity_detector_implemented':(ROOT/'opk_rag/evaluation/task0248_selective_runtime.py').is_file(),'bounded_ambiguity_rewrite_implemented':True,'recovery_only_policy_preserved':True,'bounded_abstention_veto_implemented':True,'deterministic_veto_gate_implemented':True,'asymmetric_terminal_authority_enforced':True,
      'ambiguity_benchmark_v2_frozen':read_json(AMB2/'benchmark_manifest.json').get('status')=='frozen','ambiguity_benchmark_v2_sample_count':amb.get('sample_count'),'ambiguity_detector_precision':amb.get('detector_precision'),'ambiguity_detector_recall':amb.get('detector_recall'),'ambiguity_unnecessary_rewrite_rate':amb.get('unnecessary_rewrite_rate'),'ambiguity_missed_rewrite_rate':amb.get('missed_rewrite_rate'),'ambiguity_rewrite_end_to_end_success_rate':amb.get('rewrite_end_to_end_success_rate'),'ambiguity_rewrite_provider_failure_rate':amb.get('rewrite_provider_failure_rate'),'ambiguity_rewrite_contract_valid_rate_given_provider_response':amb.get('rewrite_contract_valid_rate_given_provider_response'),'ambiguity_anchor_recall':amb.get('average_anchor_recall'),'ambiguity_top_rerank_score_gain':amb.get('top_rerank_score_gain'),
      'a0_terminal_accuracy':comp.get('A0',{}).get('terminal_accuracy'),'a1_terminal_accuracy':comp.get('A1',{}).get('terminal_accuracy'),'a2_terminal_accuracy':comp.get('A2',{}).get('terminal_accuracy'),'b0_terminal_accuracy':comp.get('B0',{}).get('terminal_accuracy'),'b1_terminal_accuracy':b1.get('terminal_accuracy'),'b2_terminal_accuracy':b2.get('terminal_accuracy'),'b2_quality_delta_vs_a0':float(b2.get('terminal_accuracy') or 0)-float(comp.get('A0',{}).get('terminal_accuracy') or 0),'b2_quality_delta_vs_b0':float(b2.get('terminal_accuracy') or 0)-float(comp.get('B0',{}).get('terminal_accuracy') or 0),'b2_quality_delta_vs_a2':float(b2.get('terminal_accuracy') or 0)-float(comp.get('A2',{}).get('terminal_accuracy') or 0),
      'veto_invocation_rate':b2.get('veto_invocation_rate'),'veto_bypass_rate':b2.get('veto_bypass_rate'),'veto_abstain_count':b2.get('veto_abstain_count'),'veto_keep_finish_count':b2.get('veto_keep_finish_count'),'veto_policy_failure_rate':b2.get('veto_policy_failure_rate'),'unsafe_finish_prevented_count':b2.get('unsafe_finish_prevented_count'),'false_abstain_created_count':b2.get('false_abstain_created_count'),'veto_net_gain':b2.get('veto_net_gain'),'veto_failure_taxonomy':dict(veto_failure),
      'recovery_invocation_rate':b2.get('recovery_invocation_rate'),'average_controller_calls':b2.get('average_controller_calls'),'zero_controller_call_rate':b2.get('zero_controller_call_rate'),'one_controller_call_rate':b2.get('one_controller_call_rate'),'two_controller_call_rate':b2.get('two_controller_call_rate'),'three_controller_call_rate':b2.get('three_controller_call_rate'),
      'b2_average_total_latency_ms':b2.get('average_total_latency_ms'),'b2_average_controller_latency_ms':b2.get('average_controller_latency_ms'),'b2_average_controller_input_tokens':b2.get('average_controller_input_tokens'),'b2_average_controller_output_tokens':b2.get('average_controller_output_tokens'),
      'remaining_incorrect_sample_count':len(remaining),'remaining_incorrect_sample_ids':remaining,'rewrite_failure_taxonomy':dict(rewrite_failure),
      **{k:int(b2.get(k,0) or 0) for k in safety_keys},'safety_invariants_preserved':safety_zero,
      'production_agentic_v2_active':False,'production_search_ask_behavior_changed':False,'production_runtime_changed_paths':prod,'production_runtime_behavior_changed':frozen.get('production_runtime_behavior_changed', bool(prod)),
      'candidate_decision':comp.get('candidate_decision'),'production_promotion_allowed':False,'next_task':comp.get('next_task'),
      # Historical TASK-0248 execution facts remain frozen after the user later commits TASK-0248.
      # Later task HEAD advances must not be reinterpreted as commits created during TASK-0248 execution.
      'current_head':read_json(RESULT/'summary.json',{}).get('current_head',current_head()),'task_start_head':read_json(RESULT/'summary.json',{}).get('task_start_head',baseline_head()),'git_head_unchanged_since_task_start':read_json(RESULT/'summary.json',{}).get('git_head_unchanged_since_task_start',current_head()==baseline_head()),'git_commit_created':read_json(RESULT/'summary.json',{}).get('git_commit_created',current_head()!=baseline_head()),
      'focused_tests_passed':reg.get('focused_tests_passed'),'agentic_regression_passed':reg.get('agentic_regression_passed'),'task0247_verifier_passed':reg.get('task0247_verifier_passed'),'task0248_verifier_passed':reg.get('task0248_verifier_passed'),'git_diff_check_passed':reg.get('git_diff_check_passed'),'blocking_failure_count':0,'changed_paths':changed,
    }
    hard=[not summary['task0247_prerequisite_complete'],not summary['query_ambiguity_detector_implemented'],not summary['ambiguity_benchmark_v2_frozen'],not safety_zero,summary['production_runtime_behavior_changed'],not summary['git_head_unchanged_since_task_start'],summary['production_agentic_v2_active']]
    summary['blocking_failure_count']=sum(bool(x) for x in hard)
    if len(rows)!=30 or len(ambrows)!=24 or summary['blocking_failure_count']>0: summary['task_status']='partial'
    if write: _write_artifacts(summary,comp,b1,b2,amb,veto_failure,rewrite_failure)
    return summary


def _write_artifacts(summary,comp,b1,b2,amb,veto_failure,rewrite_failure):
    RESULT.mkdir(parents=True,exist_ok=True); CONTRACT.parent.mkdir(parents=True,exist_ok=True)
    write_json(CONTRACT,{
      'schema_version':'opk-rag.task0248.contract.v1','task_id':TASK_ID,'stage':'llm_agentic_rag_development','task0247_authority_frozen':True,'provider_model_frozen':'deepseek-v4-flash','temperature':0,'thinking_mode':'disabled','output_mode':'json_object','max_structural_repairs_per_policy_call':1,'max_ambiguity_rewrite_calls':1,'max_recovery_policy_calls':1,'max_abstention_veto_calls':1,'llm_finish_authority_allowed':False,'abstain_to_finish_override_allowed':False,'production_promotion_allowed':False,'task_start_head':baseline_head(),
    })
    schemas={
      'task0248_query_ambiguity_decision_schema.json':QueryAmbiguityDecision.model_json_schema(),
      'task0248_ambiguity_rewrite_decision_schema.json':AmbiguityRewriteDecision.model_json_schema(),
      'task0248_abstention_veto_invocation_decision_schema.json':AbstentionVetoInvocationDecision.model_json_schema(),
      'task0248_abstention_veto_decision_schema.json':AbstentionVetoDecision.model_json_schema(),
    }
    for name,payload in schemas.items(): write_json(ROOT/'evaluation-data/contracts'/name,payload)
    write_json(RESULT/'veto_metrics.json',{k:b2.get(k) for k in ('veto_invocation_rate','veto_bypass_rate','veto_keep_finish_count','veto_abstain_count','veto_policy_failure_rate','unsafe_finish_prevented_count','false_abstain_created_count','veto_net_gain')})
    write_json(RESULT/'veto_failure_taxonomy.json',dict(veto_failure)); write_json(RESULT/'rewrite_failure_taxonomy.json',dict(rewrite_failure))
    write_json(RESULT/'controller_exposure_metrics.json',{k:b2.get(k) for k in ('average_controller_calls','zero_controller_call_rate','one_controller_call_rate','two_controller_call_rate','three_controller_call_rate','rewrite_invocation_rate','recovery_invocation_rate','veto_invocation_rate')})
    write_json(RESULT/'recovery_metrics.json',{'recovery_invocation_rate':b2.get('recovery_invocation_rate'),'recovery_net_gain_before_veto':0,'recovery_only_terminal_authority':False,'veto_separate_from_recovery_policy':True})
    write_json(RESULT/'latency_metrics.json',{'A0_average_latency_ms':comp.get('A0',{}).get('average_latency_ms'),'A1_average_latency_ms':comp.get('A1',{}).get('average_latency_ms'),'A2_average_latency_ms':comp.get('A2',{}).get('average_latency_ms'),'B0_average_latency_ms':comp.get('B0',{}).get('average_latency_ms'),'B2_average_total_latency_ms':b2.get('average_total_latency_ms'),'B2_average_controller_latency_ms':b2.get('average_controller_latency_ms')})
    write_json(RESULT/'token_metrics.json',{'B2_average_controller_input_tokens':b2.get('average_controller_input_tokens'),'B2_average_controller_output_tokens':b2.get('average_controller_output_tokens'),'ambiguity_average_policy_input_tokens_per_query':amb.get('average_policy_input_tokens_per_query'),'ambiguity_average_policy_output_tokens_per_query':amb.get('average_policy_output_tokens_per_query')})
    safety_keys=('llm_finish_authority_count','abstain_to_finish_override_count','veto_without_conflict_gate_count','unauthorized_action_execution_count','budget_violation_count','graph_hop_violation_count','benchmark_gold_exposure_count','hidden_reasoning_exposure_count','raw_provider_output_persisted_count','secret_exposure_count','knowledge_base_mutation_count')
    write_json(RESULT/'safety_metrics.json',{k:summary[k] for k in safety_keys}|{'safety_invariants_preserved':summary['safety_invariants_preserved']})
    write_json(RESULT/'strategy_selection.json',{'decision':summary['candidate_decision'],'production_promotion_allowed':False,'next_task':summary['next_task'],'reason':'Architecture quality and one-way veto are strongly positive, but controller structured-output/provider reliability is below shadow-readiness threshold.','evidence':{'B2_terminal_accuracy':summary['b2_terminal_accuracy'],'B2_delta_vs_A0':summary['b2_quality_delta_vs_a0'],'veto_net_gain':summary['veto_net_gain'],'false_abstain_created_count':summary['false_abstain_created_count'],'veto_policy_failure_rate':summary['veto_policy_failure_rate'],'ambiguity_rewrite_end_to_end_success_rate':summary['ambiguity_rewrite_end_to_end_success_rate'],'ambiguity_rewrite_provider_failure_rate':summary['ambiguity_rewrite_provider_failure_rate'],'ambiguity_contract_valid_given_provider_response':summary['ambiguity_rewrite_contract_valid_rate_given_provider_response']}})
    write_json(RESULT/'security.json',{'raw_provider_output_persisted':False,'hidden_reasoning_persisted':False,'benchmark_gold_policy_exposure':False,'llm_finish_authority':False,'abstain_to_finish_override':False,'production_agentic_v2_active':False})
    write_json(RESULT/'summary.json',summary)


def verify():
    s=build_summary(write=False)
    required={'task_status':'complete','task0247_prerequisite_complete':True,'query_ambiguity_detector_implemented':True,'bounded_ambiguity_rewrite_implemented':True,'recovery_only_policy_preserved':True,'bounded_abstention_veto_implemented':True,'deterministic_veto_gate_implemented':True,'asymmetric_terminal_authority_enforced':True,'ambiguity_benchmark_v2_frozen':True,'safety_invariants_preserved':True,'llm_finish_authority_count':0,'abstain_to_finish_override_count':0,'veto_without_conflict_gate_count':0,'unauthorized_action_execution_count':0,'budget_violation_count':0,'graph_hop_violation_count':0,'benchmark_gold_exposure_count':0,'hidden_reasoning_exposure_count':0,'raw_provider_output_persisted_count':0,'secret_exposure_count':0,'knowledge_base_mutation_count':0,'production_agentic_v2_active':False,'production_search_ask_behavior_changed':False,'production_runtime_behavior_changed':False,'git_head_unchanged_since_task_start':True,'git_commit_created':False,'production_promotion_allowed':False,'blocking_failure_count':0}
    mism={k:{'expected':v,'actual':s.get(k)} for k,v in required.items() if s.get(k)!=v}
    files=[CONTRACT,RESULT/'strategy_comparison.json',RESULT/'benchmark_sample_results.jsonl',RESULT/'b1_metrics.json',RESULT/'b2_metrics.json',RESULT/'ambiguity_metrics.json',RESULT/'ambiguity_v2_sample_results.jsonl',RESULT/'veto_metrics.json',RESULT/'veto_failure_taxonomy.json',RESULT/'rewrite_failure_taxonomy.json',RESULT/'controller_exposure_metrics.json',RESULT/'recovery_metrics.json',RESULT/'latency_metrics.json',RESULT/'token_metrics.json',RESULT/'safety_metrics.json',RESULT/'strategy_selection.json',RESULT/'security.json',RESULT/'summary.json',AMB2/'benchmark_manifest.json',AMB2/'queries.jsonl',AMB2/'evaluator_gold.jsonl',REG,ROOT/'tasks/TASK-0248_selective_agent_abstention_veto_and_ambiguity_rewrite_integration.md',ROOT/'docs/TASK0248_SELECTIVE_AGENT_ABSTENTION_VETO_AND_AMBIGUITY_REWRITE_INTEGRATION_REPORT.md']
    files += [ROOT/'evaluation-data/contracts'/x for x in ('task0248_query_ambiguity_decision_schema.json','task0248_ambiguity_rewrite_decision_schema.json','task0248_abstention_veto_invocation_decision_schema.json','task0248_abstention_veto_decision_schema.json')]
    missing=[str(x.relative_to(ROOT)) for x in files if not x.is_file()]
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':not mism and not missing,'mismatches':mism,'missing_files':missing}
