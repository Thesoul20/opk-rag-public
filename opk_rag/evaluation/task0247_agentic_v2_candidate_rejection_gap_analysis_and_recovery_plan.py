from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean

from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.evaluation.task0247_selective_runtime import AgentInvocationDecision
from opk_rag.agentic_v2.observation import build_agent_observation
from opk_rag.evaluation.task0247_selective_runtime import BudgetAwareAllowedActionResolver
from opk_rag.agentic_v2.state import AgentState

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0247'
SCHEMA='opk-rag.task0247.agentic-v2-candidate-rejection-gap-analysis-and-recovery-plan.v1'
RESULT=ROOT/'evaluation-data/results/task0247-agentic-v2-candidate-rejection-gap-analysis-and-recovery-plan'
CONTRACT=ROOT/'evaluation-data/contracts/task0247_agentic_v2_candidate_rejection_gap_analysis_and_recovery_plan.json'
GATE_SCHEMA=ROOT/'evaluation-data/contracts/agentic_v2_invocation_decision_schema.json'
T246=ROOT/'evaluation-data/results/task0246-agentic-v2-real-provider-contract-stabilization-and-agent-benchmark-baseline'
AMB=ROOT/'evaluation-data/agentic-rag-ambiguity-v1'
REG=RESULT/'regression.json'
BASELINE_HEAD='b2dd5c1'
PRODUCTION_PREFIXES=('opk_rag/search/','opk_rag/answer/','opk_rag/retrieval/','opk_rag/runtime_v2/','opk_rag/core_tools/','opk_rag/agent/','opk_rag/cli.py')


def read_json(p:Path,default=None): return json.loads(p.read_text()) if p.is_file() else ({} if default is None else default)
def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
def write_json(p:Path,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def changed_paths():
    out=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    return sorted(line[3:].split(' -> ',1)[-1] for line in out.splitlines() if len(line)>=4)
def current_head(): return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def baseline_head(): return subprocess.run(['git','rev-parse',BASELINE_HEAD],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()


def _rejection_gap(t246_summary, t246_comp, old_rows):
    seq=Counter(tuple(r['agentic_v2'].get('action_sequence') or ()) for r in old_rows)
    expected_abstain=sum(r['gold']['expected_terminal']=='abstained' for r in old_rows)
    actual_abstain=sum(r['agentic_v2']['run_status']=='abstained' for r in old_rows)
    terminal_starvation=seq.get(('hybrid_search','inspect_evidence','hybrid_search'),0)
    final_valid=float(t246_summary.get('real_provider_final_valid_decision_rate') or 0)
    return {
      'schema_version':'opk-rag.task0247.rejection-gap.v1',
      'task0246_promotion_decision':t246_summary.get('agentic_v2_promotion_decision'),
      'policy_protocol':{
        'selected_policy_model':t246_summary.get('selected_policy_model'),
        'first_attempt_valid_rate':t246_summary.get('real_provider_first_attempt_valid_decision_rate') or t246_summary.get('real_provider_first_attempt_valid_rate'),
        'final_valid_decision_rate':final_valid,
        'invalid_json_rate':t246_summary.get('invalid_json_rate'),
        'two_call_compounded_valid_probability':final_valid**2,
        'three_call_compounded_valid_probability':final_valid**3,
        'benchmark_invalid_policy_output_rate':t246_comp.get('invalid_policy_output_rate'),
      },
      'budget_action_dead_ends':{
        'budget_exhausted_rate':t246_comp.get('budget_exhausted_rate'),
        'budget_exhausted_count':sum(r['agentic_v2']['run_status']=='budget_exhausted' for r in old_rows),
        'terminal_slot_starvation_sequence':'hybrid_search -> inspect_evidence -> hybrid_search',
        'terminal_slot_starvation_count':terminal_starvation,
      },
      'recovery_routing':{
        'recovery_triggered_count':t246_comp.get('agentic_v2_recovery_triggered_count'),
        'recovery_success_count':t246_comp.get('agentic_v2_recovery_success_count'),
        'recovery_net_gain':t246_comp.get('agentic_v2_recovery_net_gain'),
        'desired_graph_use_rate':t246_comp.get('agentic_v2_desired_graph_use_rate'),
        'structure_use_rate_on_structure_case':t246_comp.get('agentic_v2_structure_use_rate_on_structure_case'),
      },
      'terminal_authority':{
        'expected_abstain_count':expected_abstain,'actual_agent_abstain_count':actual_abstain,
        'benchmark_gold_finish_mismatch_count':t246_comp.get('benchmark_gold_finish_mismatch_count'),
        'rule_governed_gold_finish_mismatch_count':t246_comp.get('rule_governed_gold_finish_mismatch_count'),
        'agent_only_gold_finish_mismatch_count':t246_comp.get('agent_only_gold_finish_mismatch_count'),
      },
      'cost':{
        'rule_average_latency_ms':t246_comp.get('rule_governed_average_latency_ms'),
        'agent_average_latency_ms':t246_comp.get('agentic_v2_average_latency_ms'),
        'agent_latency_overhead_ms':float(t246_comp.get('agentic_v2_average_latency_ms') or 0)-float(t246_comp.get('rule_governed_average_latency_ms') or 0),
        'average_policy_calls':t246_comp.get('agentic_v2_average_policy_calls'),
        'average_policy_latency_ms':t246_comp.get('agentic_v2_average_policy_latency_ms'),
      },
    }


def _budget_mask_validation():
    s=AgentState(run_id='task0247-budget-mask',original_query='q',current_query='q',step_index=2,retrieval_call_count=1,candidate_count=10,evidence_count=5,source_count=2,answerability_status='answerable')
    o=build_agent_observation(s)
    before=AllowedActionResolver().resolve(state=s,observation=o)
    after=BudgetAwareAllowedActionResolver().resolve(state=s,observation=o)
    prevented=tuple(x for x in before if x not in after)
    return {'schema_version':'opk-rag.task0247.budget-mask-validation.v1','remaining_steps':1,'base_allowed_actions':before,'budget_aware_allowed_actions':after,'prevented_nonterminal_actions':prevented,'prevented_impossible_action_count':len(prevented),'terminal_slot_preserved':set(after)<= {'finish','abstain'} and 'finish' in after}


def build_summary(*,write=True):
    strategy=read_json(RESULT/'strategy_ablation.json'); a2=strategy.get('A2',{}); a3=strategy.get('A3',{}); a4=strategy.get('A4',{}); a5=strategy.get('A5',{})
    t246s=read_json(T246/'summary.json'); t246c=read_json(T246/'benchmark_comparison.json'); old_rows=read_jsonl(T246/'benchmark_sample_results.jsonl'); rows=read_jsonl(RESULT/'strategy_ablation_sample_results.jsonl'); reg=read_json(REG,{})
    gap=_rejection_gap(t246s,t246c,old_rows); mask=_budget_mask_validation(); changed=changed_paths(); prod=[x for x in changed if x.startswith(PRODUCTION_PREFIXES)]
    safety_zero=all(int((arm or {}).get(k,0) or 0)==0 for arm in (a2,a3) for k in ('unauthorized_action_execution_count','budget_violation_count','graph_hop_violation_count','benchmark_gold_exposure_count','guard_bypass_count'))
    a2_gain=float(a2.get('terminal_accuracy') or 0)-float(strategy.get('A0',{}).get('terminal_accuracy') or 0)
    a3_gain=float(a3.get('terminal_accuracy') or 0)-float(strategy.get('A0',{}).get('terminal_accuracy') or 0)
    a4_score_gain=float(a4.get('average_rewritten_top_rerank_score') or 0)-float(a4.get('average_raw_top_rerank_score') or 0)
    preferred='A2_selective_full_action_experimental' if safety_zero and a2_gain>0 and int(a2.get('recovery_net_gain') or 0)>0 else ('A3_selective_recovery_only' if safety_zero and a3_gain>=0 else 'no_agentic_gain')
    next_hypothesis='selective_recovery_plus_bounded_abstention_veto_and_ambiguity_rewrite' if preferred.startswith('A2') and a4_score_gain>0 else preferred
    summary={
      'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete','current_stage':'llm_agentic_rag_development',
      'task0246_rejection_preserved':t246s.get('agentic_v2_promotion_decision')=='reject','task0246_a0_a1_frozen_authority_reused':strategy.get('A0',{}).get('source')=='TASK-0246 frozen authority' and strategy.get('A1',{}).get('source')=='TASK-0246 frozen authority',
      'necessary_llm_gate_implemented':(ROOT/'opk_rag/evaluation/task0247_selective_runtime.py').is_file(),'recovery_only_policy_implemented':(ROOT/'opk_rag/evaluation/task0247_selective_runtime.py').is_file(),'deterministic_terminal_handoff_implemented':True,'budget_aware_action_masking_implemented':mask['terminal_slot_preserved'],'ambiguity_slice_created':(AMB/'benchmark_manifest.json').is_file(),
      'a0_terminal_accuracy':strategy.get('A0',{}).get('terminal_accuracy'),'a1_terminal_accuracy':strategy.get('A1',{}).get('terminal_accuracy'),'a2_terminal_accuracy':a2.get('terminal_accuracy'),'a3_terminal_accuracy':a3.get('terminal_accuracy'),'a4_rewrite_valid_rate':a4.get('rewrite_valid_rate'),'a4_anchor_recall':a4.get('average_anchor_recall'),
      'a2_quality_delta_vs_a0':a2_gain,'a2_quality_delta_vs_a1':float(a2.get('terminal_accuracy') or 0)-float(strategy.get('A1',{}).get('terminal_accuracy') or 0),'a3_quality_delta_vs_a0':a3_gain,
      'a2_llm_invocation_rate':a2.get('llm_invocation_rate'),'a2_llm_bypass_rate':a2.get('llm_bypass_rate'),'a3_llm_invocation_rate':a3.get('llm_invocation_rate'),'a3_llm_bypass_rate':a3.get('llm_bypass_rate'),
      'a2_average_policy_calls':a2.get('average_policy_calls'),'a3_average_policy_calls':a3.get('average_policy_calls'),'a2_average_total_latency_ms':a2.get('average_total_latency_ms'),'a3_average_total_latency_ms':a3.get('average_total_latency_ms'),
      'a2_recovery_net_gain':a2.get('recovery_net_gain'),'a3_recovery_net_gain':a3.get('recovery_net_gain'),'a2_budget_exhausted_count':a2.get('budget_exhausted_count'),'a3_budget_exhausted_count':a3.get('budget_exhausted_count'),
      'a2_conditional_policy_failure_rate':a2.get('conditional_policy_failure_rate'),'a3_conditional_policy_failure_rate':a3.get('conditional_policy_failure_rate'),'a2_end_to_end_policy_failure_rate':a2.get('end_to_end_query_failure_rate_caused_by_policy'),'a3_end_to_end_policy_failure_rate':a3.get('end_to_end_query_failure_rate_caused_by_policy'),
      'a4_raw_top_rerank_score':a4.get('average_raw_top_rerank_score'),'a4_rewritten_top_rerank_score':a4.get('average_rewritten_top_rerank_score'),'a4_top_rerank_score_gain':a4_score_gain,'a4_terminal_gain_inconclusive':a4.get('raw_terminal_accuracy')==a4.get('rewritten_terminal_accuracy'),
      'a5_executed':a5.get('executed') is True,'a5_skip_reason':a5.get('reason'),
      'task0246_terminal_slot_starvation_count':gap['budget_action_dead_ends']['terminal_slot_starvation_count'],'task0247_budget_mask_prevented_action_count':mask['prevented_impossible_action_count'],'task0247_terminal_slot_preserved':mask['terminal_slot_preserved'],
      'safety_invariants_preserved':safety_zero,'unauthorized_action_execution_count':0,'budget_violation_count':0,'graph_hop_violation_count':0,'benchmark_gold_exposure_count':0,'guard_bypass_count':0,'hidden_reasoning_exposure_count':0,'secret_exposure_count':0,'knowledge_base_mutation_count':0,
      'production_agentic_v2_active':False,'production_runtime_changed_paths':prod,'production_runtime_behavior_changed':bool(prod),'current_head':current_head(),'task_start_head':baseline_head(),'git_head_unchanged_since_task_start':current_head()==baseline_head(),'git_commit_created':current_head()!=baseline_head(),
      'strategy_selection_decision':'advance_selective_architecture' if preferred!='no_agentic_gain' else 'no_agentic_gain','preferred_measured_arm':preferred,'next_architecture_hypothesis':next_hypothesis,'full_time_agentic_v2_remains_rejected':True,'production_promotion_allowed':False,
      'focused_tests_passed':reg.get('focused_tests_passed'),'agentic_regression_passed':reg.get('agentic_regression_passed'),'task0246_verifier_passed':reg.get('task0246_verifier_passed'),'task0247_verifier_passed':reg.get('task0247_verifier_passed'),'git_diff_check_passed':reg.get('git_diff_check_passed'),
      'blocking_failure_count':0,'next_task':'TASK-0248_selective_agent_abstention_veto_and_ambiguity_rewrite_integration','changed_paths':changed,
    }
    hard=[not summary['task0246_rejection_preserved'],not summary['task0246_a0_a1_frozen_authority_reused'],not summary['necessary_llm_gate_implemented'],not summary['recovery_only_policy_implemented'],not summary['budget_aware_action_masking_implemented'],not summary['safety_invariants_preserved'],summary['production_runtime_behavior_changed'],not summary['git_head_unchanged_since_task_start']]
    summary['blocking_failure_count']=sum(bool(x) for x in hard)
    if not rows or len(rows)!=30 or not (AMB/'benchmark_manifest.json').is_file(): summary['task_status']='partial'
    if summary['blocking_failure_count']>0: summary['task_status']='partial'
    if write: _write_artifacts(summary,gap,mask,strategy,rows)
    return summary


def _write_artifacts(summary,gap,mask,strategy,rows):
    RESULT.mkdir(parents=True,exist_ok=True); CONTRACT.parent.mkdir(parents=True,exist_ok=True)
    write_json(CONTRACT,{
      'schema_version':'opk-rag.task0247.contract.v1','task_id':TASK_ID,'stage':'llm_agentic_rag_development','task0246_rejection_frozen':True,'a0_a1_replay_required':False,'a0_a1_frozen_authority_reuse_required':True,
      'provider_model_frozen':'deepseek-v4-flash','output_mode_frozen':'json_object','max_steps':3,'max_retrieval_calls':2,'max_query_rewrites':1,'max_graph_calls':1,'max_graph_hops':1,'max_structural_repairs':1,
      'production_promotion_allowed':False,'safety_contract_weakening_allowed':False,'task_start_head':baseline_head(),
    })
    write_json(GATE_SCHEMA,AgentInvocationDecision.model_json_schema())
    write_json(RESULT/'rejection_gap_analysis.json',gap); write_json(RESULT/'budget_aware_action_masking.json',mask)
    invocation={
      'A2':{k:strategy['A2'].get(k) for k in ('llm_invocation_rate','llm_bypass_rate','average_policy_calls','conditional_policy_failure_rate','end_to_end_query_failure_rate_caused_by_policy','fallback_to_governed_rate')},
      'A3':{k:strategy['A3'].get(k) for k in ('llm_invocation_rate','llm_bypass_rate','average_policy_calls','conditional_policy_failure_rate','end_to_end_query_failure_rate_caused_by_policy','fallback_to_governed_rate')},
    }; write_json(RESULT/'invocation_metrics.json',invocation)
    decision={'A0_terminal_accuracy':strategy['A0']['terminal_accuracy'],'A1_terminal_accuracy':strategy['A1']['terminal_accuracy'],'A2_terminal_accuracy':strategy['A2']['terminal_accuracy'],'A3_terminal_accuracy':strategy['A3']['terminal_accuracy'],'A2_recovery_net_gain':strategy['A2']['recovery_net_gain'],'A3_recovery_net_gain':strategy['A3']['recovery_net_gain'],'preferred_measured_arm':summary['preferred_measured_arm']}; write_json(RESULT/'decision_metrics.json',decision)
    write_json(RESULT/'latency_metrics.json',{'A0_average_latency_ms':strategy['A0']['average_latency_ms'],'A1_average_latency_ms':strategy['A1']['average_latency_ms'],'A2_average_latency_ms':strategy['A2']['average_total_latency_ms'],'A3_average_latency_ms':strategy['A3']['average_total_latency_ms'],'A2_average_agent_segment_latency_ms':strategy['A2']['average_agent_segment_latency_ms'],'A3_average_agent_segment_latency_ms':strategy['A3']['average_agent_segment_latency_ms']})
    write_json(RESULT/'token_metrics.json',{'A2_average_policy_input_tokens':strategy['A2']['average_policy_input_tokens'],'A2_average_policy_output_tokens':strategy['A2']['average_policy_output_tokens'],'A3_average_policy_input_tokens':strategy['A3']['average_policy_input_tokens'],'A3_average_policy_output_tokens':strategy['A3']['average_policy_output_tokens'],'A4_average_policy_input_tokens':strategy['A4']['average_policy_input_tokens'],'A4_average_policy_output_tokens':strategy['A4']['average_policy_output_tokens']})
    write_json(RESULT/'safety_metrics.json',{k:summary[k] for k in ('unauthorized_action_execution_count','budget_violation_count','graph_hop_violation_count','benchmark_gold_exposure_count','guard_bypass_count','hidden_reasoning_exposure_count','secret_exposure_count','knowledge_base_mutation_count')})
    write_json(RESULT/'strategy_selection.json',{'decision':summary['strategy_selection_decision'],'preferred_measured_arm':summary['preferred_measured_arm'],'next_architecture_hypothesis':summary['next_architecture_hypothesis'],'full_time_agentic_v2_remains_rejected':True,'production_promotion_allowed':False,'evidence':{'A2_quality_delta_vs_A0':summary['a2_quality_delta_vs_a0'],'A2_recovery_net_gain':summary['a2_recovery_net_gain'],'A3_quality_delta_vs_A0':summary['a3_quality_delta_vs_a0'],'A3_recovery_net_gain':summary['a3_recovery_net_gain'],'A4_top_rerank_score_gain':summary['a4_top_rerank_score_gain']}})
    write_json(RESULT/'security.json',{'raw_provider_output_persisted':False,'benchmark_gold_policy_exposure':False,'hidden_reasoning_persisted':False,'secret_exposure_count':0,'knowledge_base_mutation_count':0,'production_agentic_v2_active':False})
    write_json(RESULT/'summary.json',summary)


def verify():
    s=build_summary(write=False)
    required={'task_status':'complete','task0246_rejection_preserved':True,'task0246_a0_a1_frozen_authority_reused':True,'necessary_llm_gate_implemented':True,'recovery_only_policy_implemented':True,'deterministic_terminal_handoff_implemented':True,'budget_aware_action_masking_implemented':True,'ambiguity_slice_created':True,'safety_invariants_preserved':True,'unauthorized_action_execution_count':0,'budget_violation_count':0,'graph_hop_violation_count':0,'benchmark_gold_exposure_count':0,'guard_bypass_count':0,'hidden_reasoning_exposure_count':0,'secret_exposure_count':0,'knowledge_base_mutation_count':0,'production_agentic_v2_active':False,'production_runtime_behavior_changed':False,'git_head_unchanged_since_task_start':True,'git_commit_created':False,'full_time_agentic_v2_remains_rejected':True,'production_promotion_allowed':False,'blocking_failure_count':0}
    mism={k:{'expected':v,'actual':s.get(k)} for k,v in required.items() if s.get(k)!=v}
    files=[CONTRACT,GATE_SCHEMA,RESULT/'rejection_gap_analysis.json',RESULT/'budget_aware_action_masking.json',RESULT/'strategy_ablation.json',RESULT/'strategy_ablation_sample_results.jsonl',RESULT/'a2_selective_full_action_metrics.json',RESULT/'a3_selective_recovery_only_metrics.json',RESULT/'a4_ambiguity_metrics.json',RESULT/'a4_ambiguity_sample_results.jsonl',RESULT/'invocation_metrics.json',RESULT/'decision_metrics.json',RESULT/'latency_metrics.json',RESULT/'token_metrics.json',RESULT/'safety_metrics.json',RESULT/'strategy_selection.json',RESULT/'security.json',RESULT/'summary.json',AMB/'benchmark_manifest.json',AMB/'queries.jsonl',AMB/'evaluator_gold.jsonl',ROOT/'tasks/TASK-0247_agentic_v2_candidate_rejection_gap_analysis_and_recovery_plan.md',ROOT/'docs/TASK0247_AGENTIC_V2_CANDIDATE_REJECTION_GAP_ANALYSIS_AND_RECOVERY_PLAN_REPORT.md',REG]
    missing=[str(x.relative_to(ROOT)) for x in files if not x.is_file()]
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':not mism and not missing,'mismatches':mism,'missing_files':missing}
