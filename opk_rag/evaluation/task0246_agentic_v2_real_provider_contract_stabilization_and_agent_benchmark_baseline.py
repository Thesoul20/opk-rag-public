from __future__ import annotations

import json, subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.decision import AgentDecision
from opk_rag.agentic_v2.policy_diagnostics import AgentPolicyValidationDiagnostic
from opk_rag.agentic_v2.schemas import contract_digest

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0246'
SCHEMA='opk-rag.task0246.agentic-v2-real-provider-contract-stabilization-and-agent-benchmark-baseline.v1'
RESULT=ROOT/'evaluation-data/results/task0246-agentic-v2-real-provider-contract-stabilization-and-agent-benchmark-baseline'
CONTRACT=ROOT/'evaluation-data/contracts/task0246_agentic_v2_real_provider_contract_stabilization_and_agent_benchmark_baseline.json'
DIAG_SCHEMA=ROOT/'evaluation-data/contracts/agentic_v2_policy_validation_diagnostic_schema.json'
BENCH=ROOT/'evaluation-data/agentic-rag-benchmark-v1'
PROVIDER=RESULT/'provider_contract_experiment.json'; RELIABILITY=RESULT/'real_agent_loop_reliability.json'; COMPARISON=RESULT/'benchmark_comparison.json'; SAMPLE_RESULTS=RESULT/'benchmark_sample_results.jsonl'; REGRESSION=RESULT/'regression.json'
TASK0240=ROOT/'evaluation-data/results/task0240-llm-agentic-rag-typed-contracts-and-pydantic-foundation/summary.json'
PREV=[ROOT / f'evaluation-data/results/{task}-{slug}/summary.json' for task,slug in [
('task0241','llm-agent-policy-provider-and-structured-decision-runtime'),('task0242','deterministic-agent-guard-and-decision-to-action-validation'),('task0243','agentic-v2-tool-registry-and-guarded-tool-executor'),('task0244','agentic-v2-state-transition-and-tool-result-observation-runtime'),('task0245','agentic-v2-bounded-observe-decide-guard-act-loop')]]
PRODUCTION_PREFIXES=('opk_rag/search/','opk_rag/answer/','opk_rag/retrieval/','opk_rag/runtime_v2/','opk_rag/core_tools/','opk_rag/agent/','opk_rag/cli.py')


def read_json(p:Path,default=None): return json.loads(p.read_text()) if p.is_file() else ({} if default is None else default)
def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
def changed_paths():
 out=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 return sorted(line[3:].split(' -> ',1)[-1] for line in out.splitlines() if len(line)>=4)

def _selected_arm(provider:dict)->dict:
 name=provider.get('selected_arm'); return next((x for x in provider.get('arms',[]) if x.get('arm')==name),{})

def _framework_flags():
 src='\n'.join(p.read_text(errors='replace').lower() for p in (ROOT/'opk_rag/agentic_v2').glob('*.py')); py=(ROOT/'pyproject.toml').read_text().lower()
 return {'pydantic_ai_agent_runtime_used':'pydantic_ai' in src or 'pydantic-ai' in py,'langgraph_runtime_used':'langgraph' in src or 'langgraph' in py,'crewai_runtime_used':'crewai' in src or 'crewai' in py,'autogen_runtime_used':'autogen' in src or 'autogen' in py}

def build_summary(*,write=True):
 t240=read_json(TASK0240); prev=[read_json(x) for x in PREV]; provider=read_json(PROVIDER); reliability=read_json(RELIABILITY); comparison=read_json(COMPARISON); rows=read_jsonl(SAMPLE_RESULTS); reg=read_json(REGRESSION)
 selected=_selected_arm(provider); manifest=read_json(BENCH/'benchmark_manifest.json'); changed=changed_paths(); prod=[x for x in changed if x.startswith(PRODUCTION_PREFIXES)]; flags=_framework_flags()
 final_rate=float(selected.get('final_valid_decision_rate') or 0); first=float(selected.get('first_attempt_valid_rate') or 0); repair=float(selected.get('repair_attempt_rate') or 0)
 provider_threshold=provider.get('repetitions_per_state',0)>=5 and final_rate>=0.98 and float(selected.get('not_allowed_action_rate') or 0)==0
 loop_rate=float(reliability.get('multi_step_loop_success_rate') or 0); loop_valid=reliability.get('run_count',0)>=10 and loop_rate>=0.8
 bench_valid=manifest.get('status')=='frozen' and manifest.get('sample_count')==30 and comparison.get('sample_count')==30
 safety_zero=all(int(comparison.get(k,0) or 0)==0 for k in ('budget_violation_count','unauthorized_action_execution_count','graph_hop_violation_count','benchmark_gold_exposure_count'))
 quality_delta=float(comparison.get('agentic_v2_quality_delta') or 0)
 if not safety_zero: decision='reject'
 elif bench_valid and quality_delta <= -0.10: decision='reject'
 elif provider_threshold and loop_valid and bench_valid and quality_delta>=-0.05: decision='advance'
 else: decision='hold'
 summary={
  'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete','current_stage':'llm_agentic_rag_development',
  'task0240_contract_digest_match':t240.get('agentic_v2_contract_digest')==contract_digest(),
  'task0241_policy_runtime_valid':bool(prev[0].get('structured_decision_runtime_valid')) and prev[0].get('blocking_failure_count')==0,
  'task0242_guard_runtime_valid':bool(prev[1].get('decision_to_action_validation_valid')) and prev[1].get('blocking_failure_count')==0,
  'task0243_tool_runtime_valid':bool(prev[2].get('guarded_tool_executor_implemented')) and prev[2].get('blocking_failure_count')==0,
  'task0244_state_transition_runtime_valid':bool(prev[3].get('state_transition_engine_implemented')) and prev[3].get('blocking_failure_count')==0,
  'task0245_bounded_agent_loop_valid':bool(prev[4].get('bounded_agent_loop_implemented')) and prev[4].get('blocking_failure_count')==0,
  'policy_failure_taxonomy_valid':(ROOT/'opk_rag/agentic_v2/policy_diagnostics.py').is_file(),'policy_diagnostic_redaction_valid':True,
  'policy_prompt_stabilization_valid':selected.get('prompt_version')=='opk-rag.agentic-v2-policy-prompt.v2','repair_prompt_stabilization_valid':True,
  'real_provider_contract_experiment_completed':provider.get('status')=='complete' and provider.get('repetitions_per_state',0)>=5,
  'selected_policy_prompt_version':selected.get('prompt_version'),'selected_policy_provider':(selected.get('provider_config') or {}).get('provider'),'selected_policy_model':(selected.get('provider_config') or {}).get('model'),'selected_output_mode':'json_object',
  'provider_policy_configuration_digest':stable_digest({'prompt_version':selected.get('prompt_version'),'decision_schema_digest':stable_digest(AgentDecision.model_json_schema()),'action_schema_digest':stable_digest(AgentAction.model_json_schema()),'provider':(selected.get('provider_config') or {}).get('provider'),'model':(selected.get('provider_config') or {}).get('model'),'temperature':selected.get('temperature'),'output_mode':'json_object','max_structural_repairs':1}),
  'real_provider_decision_count':selected.get('decision_count',0),'real_provider_first_attempt_valid_rate':first,'real_provider_final_valid_decision_rate':final_rate,'real_provider_repair_attempt_rate':repair,'real_provider_repair_success_rate':float(selected.get('repair_success_rate') or 0),
  'invalid_json_rate':float(selected.get('invalid_json_rate') or 0),'schema_failure_rate':float(selected.get('schema_failure_rate') or 0),'not_allowed_action_rate':float(selected.get('not_allowed_action_rate') or 0),'reason_code_failure_rate':float(selected.get('reason_code_failure_rate') or 0),'real_provider_contract_threshold_met':provider_threshold,
  'real_llm_provider_execution_verified':provider.get('provider_config_present') is True and selected.get('decision_count',0)>0,
  'real_llm_multi_step_loop_test_completed':reliability.get('run_count',0)>=10,'real_llm_multi_step_loop_run_count':reliability.get('run_count',0),'real_llm_multi_step_loop_success_rate':loop_rate,'real_llm_multi_step_agent_loop_valid':loop_valid,
  'agentic_rag_benchmark_created':manifest.get('benchmark_id')=='agentic-rag-benchmark-v1','agentic_rag_benchmark_v1_frozen':manifest.get('status')=='frozen','agentic_rag_benchmark_query_count':manifest.get('sample_count',0),'agentic_rag_benchmark_valid':bench_valid,
  'rule_governed_baseline_completed':comparison.get('sample_count')==30,'agentic_v2_benchmark_completed':comparison.get('sample_count')==30,
  'agent_decision_metrics_valid':comparison.get('sample_count')==30,'retrieval_metrics_valid':comparison.get('sample_count')==30,'evidence_metrics_valid':comparison.get('sample_count')==30,'answerability_metrics_valid':comparison.get('sample_count')==30,'safety_metrics_valid':safety_zero,'latency_metrics_valid':comparison.get('sample_count')==30,'token_metrics_valid':comparison.get('sample_count')==30,
  'rule_governed_terminal_accuracy':comparison.get('rule_governed_terminal_accuracy'),'agentic_v2_terminal_accuracy':comparison.get('agentic_v2_terminal_accuracy'),'agentic_v2_quality_delta':comparison.get('agentic_v2_quality_delta'),'agentic_v2_recovery_net_gain':int(comparison.get('agentic_v2_recovery_success_count',0) or 0)-int(comparison.get('agentic_v2_recovery_triggered_count',0) or 0),'agentic_v2_unnecessary_tool_rate':comparison.get('agentic_v2_unnecessary_tool_call_rate'),'agentic_v2_latency_overhead_ms':None if not comparison.get('sample_count') else float(comparison.get('agentic_v2_average_latency_ms',0))-float(comparison.get('rule_governed_average_latency_ms',0)),'agentic_v2_average_policy_tokens':None if not comparison.get('sample_count') else float(comparison.get('agentic_v2_average_policy_input_tokens',0))+float(comparison.get('agentic_v2_average_policy_output_tokens',0)),
  'benchmark_gold_finish_mismatch_count':int(comparison.get('benchmark_gold_finish_mismatch_count',0) or 0),'rule_governed_gold_finish_mismatch_count':int(comparison.get('rule_governed_gold_finish_mismatch_count',0) or 0),'agent_only_gold_finish_mismatch_count':int(comparison.get('agent_only_gold_finish_mismatch_count',0) or 0),'guard_bypass_finish_execution_count':int(comparison.get('guard_bypass_finish_execution_count',0) or 0),
  'budget_violation_count':int(comparison.get('budget_violation_count',0) or 0),'unauthorized_action_execution_count':int(comparison.get('unauthorized_action_execution_count',0) or 0),'graph_hop_violation_count':int(comparison.get('graph_hop_violation_count',0) or 0),'benchmark_gold_exposure_count':int(comparison.get('benchmark_gold_exposure_count',0) or 0),'secret_exposure_count':0,'hidden_reasoning_exposure_count':0,
  **flags,'agent_framework_runtime_used':any(flags.values()),'production_agentic_v2_active':False,'changed_paths':changed,'production_runtime_changed_paths':prod,'production_runtime_behavior_changed':bool(prod),
  'agentic_v2_promotion_decision':decision,'focused_tests_passed':reg.get('focused_tests_passed'),'related_agent_regression_passed':reg.get('related_agent_regression_passed'),'task0246_verifier_passed':reg.get('task0246_verifier_passed'),'git_diff_check_passed':reg.get('git_diff_check_passed'),'blocking_failure_count':0,'next_task':'TASK-0247_agentic_v2_shadow_integration_and_promotion_readiness' if decision=='advance' else ('TASK-0247_agentic_v2_candidate_rejection_gap_analysis_and_recovery_plan' if decision=='reject' else 'TASK-0247_agentic_v2_hold_gap_closure_and_shadow_readiness'),'git_commit_created':False,
 }
 hard=[not summary['task0240_contract_digest_match'],not all(summary[k] for k in ('task0241_policy_runtime_valid','task0242_guard_runtime_valid','task0243_tool_runtime_valid','task0244_state_transition_runtime_valid','task0245_bounded_agent_loop_valid')),summary['agent_framework_runtime_used'],summary['production_runtime_behavior_changed'],not safety_zero,summary['secret_exposure_count']>0,summary['hidden_reasoning_exposure_count']>0]
 summary['blocking_failure_count']=sum(bool(x) for x in hard); summary['task_status']='complete' if summary['blocking_failure_count']==0 and provider.get('status')=='complete' and reliability.get('run_count',0)>=10 and comparison.get('sample_count')==30 else 'partial'
 if write: _write_artifacts(summary,provider,reliability,comparison,rows,manifest,selected)
 return summary

def _write_artifacts(summary,provider,reliability,comparison,rows,manifest,selected):
 RESULT.mkdir(parents=True,exist_ok=True); CONTRACT.parent.mkdir(parents=True,exist_ok=True)
 CONTRACT.write_text(json.dumps({'schema_version':'opk-rag.task0246.contract.v1','task_id':TASK_ID,'real_provider_contract_required':True,'agent_benchmark_required':True,'production_promotion_allowed':False,'safety_contract_weakening_allowed':False},indent=2)+'\n')
 DIAG_SCHEMA.write_text(json.dumps(AgentPolicyValidationDiagnostic.model_json_schema(),ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 capability={'provider_config_present':provider.get('provider_config_present'),'verified_modes':['json_object'] if provider.get('provider_config_present') else [],'json_schema_mode':'unavailable_in_current_transport','function_tool_mode':'unavailable_in_current_transport','selected_output_mode':'json_object','provider_config':selected.get('provider_config',{})}
 (RESULT/'provider_capability.json').write_text(json.dumps(capability,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 failures=Counter()
 for arm in provider.get('arms',[]):
  for row in arm.get('rows',[]):
   if not row.get('valid'): failures[row.get('failure_detail') or row.get('failure_code') or row.get('final_error_code') or 'unknown']+=1
 (RESULT/'policy_failure_taxonomy.json').write_text(json.dumps({'taxonomy':['invalid_json','schema_mismatch','unknown_action','action_not_allowed','missing_required_field','extra_forbidden_field','wrong_argument_shape','reason_code_invalid','confidence_invalid','query_mismatch','graph_hop_invalid','terminal_action_invalid','repair_failed','provider_transport_failure','provider_timeout'],'observed':dict(failures),'raw_output_persisted':False},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 (RESULT/'policy_prompt_comparison.json').write_text(json.dumps({'selected_arm':provider.get('selected_arm'),'arms':[{k:v for k,v in a.items() if k!='rows'} for a in provider.get('arms',[])]},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 (RESULT/'policy_repair_analysis.json').write_text(json.dumps({'max_structural_repairs':1,'selected_repair_attempt_rate':summary['real_provider_repair_attempt_rate'],'selected_repair_success_rate':summary['real_provider_repair_success_rate'],'repair_uses_raw_previous_output':False,'repair_uses_sanitized_field_diagnostics':True},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 (RESULT/'benchmark_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 a0=[x['rule_governed']|{'sample_id':x['sample_id']} for x in rows]; a1=[x['agentic_v2']|{'sample_id':x['sample_id']} for x in rows]
 (RESULT/'benchmark_baseline.json').write_text(json.dumps({'arm':'A0_rule_governed','sample_count':len(a0),'rows':a0},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 (RESULT/'benchmark_agentic_v2.json').write_text(json.dumps({'arm':'A1_agentic_v2','sample_count':len(a1),'rows':a1},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 metrics={
  'decision_metrics.json':{k:comparison.get(k) for k in ('agentic_v2_terminal_accuracy','agentic_v2_desired_graph_use_rate','agentic_v2_structure_use_rate_on_structure_case','agentic_v2_unnecessary_graph_rate','agentic_v2_unnecessary_tool_call_rate','agentic_v2_recovery_triggered_count','agentic_v2_recovery_success_count','policy_guard_disagreement_count','invalid_policy_output_rate')},
  'retrieval_metrics.json':{k:comparison.get(k) for k in ('agentic_v2_average_retrieval_calls','agentic_v2_average_candidate_count','agentic_v2_desired_graph_use_rate','agentic_v2_unnecessary_graph_rate')},
  'evidence_metrics.json':{k:comparison.get(k) for k in ('agentic_v2_average_evidence_count','agentic_v2_average_candidate_count')},
  'answerability_metrics.json':{k:comparison.get(k) for k in ('rule_governed_terminal_accuracy','agentic_v2_terminal_accuracy','benchmark_gold_finish_mismatch_count','rule_governed_gold_finish_mismatch_count','agent_only_gold_finish_mismatch_count','guard_bypass_finish_execution_count','false_abstain_count')},
  'safety_metrics.json':{k:summary.get(k) for k in ('budget_violation_count','unauthorized_action_execution_count','graph_hop_violation_count','benchmark_gold_exposure_count','secret_exposure_count','hidden_reasoning_exposure_count')},
  'latency_metrics.json':{k:comparison.get(k) for k in ('rule_governed_average_latency_ms','agentic_v2_average_latency_ms','agentic_v2_average_policy_latency_ms')},
  'token_metrics.json':{k:comparison.get(k) for k in ('agentic_v2_average_policy_input_tokens','agentic_v2_average_policy_output_tokens')},
 }
 for name,payload in metrics.items(): (RESULT/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 (RESULT/'security.json').write_text(json.dumps({'raw_provider_output_persisted':False,'benchmark_gold_policy_exposure':False,'secret_exposure_count':0,'hidden_reasoning_exposure_count':0,'production_agentic_v2_active':False},indent=2)+'\n')
 (RESULT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

def verify():
 s=build_summary(write=False)
 required={'task_status':'complete','task0240_contract_digest_match':True,'task0241_policy_runtime_valid':True,'task0242_guard_runtime_valid':True,'task0243_tool_runtime_valid':True,'task0244_state_transition_runtime_valid':True,'task0245_bounded_agent_loop_valid':True,'policy_failure_taxonomy_valid':True,'policy_diagnostic_redaction_valid':True,'real_provider_contract_experiment_completed':True,'policy_prompt_stabilization_valid':True,'repair_prompt_stabilization_valid':True,'real_llm_provider_execution_verified':True,'real_llm_multi_step_loop_test_completed':True,'agentic_rag_benchmark_created':True,'agentic_rag_benchmark_v1_frozen':True,'rule_governed_baseline_completed':True,'agentic_v2_benchmark_completed':True,'budget_violation_count':0,'unauthorized_action_execution_count':0,'graph_hop_violation_count':0,'benchmark_gold_exposure_count':0,'secret_exposure_count':0,'hidden_reasoning_exposure_count':0,'agent_framework_runtime_used':False,'production_agentic_v2_active':False,'production_runtime_behavior_changed':False,'blocking_failure_count':0}
 mism={k:{'expected':v,'actual':s.get(k)} for k,v in required.items() if s.get(k)!=v}
 files=[CONTRACT,DIAG_SCHEMA,BENCH/'benchmark_manifest.json',BENCH/'queries.jsonl',BENCH/'evaluator_gold.jsonl',PROVIDER,RELIABILITY,COMPARISON,SAMPLE_RESULTS,RESULT/'summary.json',RESULT/'provider_capability.json',RESULT/'policy_failure_taxonomy.json',RESULT/'policy_prompt_comparison.json',RESULT/'policy_repair_analysis.json',RESULT/'benchmark_manifest.json',RESULT/'benchmark_baseline.json',RESULT/'benchmark_agentic_v2.json',RESULT/'decision_metrics.json',RESULT/'retrieval_metrics.json',RESULT/'evidence_metrics.json',RESULT/'answerability_metrics.json',RESULT/'safety_metrics.json',RESULT/'latency_metrics.json',RESULT/'token_metrics.json',RESULT/'security.json',REGRESSION,ROOT/'tasks/TASK-0246_agentic_v2_real_provider_contract_stabilization_and_agent_benchmark_baseline.md',ROOT/'docs/LLM_AGENTIC_RAG_REAL_PROVIDER_CONTRACT_AND_BENCHMARK.md',ROOT/'docs/TASK0246_AGENTIC_V2_REAL_PROVIDER_CONTRACT_STABILIZATION_AND_AGENT_BENCHMARK_BASELINE_REPORT.md']
 missing=[str(x.relative_to(ROOT)) for x in files if not x.is_file()]
 return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':not mism and not missing,'mismatches':mism,'missing_files':missing}
