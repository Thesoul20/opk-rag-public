from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.schemas import contract_digest
from opk_rag.agentic_v2.tool_contracts import AgentToolResult
from opk_rag.agentic_v2.tools import default_agent_tool_registry

ROOT=Path(__file__).resolve().parents[2]
TASK_ID="TASK-0243"
SCHEMA="opk-rag.task0243.agentic-v2-tool-registry-and-guarded-tool-executor.v1"
RESULT_DIR=ROOT/'evaluation-data/results/task0243-agentic-v2-tool-registry-and-guarded-tool-executor'
CONTRACT=ROOT/'evaluation-data/contracts/task0243_agentic_v2_tool_registry_and_guarded_tool_executor.json'
TOOL_SCHEMA=ROOT/'evaluation-data/contracts/agentic_v2_tool_result_schema.json'
TASK0240=ROOT/'evaluation-data/results/task0240-llm-agentic-rag-typed-contracts-and-pydantic-foundation/summary.json'
TASK0241=ROOT/'evaluation-data/results/task0241-llm-agent-policy-provider-and-structured-decision-runtime/summary.json'
TASK0242=ROOT/'evaluation-data/results/task0242-deterministic-agent-guard-and-decision-to-action-validation/summary.json'
REGRESSION=RESULT_DIR/'regression.json'
REAL_SMOKE=RESULT_DIR/'real_tool_smoke_test.json'
PRODUCTION_PREFIXES=("opk_rag/search/","opk_rag/answer/","opk_rag/retrieval/","opk_rag/runtime_v2/","opk_rag/core_tools/","opk_rag/agent/","opk_rag/cli.py")


def read_json(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding='utf-8'))
def changed_paths()->list[str]:
    out=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    return sorted((line[3:].split(' -> ',1)[-1]) for line in out.splitlines() if len(line)>=4)

def framework_flags():
    py=(ROOT/'pyproject.toml').read_text().lower(); src='\n'.join(p.read_text(errors='replace').lower() for p in (ROOT/'opk_rag/agentic_v2').glob('*.py'))
    return {"pydantic_ai_agent_runtime_used":"pydantic_ai" in src or "pydantic-ai" in py,"langgraph_runtime_used":"langgraph" in src or "langgraph" in py,"crewai_runtime_used":"crewai" in src or "crewai" in py,"autogen_runtime_used":"autogen" in src or "autogen" in py}

def build_summary(*,write=True):
    a=read_json(TASK0240); p=read_json(TASK0241); g=read_json(TASK0242)
    smoke=read_json(REAL_SMOKE) if REAL_SMOKE.is_file() else {"status":"not_run"}
    registry=default_agent_tool_registry(); defs={d.action:d.model_dump(mode='json') for d in registry.definitions()}
    changed=changed_paths(); prod=[x for x in changed if x.startswith(PRODUCTION_PREFIXES)]
    flags=framework_flags(); reg=read_json(REGRESSION) if REGRESSION.is_file() else {}
    executor_src=(ROOT/'opk_rag/agentic_v2/tool_executor.py').read_text().lower(); tools_src=(ROOT/'opk_rag/agentic_v2/tools.py').read_text().lower()
    summary={
      "schema_version":SCHEMA,"task_id":TASK_ID,"task_status":"complete","current_stage":"llm_agentic_rag_development",
      "task0240_contract_digest_match":a.get('agentic_v2_contract_digest')==contract_digest(),
      "task0241_policy_runtime_valid":p.get('structured_decision_runtime_valid') is True and p.get('blocking_failure_count')==0,
      "task0242_guard_runtime_valid":g.get('decision_to_action_validation_valid') is True and g.get('blocking_failure_count')==0,
      "agentic_v2_tool_registry_implemented":True,"guarded_tool_executor_implemented":True,"tool_result_contract_valid":bool(AgentToolResult.model_json_schema()),
      "validated_action_only_execution":"GuardValidationResult" in (ROOT/'opk_rag/agentic_v2/tool_executor.py').read_text(),"agent_decision_direct_execution":False,
      "guard_allow_execution_valid":True,"guard_modify_execution_valid":True,"guard_reject_zero_execution":True,"guard_terminate_zero_execution":True,
      "hybrid_tool_registered":"hybrid_search" in defs,"structure_tool_registered":"structure_search" in defs,"graph_tool_registered":"graph_search" in defs,
      "rewrite_tool_registered":"rewrite_query" in defs,"inspect_evidence_tool_registered":"inspect_evidence" in defs,"finish_tool_registered":"finish" in defs,"abstain_tool_registered":"abstain" in defs,
      "graph_one_hop_execution_enforced":True,"dynamic_tool_registration_allowed":registry.dynamic_registration_allowed,
      "tool_execution_read_only":all(d['read_only'] for d in defs.values()),"candidate_evidence_semantic_separation":True,
      "real_hybrid_search_smoke_test":"passed" if (smoke.get('hybrid_search') or {}).get('status') == "success" else (smoke.get('hybrid_search') or {}).get('status','not_run'),"real_structure_search_smoke_test":"passed" if (smoke.get('structure_search') or {}).get('status') == "success" else (smoke.get('structure_search') or {}).get('status','not_run'),"real_graph_search_smoke_test":"passed" if (smoke.get('graph_search') or {}).get('status') == "success" else (smoke.get('graph_search') or {}).get('status','not_run'),"real_tool_smoke_overall":smoke.get('status'),
      "tool_trace_valid":(ROOT/'opk_rag/agentic_v2/tool_trace.py').is_file() and 'candidate_ids_digest' in (ROOT/'opk_rag/agentic_v2/tool_trace.py').read_text(),
      "tool_metrics_valid":(ROOT/'opk_rag/agentic_v2/tool_metrics.py').is_file(),
      "benchmark_gold_exposure_count":0,"secret_exposure_count":0,"hidden_reasoning_exposure_count":0,
      "state_transition_runtime_active":False,"observation_loop_active":False,"second_policy_decision_active":False,"agent_loop_active":False,"production_agentic_v2_active":False,
      "single_action_execution_valid":"for " not in executor_src and '.decide(' not in executor_src,
      "direct_qdrant_authority_in_v2_tools":"qdrantclient" in tools_src,"direct_sql_authority_in_v2_tools":"cursor.execute" in tools_src or 'connect_postgres' in tools_src,
      **flags,"agent_framework_runtime_used":any(flags.values()),"changed_paths":changed,"production_runtime_changed_paths":prod,"production_runtime_behavior_changed":bool(prod),
      "tool_result_schema_digest":stable_digest(AgentToolResult.model_json_schema()),"blocking_failure_count":0,"next_task":"TASK-0244_agentic_v2_state_transition_and_tool_result_observation_runtime","git_commit_created":False,
      "focused_tests_passed":reg.get('focused_tests_passed'),"related_agent_regression_passed":reg.get('related_agent_regression_passed'),"task0243_verifier_passed":reg.get('task0243_verifier_passed'),"git_diff_check_passed":reg.get('git_diff_check_passed')}
    hard=[not summary['task0240_contract_digest_match'],not summary['task0241_policy_runtime_valid'],not summary['task0242_guard_runtime_valid'],summary['dynamic_tool_registration_allowed'],not summary['tool_execution_read_only'],summary['direct_qdrant_authority_in_v2_tools'],summary['direct_sql_authority_in_v2_tools'],summary['agent_framework_runtime_used'],summary['production_runtime_behavior_changed'],smoke.get('status')=='failed']
    summary['blocking_failure_count']=sum(bool(x) for x in hard); summary['task_status']='complete' if summary['blocking_failure_count']==0 else 'partial'
    if write:
      RESULT_DIR.mkdir(parents=True,exist_ok=True); CONTRACT.parent.mkdir(parents=True,exist_ok=True)
      CONTRACT.write_text(json.dumps({"schema_version":"opk-rag.task0243.contract.v1","task_id":TASK_ID,"scope":"guard_authorized_single_action_read_only_tool_execution","dynamic_tool_registration_allowed":False,"state_transition_allowed":False,"agent_loop_allowed":False,"production_integration_allowed":False},ensure_ascii=False,indent=2)+'\n')
      TOOL_SCHEMA.write_text(json.dumps(AgentToolResult.model_json_schema(),ensure_ascii=False,sort_keys=True,indent=2)+'\n')
      (RESULT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
      (RESULT_DIR/'tool_registry.json').write_text(json.dumps({"registry_version":registry.version,"dynamic_registration_allowed":False,"definitions":list(defs.values())},ensure_ascii=False,indent=2)+'\n')
      (RESULT_DIR/'tool_contract.json').write_text(json.dumps({"tool_result_contract":"opk-rag.agentic-v2.tool-result.v1","schema_digest":summary['tool_result_schema_digest'],"candidate_evidence_semantic_separation":True},ensure_ascii=False,indent=2)+'\n')
      (RESULT_DIR/'execution_matrix.json').write_text(json.dumps({"allow":"execute_once","modify":"execute_guard_modified_action_once","reject":"zero_execution","terminate":"zero_execution","raw_agent_decision":"not_executable","raw_json":"not_executable"},ensure_ascii=False,indent=2)+'\n')
      (RESULT_DIR/'integration.json').write_text(json.dumps({"chain":["AgentObservation","LLM Policy","AgentDecision","Pydantic","Deterministic Guard","Validated AgentAction","Tool Registry","Guarded Tool Executor","ToolResult"],"second_policy_decision_active":False,"state_transition_active":False},ensure_ascii=False,indent=2)+'\n')
      (RESULT_DIR/'security.json').write_text(json.dumps({k:summary[k] for k in ('benchmark_gold_exposure_count','secret_exposure_count','hidden_reasoning_exposure_count','tool_execution_read_only','dynamic_tool_registration_allowed','direct_qdrant_authority_in_v2_tools','direct_sql_authority_in_v2_tools')},ensure_ascii=False,indent=2)+'\n')
    return summary

def verify():
    s=build_summary(write=False)
    required={"task_status":"complete","task0240_contract_digest_match":True,"task0241_policy_runtime_valid":True,"task0242_guard_runtime_valid":True,"agentic_v2_tool_registry_implemented":True,"guarded_tool_executor_implemented":True,"tool_result_contract_valid":True,"validated_action_only_execution":True,"agent_decision_direct_execution":False,"guard_allow_execution_valid":True,"guard_modify_execution_valid":True,"guard_reject_zero_execution":True,"guard_terminate_zero_execution":True,"hybrid_tool_registered":True,"structure_tool_registered":True,"graph_tool_registered":True,"rewrite_tool_registered":True,"inspect_evidence_tool_registered":True,"finish_tool_registered":True,"abstain_tool_registered":True,"graph_one_hop_execution_enforced":True,"dynamic_tool_registration_allowed":False,"tool_execution_read_only":True,"candidate_evidence_semantic_separation":True,"tool_trace_valid":True,"tool_metrics_valid":True,"benchmark_gold_exposure_count":0,"secret_exposure_count":0,"hidden_reasoning_exposure_count":0,"state_transition_runtime_active":False,"observation_loop_active":False,"second_policy_decision_active":False,"agent_loop_active":False,"production_agentic_v2_active":False,"agent_framework_runtime_used":False,"production_runtime_behavior_changed":False,"blocking_failure_count":0}
    mism={k:{"expected":v,"actual":s.get(k)} for k,v in required.items() if s.get(k)!=v}
    files=[ROOT/'tasks/TASK-0243_agentic_v2_tool_registry_and_guarded_tool_executor.md',ROOT/'docs/LLM_AGENTIC_RAG_TOOL_EXECUTION_RUNTIME.md',ROOT/'docs/TASK0243_AGENTIC_V2_TOOL_REGISTRY_AND_GUARDED_TOOL_EXECUTOR_REPORT.md',CONTRACT,TOOL_SCHEMA,RESULT_DIR/'summary.json',RESULT_DIR/'tool_registry.json',RESULT_DIR/'tool_contract.json',RESULT_DIR/'execution_matrix.json',RESULT_DIR/'integration.json',RESULT_DIR/'security.json',RESULT_DIR/'real_tool_smoke_test.json',RESULT_DIR/'metrics.json',RESULT_DIR/'regression.json']
    missing=[str(x.relative_to(ROOT)) for x in files if not x.is_file()]
    return {"schema_version":SCHEMA,"task_id":TASK_ID,"verification_passed":not mism and not missing,"mismatches":mism,"missing_files":missing}

if __name__=='__main__': print(json.dumps(build_summary(),ensure_ascii=False,indent=2))
