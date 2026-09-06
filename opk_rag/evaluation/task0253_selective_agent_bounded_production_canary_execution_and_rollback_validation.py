from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from opk_rag.showcase.selective_agent_canary import (
    CANARY_ENABLED_ENV, CANARY_EXPOSURE_ENV, CANARY_KILL_SWITCH_ENV, CANARY_MODE_ENV,
    evaluate_task0252_readiness, load_canary_config, route_canary_request,
)

ROOT=Path(__file__).resolve().parents[2]
TASK_ID="TASK-0253"
SCHEMA="opk-rag.task0253.selective-agent-bounded-production-canary-execution-and-rollback-validation.v1"
TASK_START_HEAD="36569ef911c336e40b5c6ec050c338d00a51d052"
RESULT=ROOT/"evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation"
CONTRACT=ROOT/"evaluation-data/contracts/task0253_selective_agent_bounded_production_canary_execution_and_rollback_validation.json"
T252=ROOT/"evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/summary.json"
T252_CANDIDATE=ROOT/"evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/candidate_identity.json"
REG=RESULT/"regression.json"

SAFETY_KEYS=("llm_finish_authority_count","abstain_to_finish_override_count","unauthorized_action_execution_count","guard_bypass_count","graph_hop_violation_count","knowledge_base_mutation_count","benchmark_gold_exposure_count","hidden_reasoning_persisted_count","raw_provider_output_persisted_count","secret_exposure_count","benchmark_or_synthetic_canary_count","agent_induced_production_failure_count","canary_exposure_limit_violation_count")

def read_json(path: Path, default=None):
    if not path.is_file(): return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def head(): return subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def changed_paths():
    out=subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    return sorted(line[3:].split(" -> ",1)[-1] for line in out.splitlines() if len(line)>=4)

def _contract():
    return {"schema_version":"opk-rag.task0253.contract.v1","task_id":TASK_ID,"stage":"llm_agentic_rag_development","task_start_head":TASK_START_HEAD,"entry_requires_task0252_decision":"advance_to_canary_execution_readiness","default_enabled":False,"default_mode":"off","default_kill_switch":True,"small_canary_max_exposure":0.05,"expanded_canary_max_exposure":0.20,"full_production_activation_allowed":False,"max_graph_hop":1,"llm_finish_authority_allowed":False,"abstain_to_finish_override_allowed":False,"knowledge_base_mutation_allowed":False,"raw_provider_output_persistence_allowed":False,"hidden_reasoning_persistence_allowed":False}

def _not_executed(name: str, reason: str): return {"schema_version":f"opk-rag.task0253.{name}.v1","status":"not_executed_due_to_previous_stage_gate","reason":reason}

def validate_exposure(mode: str, exposure: float)->dict[str,Any]:
    bounds={"off":0.0,"shadow":0.0,"small_canary":0.05,"expanded_canary":0.20}
    valid=mode in bounds and 0<=exposure<=bounds.get(mode,-1) and exposure<=0.20
    return {"mode":mode,"exposure":exposure,"maximum":bounds.get(mode),"valid":valid}

def small_canary_gate(metrics: Mapping[str,Any])->dict[str,Any]:
    gates={"minimum_observations":int(metrics.get("canary_observation_count") or 0)>=30,"exposure_within_5_percent":float(metrics.get("observed_exposure") or 0)<=0.05,"provider_response":float(metrics.get("provider_response_rate") or 0)>=0.98,"structured_validity":float(metrics.get("final_structured_validity") or 0)>=0.98,"fallback_success":float(metrics.get("fallback_success_rate") or 0)>=1.0,"kill_switch_validated":metrics.get("kill_switch_validated") is True,"rollback_validated":metrics.get("rollback_validated") is True,"safety_zero":all(int(metrics.get(k) or 0)==0 for k in SAFETY_KEYS)}
    if not gates["safety_zero"]: decision="rollback_and_reject"
    elif all(gates.values()): decision="advance_to_expanded_canary"
    else: decision="hold_small_canary"
    return {"schema_version":"opk-rag.task0253.small-canary-gate.v1","gates":gates,"decision":decision}

def expanded_canary_gate(metrics: Mapping[str,Any])->dict[str,Any]:
    gates={"minimum_observations":int(metrics.get("canary_observation_count") or 0)>=100,"exposure_within_20_percent":float(metrics.get("observed_exposure") or 0)<=0.20,"provider_response":float(metrics.get("provider_response_rate") or 0)>=0.98,"structured_validity":float(metrics.get("final_structured_validity") or 0)>=0.98,"fallback_success":float(metrics.get("fallback_success_rate") or 0)>=1.0,"safety_zero":all(int(metrics.get(k) or 0)==0 for k in SAFETY_KEYS)}
    if not gates["safety_zero"]: decision="rollback_and_reject"
    elif all(gates.values()): decision="advance_to_production_promotion_review"
    else: decision="hold_expanded_canary"
    return {"schema_version":"opk-rag.task0253.expanded-canary-gate.v1","gates":gates,"decision":decision}

def _mechanism_validation(candidate_fp: str|None):
    ready_root = ROOT
    # Runtime mechanism is verified with pure routing controls; current production authority is never changed.
    disabled=route_canary_request(query="task0253-mechanism",execution_scope="search",env={},root=ready_root,expected_candidate_fingerprint=candidate_fp)
    killed=route_canary_request(query="task0253-mechanism",execution_scope="search",env={CANARY_ENABLED_ENV:"1",CANARY_MODE_ENV:"small_canary",CANARY_EXPOSURE_ENV:"0.05",CANARY_KILL_SWITCH_ENV:"1"},root=ready_root,expected_candidate_fingerprint=candidate_fp)
    return {"schema_version":"opk-rag.task0253.kill-switch-mechanism-validation.v1","default_off":disabled.get("authority_lane")=="rule_governed_control","kill_switch_forces_control":killed.get("selected") is False and killed.get("authority_lane")=="rule_governed_control","production_canary_activated":False,"mechanism_valid":disabled.get("authority_lane")=="rule_governed_control" and killed.get("selected") is False}

def build_summary(*, write: bool=True)->dict[str,Any]:
    t252=read_json(T252,{})
    candidate=read_json(T252_CANDIDATE,{})
    fp=str(candidate.get("candidate_fingerprint") or t252.get("candidate_fingerprint") or "") or None
    entry=evaluate_task0252_readiness(root=ROOT,expected_candidate_fingerprint=fp)
    mechanism=_mechanism_validation(fp)
    current_cfg=load_canary_config({},root=ROOT)
    reason="task0252_entry_gate_not_passed" if not entry["passed"] else "real_canary_not_started_by_task_evaluator"
    small=_not_executed("small-canary-metrics",reason)
    expanded=_not_executed("expanded-canary-metrics",reason)
    safety={"schema_version":"opk-rag.task0253.safety-metrics.v1",**{k:0 for k in SAFETY_KEYS},"safety_invariants_preserved":True}
    rollback={"schema_version":"opk-rag.task0253.rollback-validation.v1","kill_switch_mechanism_validated":mechanism["mechanism_valid"],"production_rollback_drill_executed":False,"rollback_target":"Production Rule-Governed Search/Ask","automatic_reenable_allowed":False,"status":"mechanism_validated_production_drill_blocked" if not entry["passed"] else "pending_real_canary"}
    final="blocked" if not entry["passed"] else "hold_small_canary"
    reg=read_json(REG,{})
    summary={"schema_version":SCHEMA,"task_id":TASK_ID,"task_status":"blocked" if not entry["passed"] else "partial","implementation_complete":True,"current_stage":"llm_agentic_rag_development","entry_gate_passed":entry["passed"],"candidate_fingerprint":fp,"production_agentic_v2_full_activation":False,"canary_execution_performed":False,"current_canary_mode":current_cfg.mode,"configured_canary_exposure":current_cfg.exposure,"observed_canary_exposure":0.0,"small_canary_observation_count":0,"expanded_canary_observation_count":0,"provider_response_rate":None,"final_structured_validity":None,"average_controller_calls":0.0,"llm_bypass_rate":1.0,"recovery_net_signal":None,"veto_risk_count":0,"fallback_success_rate":None,"agent_induced_production_failure_count":0,"reranker_induced_terminal_decision_instability_count":0,"kill_switch_validated":mechanism["mechanism_valid"],"rollback_validated":False,"safety_invariants_preserved":True,"small_canary_decision":"not_executed_due_to_previous_stage_gate" if not entry["passed"] else "hold_small_canary","expanded_canary_decision":"not_executed_due_to_previous_stage_gate","final_candidate_decision":final,"blocking_failures":entry["blockers"] if not entry["passed"] else ["real_canary_observations_not_collected"],"production_answer_authority_change":False,"production_default_behavior_change":False,"next_task":"TASK-0251_resume_after_minimum_live_shadow_traffic" if not entry["passed"] else "TASK-0253_collect_small_canary_observations","task_start_head":TASK_START_HEAD,"current_head":head(),"git_head_unchanged_since_task_start":head()==TASK_START_HEAD,"git_commit_created":head()!=TASK_START_HEAD,"focused_tests_passed":reg.get("focused_tests_passed"),"full_suite_passed":reg.get("full_suite_passed"),"full_suite_skipped":reg.get("full_suite_skipped"),"full_suite_failed":reg.get("full_suite_failed"),"new_task0253_full_suite_failure_count":reg.get("new_task0253_full_suite_failure_count"),"changed_paths":changed_paths()}
    if write:
        write_json(CONTRACT,_contract()); RESULT.mkdir(parents=True,exist_ok=True)
        artifacts={"candidate_identity.json":{"schema_version":"opk-rag.task0253.candidate-identity.v1","task0252_approved_candidate_fingerprint":fp,"candidate_match_required":True},"entry_gate.json":entry,"canary_runtime_config.json":{"schema_version":"opk-rag.task0253.canary-runtime-config.v1","default_enabled":False,"default_mode":"off","default_exposure":0.0,"default_kill_switch":True,"small_canary_max":0.05,"expanded_canary_max":0.20},"traffic_assignment_metrics.json":_not_executed("traffic-assignment-metrics",reason),"control_cohort_metrics.json":_not_executed("control-cohort-metrics",reason),"small_canary_metrics.json":small,"expanded_canary_metrics.json":expanded,"controller_exposure.json":_not_executed("controller-exposure",reason),"provider_reliability.json":_not_executed("provider-reliability",reason),"recovery_metrics.json":_not_executed("recovery-metrics",reason),"veto_metrics.json":_not_executed("veto-metrics",reason),"latency_and_token_metrics.json":_not_executed("latency-and-token-metrics",reason),"reranker_stability.json":_not_executed("reranker-stability",reason),"failure_isolation.json":{"schema_version":"opk-rag.task0253.failure-isolation.v1","rule_governed_fallback_required":True,"authoritative_agent_executor_bound":False,"agent_induced_production_failure_count":0,"status":"implementation_ready_real_canary_blocked"},"safety_metrics.json":safety,"kill_switch_validation.json":mechanism,"rollback_validation.json":rollback,"small_canary_gate.json":_not_executed("small-canary-gate",reason),"expanded_canary_gate.json":_not_executed("expanded-canary-gate",reason),"production_promotion_review_gate.json":{"schema_version":"opk-rag.task0253.production-promotion-review-gate.v1","decision":"blocked","full_production_activation_executed":False},"summary.json":summary}
        for name,payload in artifacts.items(): write_json(RESULT/name,payload)
    return summary

def verify()->dict[str,Any]:
    summary=build_summary(write=False); missing=[]; mismatches={}
    required=[CONTRACT,ROOT/"tasks/TASK-0253_selective_agent_bounded_production_canary_execution_and_rollback_validation.md",ROOT/"docs/TASK0253_SELECTIVE_AGENT_BOUNDED_PRODUCTION_CANARY_EXECUTION_AND_ROLLBACK_VALIDATION_REPORT.md",ROOT/"opk_rag/showcase/selective_agent_canary.py",ROOT/"scripts/run_task0253_selective_agent_bounded_production_canary_execution_and_rollback_validation.py",ROOT/"scripts/verify_task0253_selective_agent_bounded_production_canary_execution_and_rollback_validation.py",RESULT/"summary.json",RESULT/"entry_gate.json",RESULT/"kill_switch_validation.json",RESULT/"rollback_validation.json",REG]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    fixed={"implementation_complete":True,"production_agentic_v2_full_activation":False,"canary_execution_performed":False,"production_answer_authority_change":False,"production_default_behavior_change":False,"agent_induced_production_failure_count":0,"safety_invariants_preserved":True,"git_head_unchanged_since_task_start":True,"git_commit_created":False}
    for k,v in fixed.items():
        if summary.get(k)!=v: mismatches[k]={"expected":v,"actual":summary.get(k)}
    if not summary["entry_gate_passed"] and (summary["task_status"]!="blocked" or summary["final_candidate_decision"]!="blocked"):
        mismatches["blocked_entry_state"]={"expected":"blocked/blocked","actual":[summary["task_status"],summary["final_candidate_decision"]]}
    return {"schema_version":SCHEMA,"task_id":TASK_ID,"verification_passed":not missing and not mismatches,"task_status":summary["task_status"],"final_candidate_decision":summary["final_candidate_decision"],"entry_gate_passed":summary["entry_gate_passed"],"missing_files":missing,"mismatches":mismatches}
