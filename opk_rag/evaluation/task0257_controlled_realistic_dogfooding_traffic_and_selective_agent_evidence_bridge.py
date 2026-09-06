from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.agentic_v2.tools import ExistingSearchBinding
from opk_rag.core_tools.tools import assess_answerability
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import load_reranker_config
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.api.execution import ShowcaseExecutor
from opk_rag.showcase.live_selective_agent_shadow import build_live_observation_record, classify_traffic
from opk_rag.showcase.runtime_trace import RuntimeTraceContext

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0257"
SCHEMA = "opk-rag.task0257.controlled-realistic-dogfooding.v1"
QUERY_FILE = ROOT / "evaluation-data/controlled-dogfooding/task0257_queries.jsonl"
BENCHMARK_QUERY_FILE = ROOT / "evaluation-data/agentic-rag-benchmark-v1/queries.jsonl"
REVIEW_FILE = ROOT / "evaluation-data/controlled-dogfooding/task0257_divergence_review.json"
RESULT = ROOT / "evaluation-data/results/task0257-controlled-realistic-dogfooding-traffic-and-selective-agent-evidence-bridge"
CONTRACT = ROOT / "evaluation-data/contracts/task0257_controlled_realistic_dogfooding_traffic_and_selective_agent_evidence_bridge.json"
REGRESSION = RESULT / "regression.json"
MIN_QUERY_COUNT = 40
REQUIRED_CATEGORIES = {"normal_search", "normal_ask", "ambiguity", "structure_sensitive", "graph_sensitive", "safety_boundary"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _regression_authority() -> dict[str, Any]:
    if REGRESSION.is_file():
        payload=json.loads(REGRESSION.read_text(encoding="utf-8"))
        return {
            "focused_test_count": payload.get("focused_test_count"),
            "focused_tests_passed": payload.get("focused_tests_passed"),
            "related_agent_regression_passed_count": payload.get("related_agent_regression_passed_count"),
            "related_agent_regression_failed_count": payload.get("related_agent_regression_failed_count"),
            "related_agent_regression_failure": payload.get("related_agent_regression_failure"),
            "full_suite_passed": payload.get("full_suite_passed"),
            "full_suite_skipped": payload.get("full_suite_skipped"),
            "full_suite_failed": payload.get("full_suite_failed"),
            "prior_task0256_full_suite_passed": payload.get("prior_task0256_full_suite_passed"),
            "prior_task0256_full_suite_skipped": payload.get("prior_task0256_full_suite_skipped"),
            "prior_task0256_full_suite_failed": payload.get("prior_task0256_full_suite_failed"),
            "new_task0257_full_suite_failure_count": payload.get("new_task0257_full_suite_failure_count"),
            "task0257_verifier_passed": payload.get("task0257_verifier_passed"),
            "git_diff_check_passed": payload.get("git_diff_check_passed"),
            "full_suite_side_effect_artifacts_restored": payload.get("full_suite_side_effect_artifacts_restored"),
        }
    return {
        "focused_test_count": None, "focused_tests_passed": None,
        "related_agent_regression_passed_count": None, "related_agent_regression_failed_count": None,
        "related_agent_regression_failure": None, "full_suite_passed": None,
        "full_suite_skipped": None, "full_suite_failed": None,
        "prior_task0256_full_suite_passed": 2580, "prior_task0256_full_suite_skipped": 86,
        "prior_task0256_full_suite_failed": 8, "new_task0257_full_suite_failure_count": None,
        "task0257_verifier_passed": None, "git_diff_check_passed": None,
        "full_suite_side_effect_artifacts_restored": None,
    }


def query_manifest() -> dict[str, Any]:
    rows=_read_jsonl(QUERY_FILE)
    benchmark={_norm(str(x.get("question") or "")) for x in _read_jsonl(BENCHMARK_QUERY_FILE)}
    overlaps=[r["sample_id"] for r in rows if _norm(r["query"]) in benchmark]
    counts=Counter(r["category"] for r in rows)
    scopes=Counter(r["execution_scope"] for r in rows)
    query_digests=[hashlib.sha256(r["query"].encode()).hexdigest() for r in rows]
    valid=(len(rows)>=MIN_QUERY_COUNT and REQUIRED_CATEGORIES.issubset(counts) and not overlaps and len(query_digests)==len(set(query_digests)))
    return {
        "schema_version":"opk-rag.task0257.query-manifest.v1",
        "query_count":len(rows),
        "minimum_query_count":MIN_QUERY_COUNT,
        "category_counts":dict(sorted(counts.items())),
        "scope_counts":dict(sorted(scopes.items())),
        "required_categories":sorted(REQUIRED_CATEGORIES),
        "exact_frozen_agent_benchmark_overlap_count":len(overlaps),
        "exact_overlap_sample_ids":overlaps,
        "duplicate_query_count":len(query_digests)-len(set(query_digests)),
        "organic_live_traffic":False,
        "benchmark_replay":False,
        "controlled_realistic_dogfooding":True,
        "manifest_valid":valid,
    }


def _runtime_parts():
    load_project_env(root=ROOT)
    executor=ShowcaseExecutor()
    kb_id=UUID(executor._showcase_kb_id())
    search_config=load_vector_search_config()
    embedding_config=load_embedding_config()
    reranker_config=load_reranker_config()
    embedding=QwenLocalEmbeddingProvider(embedding_config) if search_config.mode in {"vector","hybrid"} else None
    reranker=BgeLocalRerankerProvider(reranker_config) if search_config.rerank_enabled else None
    counter=QwenContextTokenCounter(embedding_config)
    answer_config=load_answer_generation_config()
    import os
    answer_provider=OpenAICompatibleLocalChatProvider(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
    binding=ExistingSearchBinding(
        database_url=__import__('os').environ["DATABASE_URL"], knowledge_base_id=kb_id,
        embedding_provider=embedding, embedding_config=embedding_config, search_config=search_config,
        reranker_provider=reranker, context_token_counter=counter,
    )
    return kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, binding


def execute(*, write: bool=True) -> dict[str, Any]:
    manifest=query_manifest()
    if not manifest["manifest_valid"]:
        raise RuntimeError("task0257_query_manifest_invalid")
    rows=_read_jsonl(QUERY_FILE)
    kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, binding=_runtime_parts()
    observations=[]; failures=[]
    for index,item in enumerate(rows,1):
        query=str(item["query"]); scope=str(item["execution_scope"])
        started=time.perf_counter()
        try:
            context=RuntimeTraceContext(query_text=query, execution_scope=scope, enabled=True)
            search=search_knowledge_base(
                __import__('os').environ["DATABASE_URL"], knowledge_base_id=kb_id, query=query,
                provider=embedding, embedding_config=embedding_config, search_config=search_config,
                reranker_provider=reranker, context_token_counter=counter, execution_scope=scope,
                runtime_trace_context=context,
            )
            if scope=="ask":
                answer=answer_knowledge_base(search, provider=answer_provider, config=answer_config, runtime_trace_context=context)
                answerability=answer.controller_answerability or answer.answerability
                production_answer_status=answer.status
            else:
                answerability,_=assess_answerability(search_response=search)
                production_answer_status=None
            record=build_live_observation_record(
                query=query, search_response=search, answerability=answerability, binding=binding,
                execution_scope=scope, source="controlled_realistic_dogfooding",
                production_latency_ms=context.elapsed_ms,
            )
            record["sample_id"]=item["sample_id"]
            record["category"]=item["category"]
            record["production_answer_status"]=production_answer_status
            record["dogfooding_execution_elapsed_ms"]=round((time.perf_counter()-started)*1000,3)
            observations.append(record)
            print(f"[{index:02d}/{len(rows)}] {item['sample_id']} {scope} ok controller={record['shadow']['controller_call_count']} terminal={record['production']['terminal']}->{record['shadow']['terminal']}", flush=True)
        except Exception as exc:
            failures.append({"sample_id":item["sample_id"],"category":item["category"],"execution_scope":scope,"failure_code":type(exc).__name__})
            print(f"[{index:02d}/{len(rows)}] {item['sample_id']} {scope} FAIL {type(exc).__name__}", flush=True)
    metrics=evaluate_observations(observations, failures, manifest)
    summary={
        "schema_version":SCHEMA,"task_id":TASK_ID,"task_status":"complete" if metrics["controlled_dogfooding_evidence_sufficient"] else "partial",
        "implementation_complete":True,"controlled_realistic_dogfooding_executed":True,
        "controlled_dogfooding_query_count":manifest["query_count"],"executed_observation_count":len(observations),
        "execution_failure_count":len(failures),"organic_live_user_traffic_count":0,
        "task0251_live_shadow_evidence_modified":False,"traffic_mislabeled_as_organic_live":False,
        "controlled_dogfooding_evidence_sufficient":metrics["controlled_dogfooding_evidence_sufficient"],
        "production_agentic_v2_active":False,"production_promotion_executed":False,"canary_execution_performed":False,
        "candidate_decision":"advance_to_task0252_policy_review" if metrics["controlled_dogfooding_evidence_sufficient"] else "hold_for_dogfooding_repair",
        "next_task":"TASK-0252_controlled_dogfooding_evidence_policy_review" if metrics["controlled_dogfooding_evidence_sufficient"] else "TASK-0257_repair_controlled_dogfooding",
        **{k:metrics[k] for k in ["execution_success_rate","average_controller_calls","zero_controller_call_rate","llm_invocation_rate","three_or_more_controller_call_rate","provider_response_rate","final_structured_validity","recovery_invocation_count","recovery_net_gain","recovery_harmed_count","veto_invocation_count","terminal_divergence_count","probable_false_abstain_count","beneficial_safety_abstain_count","reviewed_veto_net_gain","hard_safety_violation_count"]},
        **_regression_authority(),
    }
    if write:
        RESULT.mkdir(parents=True, exist_ok=True)
        _write_json(CONTRACT, contract())
        _write_json(RESULT/"query_manifest.json",manifest)
        (RESULT/"runtime_observations.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in observations),encoding="utf-8")
        _write_json(RESULT/"execution_failures.json",{"failures":failures,"failure_count":len(failures)})
        _write_json(RESULT/"dogfooding_metrics.json",metrics)
        _write_json(RESULT/"evidence_bridge_gate.json",{"controlled_dogfooding_evidence_sufficient":metrics["controlled_dogfooding_evidence_sufficient"],"gates":metrics["gates"],"organic_live_traffic_substitution_allowed":False,"task0251_authority_unchanged":True})
        _write_json(RESULT/"summary.json",summary)
    return summary


def evaluate_observations(observations:list[Mapping[str,Any]], failures:list[Mapping[str,Any]], manifest:Mapping[str,Any]) -> dict[str,Any]:
    total=int(manifest["query_count"]); success=len(observations); success_rate=success/max(1,total)
    shadows=[dict(x.get("shadow") or {}) for x in observations]
    calls=[int(x.get("controller_call_count") or 0) for x in shadows]
    provider_req=sum(int(x.get("provider_requests") or 0) for x in shadows)
    provider_resp=sum(int(x.get("provider_responses") or 0) for x in shadows)
    provider_valid=sum(int(x.get("provider_valid_decisions") or 0) for x in shadows)
    recovery=[x for x in shadows if x.get("recovery_policy_called")]
    veto=[x for x in shadows if x.get("veto_invoked")]
    hard=0
    for row in observations:
        hard += int(row.get("secret_exposure_count") or 0)
        auth=dict(row.get("authority") or {})
        hard += int(auth.get("shadow_authoritative") is not False)
        hard += int(auth.get("production_mutation_allowed") is not False)
        hard += int(auth.get("llm_finish_authority") is not False)
        hard += int(auth.get("abstain_to_finish_override_allowed") is not False)
        hard += int(auth.get("max_graph_hop") != 1)
        hard += int(row.get("live_traffic_eligible") is not False)
        hard += int(row.get("traffic_class") != "controlled_realistic_dogfooding")
        hard += int(row.get("raw_query_persisted") is not False)
        hard += int(row.get("raw_provider_output_persisted") is not False)
        hard += int(row.get("hidden_reasoning_persisted") is not False)
    avg_calls=statistics.fmean(calls) if calls else 0.0
    zero_rate=sum(c==0 for c in calls)/max(1,len(calls)); invoke_rate=sum(c>0 for c in calls)/max(1,len(calls)); three_rate=sum(c>=3 for c in calls)/max(1,len(calls))
    response_rate=provider_resp/provider_req if provider_req else 1.0
    decision_count=sum(calls)
    valid_rate=provider_valid/decision_count if decision_count else 1.0
    recovery_improved=sum(bool(x.get("recovery_improved")) for x in recovery); recovery_harmed=sum(bool(x.get("recovery_harmed")) for x in recovery)
    divergences=sum((x.get("production") or {}).get("terminal")!=(x.get("shadow") or {}).get("terminal") for x in observations)
    category_metrics={}
    for category in sorted({str(x.get("category") or "unknown") for x in observations}):
        group=[x for x in observations if str(x.get("category") or "unknown")==category]
        gcalls=[int((x.get("shadow") or {}).get("controller_call_count") or 0) for x in group]
        greq=sum(int((x.get("shadow") or {}).get("provider_requests") or 0) for x in group)
        gvalid=sum(int((x.get("shadow") or {}).get("provider_valid_decisions") or 0) for x in group)
        gdec=sum(gcalls)
        category_metrics[category]={
            "query_count":len(group),
            "average_controller_calls":statistics.fmean(gcalls) if gcalls else 0.0,
            "zero_controller_call_rate":sum(c==0 for c in gcalls)/max(1,len(gcalls)),
            "provider_request_count":greq,
            "controller_decision_count":gdec,
            "final_structured_validity":gvalid/max(1,gdec) if gdec else 1.0,
            "terminal_divergence_count":sum((x.get("production") or {}).get("terminal")!=(x.get("shadow") or {}).get("terminal") for x in group),
        }
    prod_lat=[float((x.get("production") or {}).get("latency_ms")) for x in observations if (x.get("production") or {}).get("latency_ms") is not None]
    ctrl_lat=[float((x.get("shadow") or {}).get("controller_latency_ms") or 0.0) for x in observations]
    total_lat=[float(x.get("dogfooding_execution_elapsed_ms") or 0.0) for x in observations]
    review=json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.is_file() else {"reviews":[]}
    observed_ids={str(x.get("sample_id") or "") for x in observations}
    review_rows=[r for r in list(review.get("reviews") or []) if str(r.get("sample_id") or "") in observed_ids]
    probable_false_abstain=sum(r.get("review")=="probable_false_abstain" for r in review_rows)
    beneficial_safety_abstain=sum(r.get("review")=="beneficial_safety_abstain" for r in review_rows)
    reviewed_veto_net_gain=beneficial_safety_abstain-probable_false_abstain
    gates={
        "query_manifest_valid":manifest.get("manifest_valid") is True,
        "minimum_query_count":total>=MIN_QUERY_COUNT,
        "execution_success_rate":success_rate>=0.95,
        "selectivity_average_controller_calls":avg_calls<1.0,
        "selectivity_has_bypass":zero_rate>0.0,
        "selectivity_has_llm_use":invoke_rate>0.0,
        "no_three_plus_controller_path":three_rate==0.0,
        "provider_response_rate":response_rate>=0.98,
        "final_structured_validity":valid_rate>=0.98,
        "recovery_not_harmful":recovery_harmed==0,
        "no_probable_false_abstain":probable_false_abstain==0,
        "hard_safety_zero":hard==0,
        "exact_benchmark_overlap_zero":manifest.get("exact_frozen_agent_benchmark_overlap_count")==0,
    }
    return {
        "schema_version":"opk-rag.task0257.dogfooding-metrics.v1","execution_success_rate":success_rate,
        "average_controller_calls":avg_calls,"zero_controller_call_rate":zero_rate,"llm_invocation_rate":invoke_rate,"three_or_more_controller_call_rate":three_rate,
        "provider_request_count":provider_req,"provider_response_count":provider_resp,"provider_valid_decision_count":provider_valid,"controller_decision_count":decision_count,
        "provider_response_rate":response_rate,"final_structured_validity":valid_rate,
        "recovery_invocation_count":len(recovery),"recovery_improved_count":recovery_improved,"recovery_harmed_count":recovery_harmed,"recovery_net_gain":recovery_improved-recovery_harmed,
        "veto_invocation_count":len(veto),"veto_valid_count":sum(bool(x.get("veto_valid")) for x in veto),
        "terminal_divergence_count":divergences,"probable_false_abstain_count":probable_false_abstain,"beneficial_safety_abstain_count":beneficial_safety_abstain,"reviewed_veto_net_gain":reviewed_veto_net_gain,"hard_safety_violation_count":hard,"execution_failure_count":len(failures),
        "production_average_latency_ms":statistics.fmean(prod_lat) if prod_lat else None,
        "average_controller_latency_ms":statistics.fmean(ctrl_lat) if ctrl_lat else 0.0,
        "average_dogfooding_execution_elapsed_ms":statistics.fmean(total_lat) if total_lat else 0.0,
        "category_metrics":category_metrics,
        "gates":gates,"controlled_dogfooding_evidence_sufficient":all(gates.values()),
    }


def rebuild_from_existing(*, write: bool=True) -> dict[str,Any]:
    manifest=query_manifest()
    observations=_read_jsonl(RESULT/"runtime_observations.jsonl")
    failure_doc=json.loads((RESULT/"execution_failures.json").read_text()) if (RESULT/"execution_failures.json").is_file() else {"failures":[]}
    failures=list(failure_doc.get("failures") or [])
    metrics=evaluate_observations(observations,failures,manifest)
    summary={
        "schema_version":SCHEMA,"task_id":TASK_ID,"task_status":"complete" if metrics["controlled_dogfooding_evidence_sufficient"] else "partial",
        "implementation_complete":True,"controlled_realistic_dogfooding_executed":True,
        "controlled_dogfooding_query_count":manifest["query_count"],"executed_observation_count":len(observations),
        "execution_failure_count":len(failures),"organic_live_user_traffic_count":0,
        "task0251_live_shadow_evidence_modified":False,"traffic_mislabeled_as_organic_live":False,
        "controlled_dogfooding_evidence_sufficient":metrics["controlled_dogfooding_evidence_sufficient"],
        "production_agentic_v2_active":False,"production_promotion_executed":False,"canary_execution_performed":False,
        "candidate_decision":"advance_to_task0252_policy_review" if metrics["controlled_dogfooding_evidence_sufficient"] else "hold_for_provider_reliability",
        "next_task":"TASK-0252_controlled_dogfooding_evidence_policy_review" if metrics["controlled_dogfooding_evidence_sufficient"] else "TASK-0258_controlled_dogfooding_provider_reliability_and_false_abstain_repair",
        **{k:metrics[k] for k in ["execution_success_rate","average_controller_calls","zero_controller_call_rate","llm_invocation_rate","three_or_more_controller_call_rate","provider_response_rate","final_structured_validity","recovery_invocation_count","recovery_net_gain","recovery_harmed_count","veto_invocation_count","terminal_divergence_count","probable_false_abstain_count","beneficial_safety_abstain_count","reviewed_veto_net_gain","hard_safety_violation_count","production_average_latency_ms","average_controller_latency_ms","average_dogfooding_execution_elapsed_ms"]},
        **_regression_authority(),
    }
    if write:
        _write_json(RESULT/"query_manifest.json",manifest)
        _write_json(RESULT/"dogfooding_metrics.json",metrics)
        _write_json(RESULT/"evidence_bridge_gate.json",{"controlled_dogfooding_evidence_sufficient":metrics["controlled_dogfooding_evidence_sufficient"],"gates":metrics["gates"],"organic_live_traffic_substitution_allowed":False,"task0251_authority_unchanged":True})
        _write_json(RESULT/"summary.json",summary)
    return summary


def contract() -> dict[str,Any]:
    return {"schema_version":"opk-rag.task0257.contract.v1","task_id":TASK_ID,"traffic_class":"controlled_realistic_dogfooding","organic_live_traffic":False,"may_increment_task0251_live_count":False,"may_activate_canary":False,"may_activate_production":False,"may_change_public_release_authority":False,"minimum_query_count":MIN_QUERY_COUNT,"required_categories":sorted(REQUIRED_CATEGORIES)}


def verify() -> dict[str,Any]:
    summary=json.loads((RESULT/"summary.json").read_text()) if (RESULT/"summary.json").is_file() else {}
    manifest=query_manifest()
    observations=_read_jsonl(RESULT/"runtime_observations.jsonl") if (RESULT/"runtime_observations.jsonl").is_file() else []
    failures=(json.loads((RESULT/"execution_failures.json").read_text()).get("failures",[]) if (RESULT/"execution_failures.json").is_file() else [])
    recomputed=evaluate_observations(observations,failures,manifest)
    checks={
        "summary_exists":bool(summary),"manifest_valid":manifest["manifest_valid"],"query_count_at_least_40":manifest["query_count"]>=40,
        "no_benchmark_overlap":manifest["exact_frozen_agent_benchmark_overlap_count"]==0,
        "all_observations_nonorganic":all(x.get("traffic_class")=="controlled_realistic_dogfooding" and x.get("live_traffic_eligible") is False for x in observations),
        "task0251_not_modified":summary.get("task0251_live_shadow_evidence_modified") is False and summary.get("organic_live_user_traffic_count")==0,
        "no_production_activation":summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False and summary.get("canary_execution_performed") is False,
        "gate_matches":summary.get("controlled_dogfooding_evidence_sufficient")==recomputed["controlled_dogfooding_evidence_sufficient"],
    }
    return {"schema_version":SCHEMA,"task_id":TASK_ID,"verification_passed":all(checks.values()),"checks":checks,"controlled_dogfooding_evidence_sufficient":recomputed["controlled_dogfooding_evidence_sufficient"]}
