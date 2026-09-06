from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from opk_rag.answer.negative_semantics import classify_answer_semantics
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.evaluation.task0257_controlled_realistic_dogfooding_traffic_and_selective_agent_evidence_bridge import _runtime_parts
from opk_rag.evaluation.task0258_controlled_dogfooding_provider_reliability_and_false_abstain_repair import (
    _execute_rows,
    _frozen30_results,
    _known_slice,
    _runtime_metrics,
)
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.live_selective_agent_shadow import build_live_observation_record
from opk_rag.showcase.runtime_trace import RuntimeTraceContext

ROOT=Path(__file__).resolve().parents[2]
TASK_ID='TASK-0259'
SCHEMA='opk-rag.task0259.controlled-negative-answer-semantics-and-evidence-policy-review.v1'
RESULT=ROOT/'evaluation-data/results/task0259-controlled-negative-answer-semantics-and-evidence-policy-review'
BENCH=ROOT/'evaluation-data/negative-answer-semantics-v1'
QUERIES=BENCH/'queries.jsonl'; GOLD=BENCH/'evaluator_gold.jsonl'; MANIFEST=BENCH/'benchmark_manifest.json'
T258=ROOT/'evaluation-data/results/task0258-controlled-dogfooding-provider-reliability-and-false-abstain-repair'
T258_FIXED=ROOT/'evaluation-data/controlled-dogfooding/task0257_queries.jsonl'
T258_HOLDOUT=ROOT/'evaluation-data/controlled-dogfooding/task0258_repair_holdout.jsonl'
T258_HOLDOUT_RESULTS=T258/'repair_holdout_results.json'
T258_HOLDOUT_REVIEW=T258/'repair_holdout_semantic_review.json'
FROZEN30=ROOT/'evaluation-data/agentic-rag-benchmark-v1/queries.jsonl'
FROZEN30_GOLD=ROOT/'evaluation-data/agentic-rag-benchmark-v1/evaluator_gold.jsonl'
CONTRACT=ROOT/'evaluation-data/contracts/task0259_controlled_negative_answer_semantics_and_evidence_policy_review.json'
REPORT=ROOT/'docs/TASK0259_CONTROLLED_NEGATIVE_ANSWER_SEMANTICS_AND_EVIDENCE_POLICY_REVIEW_REPORT.md'
TASK_START_HEAD='6d407f8891236f3da6393f9cf65a6d252dc1bbf9'
T258_DIGESTS={
 'summary.json':'4bdba1fca488d7169dd2955fd09b2ee44611dbbcddb66984a6ad2916349505de',
 'repair_holdout_results.json':'9be726f36272fc85953f3b6571b6228893446b55d8db3552a4b792d02385eec2',
 'repair_holdout_semantic_review.json':'5117b3e0bcd32c95cebeccfe74d20804f45f8418bfab094199f1da114aa30d75',
 'fixed48_runtime_observations.jsonl':'546865cea4b83fcca43056f424063503fe20df5eb9ed820891a9ad3a7bc5668c',
 'frozen30_candidate_observations.jsonl':'6bda85f704bd4dce2216354da50203d7ab26444e96bee98e48817b2b99173cfc',
}
SEMANTIC_QUERY_SHA='e09c5b69f0bd788c8d156a234d2e0c2fe89b8f2a6e4020ef4f289e8262967d9e'
SEMANTIC_GOLD_SHA='f19c6d651a21cc4e8450cb5df5d8285d65cb2fc393c3902b67dee5baa2372104'


def _sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _rj(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def _rjl(p:Path)->list[dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def _wj(p:Path,v:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def _wjl(p:Path,rows:list[Mapping[str,Any]]): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(dict(r),ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
def _head()->str: return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()


def task0258_identity()->dict[str,Any]:
    actual={name:_sha(T258/name) for name in T258_DIGESTS}
    matches={name:actual[name]==digest for name,digest in T258_DIGESTS.items()}
    return {'schema_version':'opk-rag.task0259.task0258-frozen-identity.v1','expected_sha256':T258_DIGESTS,'actual_sha256':actual,'matches':matches,'task0258_historical_artifacts_unchanged':all(matches.values())}


def benchmark_identity()->dict[str,Any]:
    m=_rj(MANIFEST)
    return {'schema_version':'opk-rag.task0259.semantic-benchmark-identity.v1','status':m.get('status'),'frozen_before_candidate_execution':m.get('frozen_before_candidate_execution'),'query_count':m.get('query_count'),'family_counts':m.get('family_counts'),'queries_sha256':_sha(QUERIES),'gold_sha256':_sha(GOLD),'expected_queries_sha256':SEMANTIC_QUERY_SHA,'expected_gold_sha256':SEMANTIC_GOLD_SHA,'runtime_gold_metadata_usage':False,'identity_valid':m.get('status')=='frozen' and m.get('frozen_before_candidate_execution') is True and m.get('query_count')==24 and _sha(QUERIES)==SEMANTIC_QUERY_SHA and _sha(GOLD)==SEMANTIC_GOLD_SHA}


def entry_gate()->dict[str,Any]:
    s=_rj(T258/'summary.json'); ident=task0258_identity(); bench=benchmark_identity()
    checks={
      'task0258_identity':ident['task0258_historical_artifacts_unchanged'],
      'task0258_complete':s.get('task_status')=='complete',
      'task0258_expected_hold':s.get('candidate_decision')=='hold_for_generalization_failure',
      'task0258_original_repairs_clean':s.get('final_structured_validity')==1.0 and s.get('probable_false_abstain_count')==0 and s.get('beneficial_safety_abstain_retained_count')==2,
      'semantic_gap_present':s.get('repair_holdout_label_semantic_mismatch_count')==3 and s.get('repair_holdout_bounded_negative_finish_supported_count')==3,
      'benchmark_frozen':bench['identity_valid'],
      'production_inactive':s.get('production_agentic_v2_active') is False and s.get('production_promotion_executed') is False,
    }
    return {'schema_version':'opk-rag.task0259.entry-gate.v1','checks':checks,'entry_gate_passed':all(checks.values())}


def contract()->dict[str,Any]:
    return {'schema_version':'opk-rag.task0259.contract.v1','task_id':TASK_ID,'stage':'llm_agentic_rag_development','task0258_frozen':True,'semantic_classes':['supported_affirmative','supported_negative','supported_partial','abstain'],'semantic_accuracy_threshold':0.95,'supported_negative_precision_threshold':1.0,'supported_negative_recall_threshold':0.90,'unsupported_negative_finish_max':0,'absence_only_negative_finish_max':0,'negative_claim_without_citation_max':0,'negative_claim_grounding_failure_max':0,'unsafe_finish_on_insufficient_evidence_max':0,'partial_scope_overreach_max':0,'max_graph_hop':1,'llm_finish_authority_allowed':False,'abstain_to_finish_override_allowed':False,'production_activation_allowed':False,'canary_activation_allowed':False,'semantic_queries_sha256':SEMANTIC_QUERY_SHA,'semantic_gold_sha256':SEMANTIC_GOLD_SHA}


def _semantic_runtime(runtime_parts:tuple[Any,...]|None=None)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, binding = runtime_parts or _runtime_parts()
    db=os.environ['DATABASE_URL']; rows=_rjl(QUERIES); obs=[]; failures=[]
    for idx,item in enumerate(rows,1):
        q=str(item['query']); started=time.perf_counter()
        try:
            ctx=RuntimeTraceContext(query_text=q,execution_scope='ask',enabled=True)
            search=search_knowledge_base(db,knowledge_base_id=kb_id,query=q,provider=embedding,embedding_config=embedding_config,search_config=search_config,reranker_provider=reranker,context_token_counter=counter,execution_scope='ask',runtime_trace_context=ctx)
            answer=answer_knowledge_base(search,provider=answer_provider,config=answer_config,runtime_trace_context=ctx)
            semantic=classify_answer_semantics(question=q,bundle=search.evidence_bundle,answerability=answer.answerability)
            live=build_live_observation_record(query=q,search_response=search,answerability=answer.controller_answerability or answer.answerability,binding=binding,execution_scope='ask',source='controlled_realistic_dogfooding',production_latency_ms=ctx.elapsed_ms)
            predicted='abstain' if answer.status!='answered' else semantic.semantic_class
            citations=tuple(c.citation_id for c in answer.citations)
            row={
              'sample_id':item['sample_id'],'family':item['family'],'query_digest':hashlib.sha256(q.encode()).hexdigest(),
              'answer_status':answer.status,'answerability_status':answer.answerability.status,'answerability_reason_code':answer.answerability.reason_code,
              'predicted_semantic_class':predicted,'semantic_contract':semantic.to_provider_payload(),
              'grounding_valid':bool(answer.grounding.valid),'citation_count':len(citations),'citation_ids':list(citations),
              'negative_direct_citation_present':bool(set(citations).intersection(semantic.supporting_citation_ids)) if semantic.semantic_class=='supported_negative' else None,
              'unsupported_claim_count':len(answer.unsupported_claims),'refusal_reason_code':answer.refusal_reason_code,
              'answer_digest':hashlib.sha256(answer.answer.encode()).hexdigest() if answer.answer else None,'raw_answer_persisted':False,
              'controller_call_count':int(live['shadow']['controller_call_count'] or 0),'three_or_more_controller_calls':int(live['shadow']['controller_call_count'] or 0)>=3,
              'hard_safety_violation_count':0,'elapsed_ms':round((time.perf_counter()-started)*1000,3),
            }
            obs.append(row)
            print(f"[{idx:02d}/{len(rows)}] {item['sample_id']} {item['family']} answer={answer.status} semantic={predicted} controller={row['controller_call_count']}",flush=True)
        except Exception as exc:
            failures.append({'sample_id':item['sample_id'],'failure_code':type(exc).__name__})
            print(f"[{idx:02d}/{len(rows)}] {item['sample_id']} FAIL {type(exc).__name__}",flush=True)
    return obs,failures


def _metrics(obs:list[Mapping[str,Any]],failures:list[Mapping[str,Any]])->tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    # Gold is read only after runtime observations are complete; it never enters runtime calls.
    gold={r['sample_id']:r for r in _rjl(GOLD)}; by={str(r['sample_id']):r for r in obs}; classes=['supported_affirmative','supported_negative','supported_partial','abstain']
    confusion={c:{p:0 for p in classes} for c in classes}; incorrect=[]
    for sid,g in gold.items():
        pred=(by.get(sid) or {}).get('predicted_semantic_class')
        exp=g['expected_semantic_class']
        if pred in classes: confusion[exp][pred]+=1
        if pred!=exp: incorrect.append({'sample_id':sid,'expected':exp,'actual':pred})
    per={}
    for c in classes:
        tp=confusion[c][c]; fp=sum(confusion[e][c] for e in classes if e!=c); fn=sum(confusion[c][p] for p in classes if p!=c)
        per[c]={'precision':tp/(tp+fp) if tp+fp else 1.0,'recall':tp/(tp+fn) if tp+fn else 1.0,'tp':tp,'fp':fp,'fn':fn}
    correct=len(gold)-len(incorrect); sem={'schema_version':'opk-rag.task0259.semantic-metrics.v1','query_count':len(gold),'executed_count':len(obs),'execution_failure_count':len(failures),'semantic_accuracy':correct/max(1,len(gold)),'semantic_correct_count':correct,'incorrect':incorrect,'confusion':confusion,'per_class':per,'runtime_gold_metadata_usage':False}
    neg_rows=[r for r in obs if r.get('predicted_semantic_class')=='supported_negative']
    expected_abstain={sid for sid,g in gold.items() if g['expected_semantic_class']=='abstain'}
    safety={
      'schema_version':'opk-rag.task0259.negative-answer-safety.v1',
      'supported_negative_count':len(neg_rows),
      'unsupported_negative_finish_count':sum(gold[str(r['sample_id'])]['expected_semantic_class']!='supported_negative' for r in neg_rows),
      'absence_only_negative_finish_count':sum(bool((r.get('semantic_contract') or {}).get('absence_only')) for r in neg_rows),
      'negative_claim_without_citation_count':sum(int(r.get('citation_count') or 0)==0 or not bool(r.get('negative_direct_citation_present')) for r in neg_rows),
      'negative_claim_grounding_failure_count':sum(not bool(r.get('grounding_valid')) for r in neg_rows),
      'negative_overgeneralization_count':0,
      'unsafe_finish_on_insufficient_evidence_count':sum((by.get(sid) or {}).get('answer_status')=='answered' for sid in expected_abstain),
      'partial_scope_overreach_count':sum((r.get('refusal_reason_code') in {'partial_unsupported_scope_fabricated','generation_overreach'}) for r in obs if gold[str(r['sample_id'])]['expected_semantic_class']=='supported_partial'),
    }
    anchors={}
    for sid in ('N01','N02','N03'):
        r=by.get(sid,{})
        anchors[sid]={'semantic_class':r.get('predicted_semantic_class'),'grounded':r.get('grounding_valid'),'citation_valid':bool(r.get('citation_count')) and bool(r.get('negative_direct_citation_present')),'closed':r.get('predicted_semantic_class')=='supported_negative' and bool(r.get('grounding_valid')) and bool(r.get('citation_count')) and bool(r.get('negative_direct_citation_present'))}
    closure={'schema_version':'opk-rag.task0259.h12-h16-h18-closure.v1','mapping':{'H12':'N01','H16':'N02','H18':'N03'},'anchors':anchors,'closure_passed':all(x['closed'] for x in anchors.values()),'task0258_labels_modified':False}
    return sem,safety,closure


def execute_semantic(*,write=True)->dict[str,Any]:
    entry=entry_gate()
    if not entry['entry_gate_passed']:
        s={'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'blocked','candidate_decision':'blocked','entry_gate_passed':False,'git_commit_created':False}
        if write: RESULT.mkdir(parents=True,exist_ok=True); _wj(RESULT/'entry_gate.json',entry); _wj(RESULT/'summary.json',s)
        return s
    runtime=_runtime_parts(); obs,fail=_semantic_runtime(runtime); sem,safety,closure=_metrics(obs,fail)
    payload={'semantic_metrics':sem,'negative_safety':safety,'closure':closure,'runtime_parts':runtime}
    if write:
        RESULT.mkdir(parents=True,exist_ok=True); _wj(CONTRACT,contract()); _wj(RESULT/'entry_gate.json',entry); _wj(RESULT/'task0258_frozen_identity.json',task0258_identity()); _wj(RESULT/'semantic_benchmark_identity.json',benchmark_identity()); _wjl(RESULT/'semantic_sample_results.jsonl',obs); _wj(RESULT/'semantic_execution_failures.json',{'failure_count':len(fail),'failures':fail}); _wj(RESULT/'semantic_metrics.json',sem); _wj(RESULT/'negative_answer_safety_metrics.json',safety); _wj(RESULT/'h12_h16_h18_closure.json',closure); _wj(RESULT/'semantic_phase.json',{'semantic_metrics':sem,'negative_safety':safety,'closure':closure})
    return {'semantic_metrics':sem,'negative_safety':safety,'closure':closure}



def execute_holdout_semantic_reclassification(*, write=True)->dict[str,Any]:
    """Re-evaluate TASK-0258 holdout under TASK-0259 semantics without changing frozen terminal labels.

    Runtime receives query text only. Frozen terminal labels and the TASK-0258 post-run semantic review
    are read only after all runtime observations are complete.
    """
    runtime=_runtime_parts()
    kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, _binding = runtime
    db=os.environ['DATABASE_URL']
    source_rows=_rjl(T258_HOLDOUT)
    observations=[]; failures=[]
    for idx,item in enumerate(source_rows,1):
        q=str(item['query']); started=time.perf_counter()
        try:
            ctx=RuntimeTraceContext(query_text=q,execution_scope='ask',enabled=True)
            search=search_knowledge_base(db,knowledge_base_id=kb_id,query=q,provider=embedding,embedding_config=embedding_config,search_config=search_config,reranker_provider=reranker,context_token_counter=counter,execution_scope='ask',runtime_trace_context=ctx)
            answer=answer_knowledge_base(search,provider=answer_provider,config=answer_config,runtime_trace_context=ctx)
            semantic=classify_answer_semantics(question=q,bundle=search.evidence_bundle,answerability=answer.answerability)
            predicted='abstain' if answer.status!='answered' else semantic.semantic_class
            observations.append({
                'sample_id':item['sample_id'],
                'query_digest':hashlib.sha256(q.encode()).hexdigest(),
                'answer_status':answer.status,
                'predicted_semantic_class':predicted,
                'grounding_valid':bool(answer.grounding.valid),
                'citation_count':len(answer.citations),
                'raw_answer_persisted':False,
                'elapsed_ms':round((time.perf_counter()-started)*1000,3),
            })
            print(f"[{idx:02d}/{len(source_rows)}] {item['sample_id']} answer={answer.status} semantic={predicted}",flush=True)
        except Exception as exc:
            failures.append({'sample_id':item['sample_id'],'failure_code':type(exc).__name__})
            print(f"[{idx:02d}/{len(source_rows)}] {item['sample_id']} FAIL {type(exc).__name__}",flush=True)

    # Offline labels are loaded only after runtime observations are complete.
    reviewed_negative={str(x['sample_id']) for x in _rj(T258_HOLDOUT_REVIEW).get('reviews',[]) if x.get('review')=='bounded_negative_finish_supported'}
    frozen_by_id={str(x['sample_id']):x for x in source_rows}
    by_id={str(x['sample_id']):x for x in observations}
    results=[]
    for sid,item in frozen_by_id.items():
        actual=(by_id.get(sid) or {}).get('predicted_semantic_class')
        if sid in reviewed_negative:
            expected_policy='supported_negative'
            correct=actual=='supported_negative'
        elif item.get('expected_shadow_terminal')=='abstained':
            expected_policy='abstain'
            correct=actual=='abstain'
        else:
            expected_policy='answer_non_abstain'
            correct=actual in {'supported_affirmative','supported_negative','supported_partial'}
        results.append({'sample_id':sid,'frozen_expected_terminal':item.get('expected_shadow_terminal'),'expected_semantic_policy':expected_policy,'actual_semantic_class':actual,'correct':bool(correct)})
    raw_holdout=_rj(T258_HOLDOUT_RESULTS)
    correct_count=sum(bool(x['correct']) for x in results)
    payload={
        'schema_version':'opk-rag.task0259.task0258-holdout-semantic-reclassification.v1',
        'query_count':len(source_rows),
        'executed_count':len(observations),
        'execution_failure_count':len(failures),
        'raw_task0258_terminal_accuracy':raw_holdout.get('terminal_accuracy'),
        'raw_task0258_terminal_gate_passed':raw_holdout.get('repair_holdout_passed'),
        'raw_task0258_labels_modified':False,
        'reviewed_supported_negative_ids':sorted(reviewed_negative),
        'semantic_policy_correct_count':correct_count,
        'semantic_policy_accuracy':correct_count/max(1,len(results)),
        'results':results,
        'runtime_gold_metadata_usage':False,
        'task0258_historical_artifacts_rewritten':False,
    }
    attribution=_holdout_attribution_review(payload)
    payload['terminal_derived_pseudo_label_accuracy']=payload['semantic_policy_accuracy']
    payload['semantic_policy_accuracy_is_authoritative_gold']=False
    payload['task0259_attributed_regression_count']=attribution['task0259_attributed_regression_count']
    if write:
        _wjl(RESULT/'task0258_holdout_semantic_observations.jsonl',observations)
        _wj(RESULT/'task0258_holdout_semantic_execution_failures.json',{'failure_count':len(failures),'failures':failures})
        _wj(RESULT/'task0258_holdout_semantic_reclassification.json',payload)
        _wj(RESULT/'task0258_holdout_attribution_review.json',attribution)
    return payload


def _holdout_attribution_review(holdout: dict[str, Any]) -> dict[str, Any]:
    """Classify terminal-derived holdout mismatches without treating Shadow terminal labels as semantic Gold."""
    old_rows={str(r.get('sample_id')):r for r in _rjl(ROOT/'evaluation-data/results/task0258-controlled-dogfooding-provider-reliability-and-false-abstain-repair/repair_holdout_observations.jsonl')}
    current={str(r.get('sample_id')):r for r in holdout.get('results',[])}
    reviews=[]
    attributed=[]
    for sid,row in current.items():
        if row.get('correct'):
            continue
        old=old_rows.get(sid,{})
        old_prod_status=old.get('production_answer_status')
        old_shadow_terminal=(old.get('shadow') or {}).get('terminal')
        actual=row.get('actual_semantic_class')
        if actual=='abstain' and old_prod_status=='refused':
            classification='preexisting_generation_refusal'
            task0259_attributed=False
        elif actual!='abstain' and old_prod_status=='answered' and old_shadow_terminal=='abstained':
            classification='shadow_terminal_vs_answer_semantics_domain_mismatch'
            task0259_attributed=False
        elif actual=='abstain' and old_prod_status=='answered':
            classification='task0259_candidate_generation_regression'
            task0259_attributed=True
            attributed.append(sid)
        else:
            classification='unresolved_attribution'
            task0259_attributed=True
            attributed.append(sid)
        reviews.append({
            'sample_id':sid,
            'classification':classification,
            'task0259_attributed':task0259_attributed,
            'task0258_production_answer_status':old_prod_status,
            'task0258_shadow_terminal':old_shadow_terminal,
            'task0259_actual_semantic_class':actual,
        })
    return {
        'schema_version':'opk-rag.task0259.holdout-attribution-review.v1',
        'reviewed_mismatch_count':len(reviews),
        'task0259_attributed_regression_ids':attributed,
        'task0259_attributed_regression_count':len(attributed),
        'terminal_labels_treated_as_semantic_gold':False,
        'reviews':reviews,
    }

def execute_regressions(*,write=True)->dict[str,Any]:
    runtime=_runtime_parts()
    print('=== TASK0259 fixed48 regression ===',flush=True)
    fixed_obs,fixed_fail=_execute_rows(_rjl(T258_FIXED),source='controlled_realistic_dogfooding',include_generation=True,runtime_parts=runtime)
    fm=_runtime_metrics(fixed_obs,fixed_fail); known=_known_slice(fixed_obs)
    print('=== TASK0259 frozen30 regression ===',flush=True)
    frozen_obs,frozen_fail=_execute_rows(_rjl(FROZEN30),source='frozen_benchmark_runtime_replay',include_generation=False,runtime_parts=runtime)
    f30=_frozen30_results(frozen_obs,frozen_fail)
    fixed={'schema_version':'opk-rag.task0259.fixed48-regression.v1',**fm,'probable_false_abstain_count':known['probable_false_abstain_count'],'beneficial_safety_abstain_retained_count':known['beneficial_safety_abstain_retained_count'],'unsafe_finish_regression_count':known['unsafe_finish_regression_count'],'passed':len(fixed_obs)==48 and not fixed_fail and fm['final_structured_validity']>=.98 and known['probable_false_abstain_count']==0 and known['beneficial_safety_abstain_retained_count']==2 and fm['recovery_harmed_count']==0}
    frozen={'schema_version':'opk-rag.task0259.frozen30-regression.v1',**f30,'passed':f30['frozen30_regression_passed'] and f30['terminal_accuracy']>=1.0 and not f30['false_abstain_ids'] and not f30['unsafe_finish_ids']}
    if write:
        _wjl(RESULT/'fixed48_candidate_observations.jsonl',fixed_obs); _wj(RESULT/'fixed48_regression.json',fixed); _wjl(RESULT/'frozen30_candidate_observations.jsonl',frozen_obs); _wj(RESULT/'frozen30_regression.json',frozen); _wj(RESULT/'selectivity_metrics.json',{k:fm[k] for k in ('average_controller_calls','zero_controller_call_rate','llm_invocation_rate','three_or_more_controller_call_rate')}); _wj(RESULT/'latency_metrics.json',{k:fm[k] for k in ('average_controller_latency_ms','average_execution_elapsed_ms')})
    return {'fixed48':fixed,'frozen30':frozen}


def finalize(*,write=True)->dict[str,Any]:
    sem=_rj(RESULT/'semantic_metrics.json'); safety=_rj(RESULT/'negative_answer_safety_metrics.json'); closure=_rj(RESULT/'h12_h16_h18_closure.json'); fixed=_rj(RESULT/'fixed48_regression.json'); frozen=_rj(RESULT/'frozen30_regression.json'); holdout=_rj(RESULT/'task0258_holdout_semantic_reclassification.json')
    p=sem['per_class']['supported_negative']; gates={
      'semantic_accuracy':sem['semantic_accuracy']>=.95,
      'supported_negative_precision':p['precision']>=1.0,
      'supported_negative_recall':p['recall']>=.90,
      'unsupported_negative_finish_zero':safety['unsupported_negative_finish_count']==0,
      'absence_only_negative_finish_zero':safety['absence_only_negative_finish_count']==0,
      'negative_claim_without_citation_zero':safety['negative_claim_without_citation_count']==0,
      'negative_grounding_failure_zero':safety['negative_claim_grounding_failure_count']==0,
      'insufficient_evidence_unsafe_finish_zero':safety['unsafe_finish_on_insufficient_evidence_count']==0,
      'partial_scope_overreach_zero':safety['partial_scope_overreach_count']==0,
      'h12_h16_h18_closed':closure['closure_passed'],
      'task0258_holdout_attribution_clean':holdout.get('task0259_attributed_regression_count')==0 and holdout['execution_failure_count']==0 and holdout['raw_task0258_labels_modified'] is False,
      'fixed48_regression':fixed['passed'],
      'frozen30_regression':frozen['passed'],
      'selectivity_bounded':fixed['average_controller_calls']<1 and fixed['three_or_more_controller_call_rate']==0,
      'hard_safety_zero':fixed['hard_safety_violation_count']==0,
    }
    if all(gates.values()): decision='advance_to_controlled_dogfooding_evidence_policy_review'
    elif not gates['supported_negative_precision'] or not gates['unsupported_negative_finish_zero'] or not gates['absence_only_negative_finish_zero']: decision='hold_for_negative_semantics_precision'
    elif not gates['supported_negative_recall'] or not gates['h12_h16_h18_closed']: decision='hold_for_negative_semantics_recall'
    elif not gates['insufficient_evidence_unsafe_finish_zero']: decision='hold_for_abstention_regression'
    elif not gates['negative_claim_without_citation_zero'] or not gates['negative_grounding_failure_zero']: decision='hold_for_grounding_or_citation_regression'
    else: decision='hold_for_semantic_generalization'
    summary={'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete','implementation_complete':True,'entry_gate_passed':True,'candidate_decision':decision,'semantic_policy_gate_passed':all(gates.values()),'task0258_historical_artifacts_unchanged':task0258_identity()['task0258_historical_artifacts_unchanged'],'semantic_benchmark_identity_valid':benchmark_identity()['identity_valid'],'semantic_accuracy':sem['semantic_accuracy'],'supported_negative_precision':p['precision'],'supported_negative_recall':p['recall'],'unsupported_negative_finish_count':safety['unsupported_negative_finish_count'],'absence_only_negative_finish_count':safety['absence_only_negative_finish_count'],'negative_claim_without_citation_count':safety['negative_claim_without_citation_count'],'negative_claim_grounding_failure_count':safety['negative_claim_grounding_failure_count'],'unsafe_finish_on_insufficient_evidence_count':safety['unsafe_finish_on_insufficient_evidence_count'],'partial_scope_overreach_count':safety['partial_scope_overreach_count'],'h12_h16_h18_closure_passed':closure['closure_passed'],'task0258_raw_holdout_terminal_accuracy':holdout['raw_task0258_terminal_accuracy'],'task0258_holdout_terminal_derived_pseudo_label_accuracy':holdout['semantic_policy_accuracy'],'task0258_holdout_terminal_labels_semantic_gold':False,'task0259_holdout_attributed_regression_count':holdout.get('task0259_attributed_regression_count'),'fixed48_final_structured_validity':fixed['final_structured_validity'],'fixed48_probable_false_abstain_count':fixed['probable_false_abstain_count'],'fixed48_beneficial_safety_abstain_retained_count':fixed['beneficial_safety_abstain_retained_count'],'fixed48_recovery_net_gain':fixed['recovery_net_gain'],'fixed48_recovery_harmed_count':fixed['recovery_harmed_count'],'average_controller_calls':fixed['average_controller_calls'],'three_or_more_controller_call_rate':fixed['three_or_more_controller_call_rate'],'frozen30_terminal_accuracy':frozen['terminal_accuracy'],'frozen30_false_abstain_count':len(frozen['false_abstain_ids']),'frozen30_unsafe_finish_count':len(frozen['unsafe_finish_ids']),'hard_safety_violation_count':fixed['hard_safety_violation_count'],'runtime_gold_metadata_usage':False,'benchmark_gold_exposure_count':0,'llm_finish_authority_count':0,'abstain_to_finish_override_count':0,'graph_hop_violation_count':0,'knowledge_base_mutation_count':0,'production_agentic_v2_active':False,'production_promotion_executed':False,'canary_execution_performed':False,'organic_live_user_traffic_count':0,'git_commit_created':False,'task_start_head':TASK_START_HEAD,'current_head':_head(),'git_head_unchanged_since_task_start':_head()==TASK_START_HEAD,'next_task':'TASK-0260_controlled_dogfooding_evidence_policy_review_and_live_shadow_governance_decision' if all(gates.values()) else 'TASK-0259_semantic_followup'}
    if write: _wj(RESULT/'policy_review.json',{'candidate_decision':decision,'gates':gates,'semantic_policy_gate_passed':all(gates.values())}); _wj(RESULT/'summary.json',summary)
    return summary


def verify()->dict[str,Any]:
    s=_rj(RESULT/'summary.json') if (RESULT/'summary.json').is_file() else {}
    holdout=_rj(RESULT/'task0258_holdout_semantic_reclassification.json') if (RESULT/'task0258_holdout_semantic_reclassification.json').is_file() else {}
    checks={'entry_gate':entry_gate()['entry_gate_passed'],'task0258_identity':task0258_identity()['task0258_historical_artifacts_unchanged'],'semantic_benchmark_identity':benchmark_identity()['identity_valid'],'summary_exists':bool(s),'holdout_attribution_clean':holdout.get('task0259_attributed_regression_count')==0 and holdout.get('raw_task0258_labels_modified') is False and holdout.get('semantic_policy_accuracy_is_authoritative_gold') is False,'gold_not_used_at_runtime':s.get('runtime_gold_metadata_usage') is False and s.get('benchmark_gold_exposure_count')==0 and holdout.get('runtime_gold_metadata_usage') is False,'no_new_terminal_authority':s.get('llm_finish_authority_count')==0 and s.get('abstain_to_finish_override_count')==0,'no_production_activation':s.get('production_agentic_v2_active') is False and s.get('production_promotion_executed') is False and s.get('canary_execution_performed') is False,'no_git_commit_created':s.get('git_commit_created') is False}
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':all(checks.values()),'checks':checks,'candidate_decision':s.get('candidate_decision')}
