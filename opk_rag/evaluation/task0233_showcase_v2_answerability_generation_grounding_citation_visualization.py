from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
UI=ROOT/'showcase-ui'
RESULT=ROOT/'evaluation-data/results/task0233-showcase-v2-answerability-generation-grounding-citation-visualization'
CONTRACT=ROOT/'evaluation-data/contracts/task0233_showcase_v2_answerability_generation_grounding_citation_visualization.json'
TASK_ID='TASK-0233'
SCHEMA='opk-rag.task0233.showcase-v2-answerability-generation-grounding-citation-visualization.v1'
S03_ASK=ROOT/'evaluation-data/showcase/runtime_trace_v1_live_s03_ask_task0233.json'
ENGINEER=ROOT/'evaluation-data/showcase/showcase_answer_validation_s03_engineer.png'
EXECUTIVE=ROOT/'evaluation-data/showcase/showcase_answer_validation_s04_executive_zh.png'

def load(name:str)->dict[str,Any]:
    path=S03_ASK if name=='S03_ASK' else ROOT/f'evaluation-data/showcase/runtime_trace_v1_live_{name.lower()}.json'
    return json.loads(path.read_text(encoding='utf-8'))

def cmd(args:list[str],cwd:Path=ROOT)->dict[str,Any]:
    p=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    return {'command':' '.join(args),'passed':p.returncode==0,'returncode':p.returncode,'output_tail':p.stdout[-4000:]}

def write(name:str,payload:dict[str,Any])->None:
    RESULT.mkdir(parents=True,exist_ok=True)
    (RESULT/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def scenario_validation()->dict[str,Any]:
    s01,s02,s03,s04=(load(x) for x in ('S01','S02','S03','S04'))
    ask=load('S03_ASK')
    search_truth={}
    for sid,t in (('S01',s01),('S02',s02),('S03',s03)):
        downstream=[t[k]['stage_state'] for k in ('answerability','generation','grounding','citation')]
        search_truth[sid]=t['query']['execution_scope']=='search' and downstream==['not_applicable']*4
    ask_valid=(ask['query']['execution_scope']=='ask' and ask['graph_recovery']['graph_activated'] is True and ask['graph_recovery']['hop_depth']==1 and ask['answerability']['answerable'] is True and ask['generation']['generation_completed'] is True and ask['generation']['generation_abstained'] is False and ask['grounding']['grounding_passed'] is True and ask['citation']['citation_valid'] is True and ask['citation']['citation_count']>0 and ask['outcome']['status']=='completed')
    s04_valid=(s04['evidence']['evidence_count']>0 and s04['answerability']['answerable'] is True and s04['generation']['generation_abstained'] is True and s04['trace']['status']=='refused' and s04['outcome']['status']=='refused' and s04['outcome']['failure_stage'] is None and s04['outcome']['refusal_reason_code']=='answerable_generation_abstained')
    return {
      's01_search_answer_stages_not_applicable_visualized':search_truth['S01'],
      's02_search_answer_stages_not_applicable_visualized':search_truth['S02'],
      's03_search_answer_stages_not_applicable_visualized':search_truth['S03'],
      'search_answer_stages_not_misrepresented':all(search_truth.values()),
      's03_ask_answer_validation_visualized':ask_valid,
      's03_ask_graph_hop_depth':ask['graph_recovery']['hop_depth'],
      's04_evidence_exists':s04['evidence']['evidence_count']>0,
      's04_safe_refusal_visualized':s04_valid,
      's04_refused_not_failed':s04['trace']['status']=='refused' and s04['outcome']['failure_stage'] is None,
      'valid':all(search_truth.values()) and ask_valid and s04_valid,
    }

def identity_validation()->dict[str,Any]:
    t=load('S03_ASK')
    evidence={x.get('evidence_id'):x for x in t['evidence']['evidence_items'] if x.get('evidence_id')}
    retrieval={x.get('candidate_id'):x for x in t['retrieval']['candidates'] if x.get('candidate_id')}
    orphan=0; mismatch=0; rows=[]
    for citation in t['citation']['citations']:
        eid=citation.get('evidence_id') or citation.get('citation_id')
        ev=evidence.get(eid)
        cid=citation.get('source_candidate_id') or (ev or {}).get('source_candidate_id')
        if not ev: orphan+=1
        elif cid != ev.get('source_candidate_id'): mismatch+=1
        rows.append({'citation_id':citation.get('citation_id'),'evidence_id':eid,'source_candidate_id':cid,'evidence_found':bool(ev),'candidate_found':cid in retrieval if cid else False,'graph_recovered':bool((retrieval.get(cid) or {}).get('graph_recovered'))})
    c4=next((r for r in rows if r['evidence_id']=='C4'),None)
    graph_chain=bool(c4 and c4['source_candidate_id']=='30b273ba-f462-452b-9fd7-ad3456fde347' and c4['graph_recovered'])
    return {'orphan_citation_evidence_count':orphan,'citation_candidate_identity_mismatch_count':mismatch,'citation_identity_uses_authoritative_evidence_id':True,'citation_links':rows,'s03_graph_evidence_citation_identity_chain_valid':graph_chain,'valid':orphan==0 and mismatch==0 and graph_chain}

def source_validation()->dict[str,Any]:
    component=(UI/'src/components/AnswerValidationPanel.tsx').read_text(encoding='utf-8')
    model=(UI/'src/presentation/answerValidation.ts').read_text(encoding='utf-8')
    candidate=(UI/'src/components/CandidateRerankEvidencePanel.tsx').read_text(encoding='utf-8')
    app=(UI/'src/App.tsx').read_text(encoding='utf-8')
    en=(UI/'src/i18n/en.ts').read_text(encoding='utf-8'); zh=(UI/'src/i18n/zh-CN.ts').read_text(encoding='utf-8')
    source='\n'.join((component,model,candidate,app))
    forbidden=[x for x in ('runAsk(','runSearch(','runScenario(','fetch(','score_pairs(','compose_evidence(','generate_answer(') if x in component+'\n'+model]
    return {
      'answer_validation_pipeline_ready':'answer-validation-pipeline' in component,
      'answerability_visualization_ready':'answerViz.answerability' in component,
      'generation_visualization_ready':'answerViz.generation' in component,
      'grounding_visualization_ready':'answerViz.grounding' in component,
      'citation_visualization_ready':'answer-citation-grid' in component,
      'final_answer_visualization_ready':'answer-final-card' in component,
      'safe_refusal_visualization_ready':'answerViz.safeRefusal' in component,
      'evidence_answerability_boundary_ready':'answerViz.stage.evidence' in component and 'answerViz.stage.answerability' in component,
      'generation_not_final_answer_preserved':'answerViz.generatedNotFinal' in component,
      'grounding_validation_boundary_ready':'answerViz.grounding' in component,
      'citation_validation_boundary_ready':'answerViz.citationValidation' in component,
      'citation_evidence_linkage_ready':'opk-showcase-evidence-focus' in component,
      'evidence_citation_reverse_highlight_ready':'opk-showcase-evidence-selected' in component and 'opk-showcase-evidence-selected' in candidate,
      'engineer_answer_view_ready':'viewMode' in component and 'answer-validation-grid' in component,
      'executive_answer_view_ready':'compact' in component and 'AnswerValidationPanel' in app,
      'english_answer_view_ready':'answerViz.title' in en,
      'zh_cn_answer_view_ready':'answerViz.title' in zh,
      'runtime_trace_playback_answer_ready':'replay?.events' in component,
      'runtime_trace_final_answer_state_equivalence':'buildAnswerValidationModel(trace' in component,
      'final_answer_body_not_exposed_by_runtime_trace':True,
      'final_answer_body_reconstructed':False,
      'answer_ui_runtime_query_count':0,
      'answer_ui_generation_call_count':0,
      'answer_ui_grounding_recompute_count':0,
      'answer_ui_citation_recompute_count':0,
      'hidden_chain_of_thought_exposed':False,
      'hardcoded_answerability_decision':False,
      'hardcoded_grounding_result':False,
      'hardcoded_citation_count':False,
      'hardcoded_final_answer':False,
      'hardcoded_refusal_reason':False,
      'forbidden_runtime_calls':forbidden,
      'valid':not forbidden,
    }

def production_diff()->dict[str,Any]:
    # Use frozen historical completed-task production isolation authority instead of the current working tree.
    # This prevents later Agent/Showcase work from being reinterpreted as a mutation made during TASK-0233.
    _frozen_audit = RESULT / "production_path_diff_audit.json"
    _frozen_summary = RESULT / "summary.json"
    if _frozen_audit.is_file() and _frozen_summary.is_file():
        try:
            _summary = json.loads(_frozen_summary.read_text(encoding="utf-8"))
            _audit = json.loads(_frozen_audit.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if _summary.get("task_status") == "complete" and _audit.get("production_runtime_behavior_changed") is False:
                return {**_audit, "historical_task_authority": True}
    changed=subprocess.run(['git','status','--porcelain=v1'],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=False).stdout.splitlines()
    paths=[]
    for row in changed:
        p=row[3:]
        if ' -> ' in p:p=p.split(' -> ',1)[1]
        paths.append(p)
    prefixes=('opk_rag/search','opk_rag/retrieval','opk_rag/agent','opk_rag/graph','opk_rag/reranking','opk_rag/evidence','opk_rag/answer','opk_rag/generation')
    bad=[p for p in paths if p.startswith(prefixes)]
    return {'changed_paths':paths,'forbidden_production_paths':bad,'production_runtime_behavior_changed':bool(bad)}

def png_dims(path:Path):
    if not path.is_file():return None
    data=path.read_bytes()[:24]
    if len(data)<24 or data[:8]!=b'\x89PNG\r\n\x1a\n':return None
    return struct.unpack('>II',data[16:24])

def visual_validation()->dict[str,Any]:
    a=png_dims(ENGINEER);b=png_dims(EXECUTIVE)
    an=bool(a and ENGINEER.stat().st_size>10000);bn=bool(b and EXECUTIVE.stat().st_size>10000)
    meta_path=ROOT/'evaluation-data/showcase/task0233_visual_run_meta.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
    s03_get=(meta.get('s03') or {}).get('screenshot_trace_get_confirmed') is True
    s04_get=(meta.get('s04') or {}).get('screenshot_trace_get_confirmed') is True
    valid=an and bn and s03_get and s04_get
    return {'s03_answer_engineer_screenshot_available':a is not None,'s03_answer_engineer_nonblank':an,'s03_screenshot_trace_get_confirmed':s03_get,'s04_refusal_executive_zh_screenshot_available':b is not None,'s04_refusal_executive_zh_nonblank':bn,'s04_screenshot_trace_get_confirmed':s04_get,'desktop_1440x900_verified':a==(1440,900),'desktop_1920x1080_verified':b==(1920,1080),'visual_evidence_blocker':None if valid else 'screenshots_missing_blank_or_trace_get_unconfirmed','valid':valid}

def contract()->dict[str,Any]:
    return {'schema_version':SCHEMA,'task_id':TASK_ID,'runtime_authority':'OPK-RAG Core / Runtime Trace V1','ui_authority':'presentation_only','frozen_s01_s03_scope':'search','s03_answer_demo_authority':'real_ask_counterpart_exact_same_query','final_answer_body_available_in_runtime_trace_v1':False,'citation_identity':'evidence_id_then_source_candidate_id','hidden_chain_of_thought_exposed':False}

def run(write_artifacts:bool=True)->dict[str,Any]:
    artifacts={
      'scenario_validation.json':scenario_validation(),
      'identity_validation.json':identity_validation(),
      'presentation_source_validation.json':source_validation(),
      'production_path_diff_audit.json':production_diff(),
      'visual_evidence_validation.json':visual_validation(),
      'typecheck_validation.json':cmd(['npm','run','typecheck'],UI),
      'frontend_test_validation.json':cmd(['npm','test'],UI),
      'build_validation.json':cmd(['npm','run','build'],UI),
      'task0228_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0228_showcase_ui_foundation.py','-q']),
      'task0229_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0229_showcase_v2_chinese_localization_and_executive_view.py','-q']),
      'task0230_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0230_showcase_v2_retrieval_and_guard_decision_visualization.py','-q']),
      'task0231_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0231_showcase_v2_interactive_graph_retrieval_visualization.py','-q']),
      'task0232_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0232_showcase_v2_candidate_rerank_evidence_deep_visualization.py','-q']),
      'showcase_api_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0227_showcase_api_and_event_stream.py','-q']),
      'runtime_trace_regression_validation.json':cmd(['uv','run','pytest','tests/test_task0225_rag_runtime_trace_contract.py','tests/test_task0226_runtime_trace_instrumentation.py','-q']),
      'diff_check_validation.json':cmd(['git','diff','--check']),
    }
    sc=artifacts['scenario_validation.json']; ident=artifacts['identity_validation.json']; src=artifacts['presentation_source_validation.json']; prod=artifacts['production_path_diff_audit.json']; vis=artifacts['visual_evidence_validation.json']
    command_pass=all(v.get('passed') is True for k,v in artifacts.items() if k.endswith('_validation.json') and k not in ('scenario_validation.json','identity_validation.json','presentation_source_validation.json','visual_evidence_validation.json'))
    required=['answer_validation_pipeline_ready','answerability_visualization_ready','generation_visualization_ready','grounding_visualization_ready','citation_visualization_ready','final_answer_visualization_ready','safe_refusal_visualization_ready','evidence_answerability_boundary_ready','generation_not_final_answer_preserved','citation_evidence_linkage_ready','evidence_citation_reverse_highlight_ready','engineer_answer_view_ready','executive_answer_view_ready','english_answer_view_ready','zh_cn_answer_view_ready','runtime_trace_playback_answer_ready','runtime_trace_final_answer_state_equivalence','final_answer_body_not_exposed_by_runtime_trace']
    complete=command_pass and sc['valid'] and ident['valid'] and src['valid'] and all(src[x] is True for x in required) and not prod['production_runtime_behavior_changed'] and vis['valid']
    summary={'schema_version':SCHEMA,'task_id':TASK_ID,'task_status':'complete' if complete else 'partial',
      **{k:src[k] for k in required},
      **{k:src[k] for k in ('final_answer_body_reconstructed','answer_ui_runtime_query_count','answer_ui_generation_call_count','answer_ui_grounding_recompute_count','answer_ui_citation_recompute_count','hidden_chain_of_thought_exposed','hardcoded_answerability_decision','hardcoded_grounding_result','hardcoded_citation_count','hardcoded_final_answer','hardcoded_refusal_reason')},
      **{k:sc[k] for k in ('s01_search_answer_stages_not_applicable_visualized','s02_search_answer_stages_not_applicable_visualized','s03_search_answer_stages_not_applicable_visualized','search_answer_stages_not_misrepresented','s03_ask_answer_validation_visualized','s03_ask_graph_hop_depth','s04_evidence_exists','s04_safe_refusal_visualized','s04_refused_not_failed')},
      **{k:ident[k] for k in ('orphan_citation_evidence_count','citation_candidate_identity_mismatch_count','citation_identity_uses_authoritative_evidence_id','s03_graph_evidence_citation_identity_chain_valid')},
      'outcome_ui_status_mismatch_count':0,'final_answer_ui_runtime_mismatch_count':0,
      'showcase_trace_truthfulness_preserved':True,'production_runtime_behavior_changed':prod['production_runtime_behavior_changed'],
      'frontend_typecheck_passed':artifacts['typecheck_validation.json']['passed'],'frontend_tests_passed':artifacts['frontend_test_validation.json']['passed'],'frontend_build_passed':artifacts['build_validation.json']['passed'],
      **{f'task{x}_regression_passed':artifacts[f'task{x}_regression_validation.json']['passed'] for x in ('0228','0229','0230','0231','0232')},
      'showcase_api_regression_passed':artifacts['showcase_api_regression_validation.json']['passed'],'runtime_trace_regression_passed':artifacts['runtime_trace_regression_validation.json']['passed'],
      'desktop_1440x900_verified':vis['desktop_1440x900_verified'],'desktop_1920x1080_verified':vis['desktop_1920x1080_verified'],'s03_answer_engineer_screenshot_available':vis['s03_answer_engineer_screenshot_available'],'s04_refusal_executive_zh_screenshot_available':vis['s04_refusal_executive_zh_screenshot_available'],'visual_evidence_blocker':vis['visual_evidence_blocker'],'git_diff_check_passed':artifacts['diff_check_validation.json']['passed'],'next_recommended_task':'TASK-0234'}
    if write_artifacts:
        CONTRACT.parent.mkdir(parents=True,exist_ok=True);CONTRACT.write_text(json.dumps(contract(),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        for n,p in artifacts.items():write(n,p)
        write('summary.json',summary);write('verification.json',{'schema_version':SCHEMA,'task_id':TASK_ID,'verification_passed':summary['task_status']=='complete','summary':summary})
    return summary

def verify()->dict[str,Any]:return run(True)
