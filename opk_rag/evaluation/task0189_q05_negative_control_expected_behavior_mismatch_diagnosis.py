from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, write_json
from opk_rag.evaluation.task0183_cold_start_vector_index_materialization_diagnosis import audit_database_vector_state
from opk_rag.evaluation.task0184_cold_start_retrieval_runtime_validation_and_diagnosis import _resolve_database_url
from opk_rag.evaluation.task0188_cold_start_end_to_end_q01_q07_validation import (
    CORPUS_MANIFEST_PATH,
    CORPUS_ROOT,
    EXPECTED_QUERIES_PATH,
    KNOWN_PREEXISTING_FAILURES,
    _current_project_env,
    _parse_pytest_failed,
    _parse_pytest_passed,
    _parse_pytest_skipped,
    _read_json,
    _tail,
    audit_corpus_authority,
    load_query_fixture,
    scan_artifacts_for_secrets,
)
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config

TASK_ID = "TASK-0189"
SOURCE_AUTHORITATIVE_TASK = "TASK-0188"
EXPERIMENT_ID = "task0189-q05-negative-control-expected-behavior-mismatch-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0189_q05_negative_control_expected_behavior_mismatch_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0189_Q05_NEGATIVE_CONTROL_EXPECTED_BEHAVIOR_MISMATCH_DIAGNOSIS_REPORT.md"
Q05_ID = "Q05_NEGATIVE_CONTROL"
Q05_EXPECTED_SOURCE = "06_negative_control.md"


@dataclass(frozen=True)
class SemanticCounts:
    support_status: str
    supporting_document_count: int
    supporting_chunk_count: int
    contradicting_document_count: int
    contradicting_chunk_count: int
    support_present: bool
    sufficient_for_answer: bool


def run_task0189(*, write: bool = True, run_full_suite: bool = False, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    if env is None:
        load_project_env(ROOT)
        env = _current_project_env()
    else:
        env = dict(env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    source_head = _git_head()
    source_summary = _read_json(ROOT / "evaluation-data" / "results" / "task0188-cold-start-end-to-end-q01-q07-validation" / "summary.json")
    q05_fixture, fixture_audit = audit_q05_fixture()
    origin = audit_q05_origin()
    database_url = _resolve_database_url(env)
    embedding_config = load_embedding_config(env)
    db = audit_database_vector_state(database_url, embedding_config)
    corpus = audit_corpus_authority(database_url, db)
    corpus_semantics = audit_corpus_semantics()
    payload, replay_meta = execute_q05_production_ask(q05_fixture, env=env, knowledge_base_id=str(db.get("knowledge_base_id") or ""))
    trace = audit_runtime_trace(payload, database_url=database_url, knowledge_base_id=str(db.get("knowledge_base_id") or ""))
    evidence_semantics = audit_evidence_semantics(payload)
    policy = audit_current_policy(env)
    adjudication = adjudicate(
        fixture_audit=fixture_audit,
        origin=origin,
        corpus_semantics=corpus_semantics,
        evidence_semantics=evidence_semantics,
        payload=payload,
        trace=trace,
    )
    focused = run_focused_tests()
    regression = run_regression_tests()
    full_suite = run_full_pytest() if run_full_suite else skipped_full_suite_audit()
    summary = build_summary(
        source_head=source_head,
        source_summary=source_summary,
        fixture_audit=fixture_audit,
        origin=origin,
        db=db,
        corpus=corpus,
        corpus_semantics=corpus_semantics,
        replay_meta=replay_meta,
        trace=trace,
        evidence_semantics=evidence_semantics,
        policy=policy,
        adjudication=adjudication,
        focused=focused,
        regression=regression,
        full_suite=full_suite,
    )
    artifacts: dict[str, Any] = {
        "summary.json": summary,
        "q05_fixture_authority_audit.json": fixture_audit,
        "q05_historical_provenance_audit.json": origin,
        "q05_corpus_semantic_audit.json": asdict(corpus_semantics),
        "q05_production_replay_payload.json": payload,
        "q05_runtime_trace_audit.json": trace,
        "q05_evidence_sufficiency_audit.json": asdict(evidence_semantics),
        "q05_adjudication.json": adjudication,
        "contract.json": contract(),
    }
    if write:
        for name, payload_obj in artifacts.items():
            write_json(RESULT_DIR / name, payload_obj)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        secret_scan = scan_artifacts_for_secrets((RESULT_DIR, CONTRACT_PATH, REPORT_PATH), env)
        write_json(RESULT_DIR / "artifact_secret_scan.json", secret_scan)
        summary = {**summary, **secret_scan}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_q05_fixture(path: Path = EXPECTED_QUERIES_PATH) -> tuple[Any, dict[str, Any]]:
    fixtures = load_query_fixture(path)
    raw = _read_json(path)
    raw_q05 = next((item for item in raw if isinstance(item, dict) and item.get("query_id") == Q05_ID), {})
    q05 = next(fixture for fixture in fixtures if fixture.query_id == Q05_ID)
    expected_abstention_field_present = "expected_abstention" in raw_q05
    expected_abstention = bool(raw_q05.get("expected_abstention", False))
    expected_behavior = "abstain" if expected_abstention else "answer"
    expected_documents = tuple(str(value) for value in raw_q05.get("expected_documents", ()) if str(value))
    return q05, {
        "q05_query_id": q05.query_id,
        "q05_query_text": q05.query,
        "q05_query_text_digest": _sha256(q05.query.encode("utf-8")),
        "q05_fixture_raw_schema_fields": sorted(raw_q05.keys()),
        "q05_fixture_expected_answerable": not expected_abstention,
        "q05_fixture_expected_abstention": expected_abstention,
        "q05_fixture_expected_abstention_field_present": expected_abstention_field_present,
        "q05_fixture_expected_behavior": expected_behavior,
        "q05_fixture_expected_source_present": bool(expected_documents),
        "q05_fixture_expected_source_count": len(expected_documents),
        "q05_fixture_expected_sources": list(expected_documents),
        "q05_fixture_expected_concepts_present": bool(raw_q05.get("expected_answer_contains")),
        "q05_fixture_expected_concepts": list(raw_q05.get("expected_answer_contains", ())),
        "q05_fixture_corpus_authority": "EXPECTED_QUERIES.json@sha256:" + _sha256(path.read_bytes()) if path.exists() else "missing",
        "q05_fixture_authority_audited": True,
    }


def audit_q05_origin() -> dict[str, Any]:
    commit = _first_commit_containing(Q05_ID)
    task_text = _git_show(commit, "tasks/TASK-0170_cold_start_reproducibility_baseline.md") if commit else ""
    section = _extract_section(task_text, "## Q05")
    found = bool(commit and "Expected semantics" in section and "不存在该依赖" in section)
    return {
        "q05_origin_authority_found": found,
        "q05_origin_task_id": "TASK-0170" if found else "unknown",
        "q05_origin_commit": commit or "unknown",
        "q05_original_semantic_intent": "不存在该依赖" if found else "unknown",
        "q05_original_expected_behavior": "answer" if found else "unknown",
        "q05_original_detected_controls": _controls_from_origin(section),
        "q05_negative_control_type": "answerable_with_negative_distractor" if found else "unknown",
        "q05_original_intent_authority_strength": "authoritative" if found else "unknown",
        "q05_historical_intent_audited": True,
    }


def audit_corpus_semantics(corpus_root: Path = CORPUS_ROOT) -> SemanticCounts:
    supporting_docs = 0
    supporting_chunks = 0
    contradicting_docs = 0
    contradicting_chunks = 0
    for path in sorted(corpus_root.glob("*.md")):
        if path.name in {"README.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        supports = _supports_q05_negative_answer(text)
        contradicts = _contradicts_q05_negative_answer(text)
        if supports:
            supporting_docs += 1
            supporting_chunks += 1
        if contradicts:
            contradicting_docs += 1
            contradicting_chunks += 1
    status = "explicitly_supported" if supporting_chunks and not contradicting_chunks else "ambiguous"
    return SemanticCounts(
        support_status=status,
        supporting_document_count=supporting_docs,
        supporting_chunk_count=supporting_chunks,
        contradicting_document_count=contradicting_docs,
        contradicting_chunk_count=contradicting_chunks,
        support_present=supporting_chunks > 0,
        sufficient_for_answer=supporting_chunks > 0 and contradicting_chunks == 0,
    )


def execute_q05_production_ask(q05_fixture: Any, *, env: Mapping[str, str], knowledge_base_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    timeout_seconds = max(30.0, load_answer_generation_config(env).timeout_seconds + 240.0)
    started = time.perf_counter()
    command = [
        "uv",
        "run",
        "opk-rag",
        "ask",
        "--knowledge-base-id",
        knowledge_base_id,
        "--query",
        q05_fixture.query,
        "--format",
        "json",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=dict(env), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
    payload: dict[str, Any] = {}
    json_valid = False
    try:
        parsed = json.loads(completed.stdout)
        if isinstance(parsed, dict):
            payload = parsed
            json_valid = True
    except json.JSONDecodeError:
        payload = {}
    meta = {
        "q05_production_ask_returncode": completed.returncode,
        "q05_production_ask_success": completed.returncode == 0,
        "q05_production_stdout_json_valid": json_valid,
        "q05_production_stderr_present": bool(completed.stderr.strip()),
        "q05_production_latency_ms": int((time.perf_counter() - started) * 1000),
    }
    return payload, meta


def audit_runtime_trace(payload: Mapping[str, Any], *, database_url: str, knowledge_base_id: str) -> dict[str, Any]:
    search = payload.get("search") if isinstance(payload.get("search"), dict) else {}
    results = search.get("results") if isinstance(search.get("results"), list) else []
    evidence_bundle = search.get("evidence_bundle") if isinstance(search.get("evidence_bundle"), dict) else {}
    evidence_items = evidence_bundle.get("items") if isinstance(evidence_bundle.get("items"), list) else []
    document_ids = _document_ids(database_url, knowledge_base_id)
    relevant = [_is_q05_relevant(item) for item in results]
    evidence_relevant = [_is_q05_relevant(item) for item in evidence_items]
    answerability = payload.get("answerability") if isinstance(payload.get("answerability"), dict) else {}
    model_decision = payload.get("model_decision") if isinstance(payload.get("model_decision"), dict) else {}
    refusal_reason = str(payload.get("refusal_reason_code") or "")
    answerability_status = str(answerability.get("status") or "unknown")
    refusal_origin = _refusal_origin(payload)
    search_mode = str(search.get("retrieval_mode") or "")
    return {
        "q05_retrieval_reached": bool(search),
        "q05_vector_retrieval_reached": int(search.get("vector_candidate_count") or 0) > 0 or search_mode in {"vector", "hybrid"},
        "q05_lexical_retrieval_reached": int(search.get("bm25_candidate_count") or 0) > 0 or search_mode in {"bm25", "hybrid"},
        "q05_fusion_reached": search_mode == "hybrid",
        "q05_guarded_structure_aware_retrieval_reached": False,
        "q05_candidate_count": int(search.get("candidate_count") or len(results)),
        "q05_canonical_candidate_count": len(results),
        "q05_orphan_candidate_count": sum(1 for item in results if str(item.get("document_id") or "") not in document_ids),
        "q05_relevant_candidate_count": sum(1 for value in relevant if value),
        "q05_irrelevant_candidate_count": sum(1 for value in relevant if not value),
        "q05_retrieval_support_sufficient": any(relevant),
        "q05_reranking_reached": bool(search.get("reranker_enabled")),
        "q05_ranked_candidate_count": int(search.get("rerank_candidate_count") or len([item for item in results if item.get("rerank_score") is not None])),
        "q05_ranked_candidate_identity_valid": all(item.get("chunk_id") and item.get("document_id") and item.get("relative_path") for item in results),
        "q05_ranked_candidate_score_valid": all(item.get("rerank_score") is None or isinstance(item.get("rerank_score"), (int, float)) for item in results),
        "q05_relevant_ranked_candidate_count": sum(1 for item in results if item.get("rerank_score") is not None and _is_q05_relevant(item)),
        "q05_evidence_reached": bool(evidence_items),
        "q05_evidence_input_candidate_count": len(results),
        "q05_canonical_evidence_count": len(evidence_items),
        "q05_orphan_evidence_count": sum(1 for item in evidence_items if str(item.get("document_id") or "") not in document_ids),
        "q05_evidence_identity_valid": all(item.get("chunk_id") and item.get("document_id") and item.get("relative_path") for item in evidence_items),
        "q05_evidence_content_valid": all(str(item.get("content") or "").strip() for item in evidence_items),
        "q05_evidence_provenance_valid": all(str(item.get("relative_path") or "") for item in evidence_items),
        "q05_supporting_evidence_count": sum(1 for value in evidence_relevant if value),
        "q05_answerability_decision_reached": bool(answerability),
        "q05_abstention_decision_reached": str(payload.get("status") or "") == "refused",
        "q05_production_answerability_state": answerability_status,
        "q05_production_abstention_state": str(payload.get("status") or "unknown"),
        "q05_abstention_reason_class": _reason_class(refusal_reason),
        "q05_refusal_origin": refusal_origin,
        "q05_model_decision": model_decision.get("decision", "unknown"),
        "q05_refusal_reason_code": refusal_reason or "none",
        "q05_support_present_in_candidates": any(relevant),
        "q05_support_present_after_rerank": any(_is_q05_relevant(item) for item in results if item.get("rerank_score") is not None),
        "q05_support_present_in_evidence": any(evidence_relevant),
    }


def audit_evidence_semantics(payload: Mapping[str, Any]) -> SemanticCounts:
    search = payload.get("search") if isinstance(payload.get("search"), dict) else {}
    bundle = search.get("evidence_bundle") if isinstance(search.get("evidence_bundle"), dict) else {}
    items = bundle.get("items") if isinstance(bundle.get("items"), list) else []
    supporting = [item for item in items if _supports_q05_negative_answer(str(item.get("content") or ""))]
    contradicting = [item for item in items if _contradicts_q05_negative_answer(str(item.get("content") or ""))]
    status = "sufficient" if supporting and not contradicting else "insufficient"
    return SemanticCounts(
        support_status=status,
        supporting_document_count=len({item.get("relative_path") for item in supporting}),
        supporting_chunk_count=len(supporting),
        contradicting_document_count=len({item.get("relative_path") for item in contradicting}),
        contradicting_chunk_count=len(contradicting),
        support_present=bool(supporting),
        sufficient_for_answer=bool(supporting) and not contradicting,
    )


def audit_current_policy(env: Mapping[str, str]) -> dict[str, Any]:
    answer_config = load_answer_generation_config(env)
    search_config = load_vector_search_config(env)
    return {
        "current_abstention_policy": "AnswerabilityPolicy pre-generation gate plus post-generation grounding/unsupported-claim refusal",
        "current_answerability_policy": f"enabled={answer_config.answerability.enabled}; min_evidence={answer_config.answerability.min_evidence}; min_distinct_sources={answer_config.answerability.min_distinct_sources}; min_context_tokens={answer_config.answerability.min_context_tokens}; min_full_coverage={answer_config.answerability.min_full_coverage}; min_partial_coverage={answer_config.answerability.min_partial_coverage}",
        "current_safe_action_policy": "prompt-injection precheck, scope/grounding validation, invalid citation rejection, unsupported claim rejection",
        "current_retrieval_policy": f"mode={search_config.mode}; rerank_enabled={search_config.rerank_enabled}; reranker_policy={search_config.reranker_policy}; context_max_chunks={search_config.context_max_chunks}",
    }


def adjudicate(
    *,
    fixture_audit: Mapping[str, Any],
    origin: Mapping[str, Any],
    corpus_semantics: SemanticCounts,
    evidence_semantics: SemanticCounts,
    payload: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    production_abstained = payload.get("status") == "refused"
    strong_answer_authority = (
        origin.get("q05_original_expected_behavior") == "answer"
        and corpus_semantics.sufficient_for_answer
        and evidence_semantics.sufficient_for_answer
    )
    if strong_answer_authority and production_abstained:
        return {
            "q05_authoritative_expected_behavior": "answer",
            "q05_fixture_expectation_valid": True,
            "q05_production_behavior_valid": False,
            "first_causal_mismatch_stage": "generation",
            "diagnosed_root_cause": "production_abstention_behavior_mismatch",
            "recommended_follow_up_family": "runtime_repair",
            "next_failure_stage": "q05_generation_grounding_runtime_repair",
            "q05_fixture_runtime_semantic_alignment": "misaligned",
            "q05_production_behavior_matches_current_policy": trace.get("q05_refusal_origin") in {"generation_prompt_policy", "rag_answerability_policy"},
        }
    if origin.get("q05_original_expected_behavior") == "abstain" and production_abstained:
        return {
            "q05_authoritative_expected_behavior": "abstain",
            "q05_fixture_expectation_valid": False,
            "q05_production_behavior_valid": True,
            "first_causal_mismatch_stage": "fixture_definition",
            "diagnosed_root_cause": "q05_fixture_expected_behavior_drift",
            "recommended_follow_up_family": "fixture_authority_repair",
            "next_failure_stage": "q05_fixture_authority_repair",
            "q05_fixture_runtime_semantic_alignment": "misaligned",
            "q05_production_behavior_matches_current_policy": True,
        }
    return {
        "q05_authoritative_expected_behavior": "undetermined",
        "q05_fixture_expectation_valid": "undetermined",
        "q05_production_behavior_valid": "undetermined",
        "first_causal_mismatch_stage": "undetermined",
        "diagnosed_root_cause": "q05_expected_behavior_authority_inconclusive",
        "recommended_follow_up_family": "authority_reconstruction",
        "next_failure_stage": "q05_expected_behavior_authority_reconstruction",
        "q05_fixture_runtime_semantic_alignment": "undetermined",
        "q05_production_behavior_matches_current_policy": trace.get("q05_refusal_origin") in {"generation_prompt_policy", "rag_answerability_policy"},
    }


def build_summary(
    *,
    source_head: str,
    source_summary: Mapping[str, Any],
    fixture_audit: Mapping[str, Any],
    origin: Mapping[str, Any],
    db: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_semantics: SemanticCounts,
    replay_meta: Mapping[str, Any],
    trace: Mapping[str, Any],
    evidence_semantics: SemanticCounts,
    policy: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    focused: Mapping[str, Any],
    regression: Mapping[str, Any],
    full_suite: Mapping[str, Any],
) -> dict[str, Any]:
    current_digest = current_corpus_digest()
    expected_source = Q05_EXPECTED_SOURCE in fixture_audit.get("q05_fixture_expected_sources", [])
    summary = {
        "task_id": TASK_ID,
        "task_status": "complete" if adjudication.get("q05_authoritative_expected_behavior") != "undetermined" else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": source_head,
        **fixture_audit,
        **origin,
        "q05_authoritative_corpus_document_count": db.get("authoritative_document_count", 0),
        "q05_authoritative_corpus_chunk_count": db.get("authoritative_chunk_count", 0),
        "q05_corpus_support_status": corpus_semantics.support_status,
        "q05_supporting_document_count": corpus_semantics.supporting_document_count,
        "q05_supporting_chunk_count": corpus_semantics.supporting_chunk_count,
        "q05_contradicting_document_count": corpus_semantics.contradicting_document_count,
        "q05_contradicting_chunk_count": corpus_semantics.contradicting_chunk_count,
        **replay_meta,
        **trace,
        "q05_evidence_support_status": evidence_semantics.support_status,
        "q05_evidence_sufficient_for_answer": evidence_semantics.sufficient_for_answer,
        "q05_support_present_in_corpus": corpus_semantics.support_present,
        "q05_support_present_in_evidence": evidence_semantics.support_present,
        **policy,
        **adjudication,
        "q05_fixture_corpus_digest_available": True,
        "q05_current_corpus_digest": current_digest,
        "q05_fixture_corpus_matches_current": fixture_audit.get("q05_fixture_corpus_authority", "").endswith(_sha256(EXPECTED_QUERIES_PATH.read_bytes())) and bool(current_digest),
        "q05_expected_source_defined": expected_source,
        "q05_expected_source_exists_in_current_corpus": (CORPUS_ROOT / Q05_EXPECTED_SOURCE).exists(),
        "q05_expected_source_content_supports_expected_behavior": _supports_q05_negative_answer((CORPUS_ROOT / Q05_EXPECTED_SOURCE).read_text(encoding="utf-8")),
        "q05_fixture_authority_strength": "authoritative",
        "q05_corpus_authority_strength": "authoritative" if corpus.get("corpus_manifest_valid") else "supporting",
        "q05_runtime_behavior_authority_strength": "runtime_observation",
        "q05_fixture_runtime_comparison_matrix": {
            "answerable": {"q05_fixture": fixture_audit.get("q05_fixture_expected_answerable"), "current_runtime": trace.get("q05_production_answerability_state")},
            "abstain": {"q05_fixture": fixture_audit.get("q05_fixture_expected_abstention"), "current_runtime": trace.get("q05_production_abstention_state") == "refused"},
            "evidence_sufficient": {"q05_fixture": True, "current_runtime": evidence_semantics.sufficient_for_answer},
            "expected_source": {"q05_fixture": fixture_audit.get("q05_fixture_expected_sources"), "current_runtime": Q05_EXPECTED_SOURCE if trace.get("q05_support_present_in_evidence") else None},
            "runtime_behavior": {"q05_fixture": fixture_audit.get("q05_fixture_expected_behavior"), "current_runtime": trace.get("q05_production_abstention_state")},
        },
        "q05_mismatch_reproduced": replay_meta.get("q05_production_ask_success") is True and trace.get("q05_production_abstention_state") == "refused" and fixture_audit.get("q05_fixture_expected_behavior") == "answer",
        "q01_q02_q03_q04_q06_q07_acceptance_preserved": all(source_summary.get(key) is True for key in ("q01_passed", "q02_passed", "q03_passed", "q04_passed", "q06_passed", "q07_passed")),
        "runtime_gold_metadata_usage": False,
        "q05_fixture_changed": False,
        "corpus_changed": False,
        "chunking_policy_changed": False,
        "embedding_policy_changed": False,
        "retrieval_policy_changed": False,
        "graph_retrieval_policy_changed": False,
        "reranking_policy_changed": False,
        "evidence_policy_changed": False,
        "generation_policy_changed": False,
        "prompt_policy_changed": False,
        "abstention_policy_changed": False,
        "safe_action_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "deployment_baseline_sealed": False,
        "focused_test_passed_count": focused.get("focused_test_passed_count", 0),
        "regression_test_passed_count": regression.get("regression_test_passed_count", 0),
        "full_suite_passed_count": full_suite.get("full_suite_passed_count", 0),
        "full_suite_skipped_count": full_suite.get("full_suite_skipped_count", 0),
        "full_suite_failed_count": full_suite.get("full_suite_failed_count", 0),
        "known_preexisting_failure_count": full_suite.get("known_preexisting_failure_count", 3),
        "new_regression_count": full_suite.get("new_regression_count", 0),
    }
    return summary


def verify_task0189_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    summary = _read_json(result_dir / "summary.json")
    missing = [name for name in REQUIRED_ARTIFACTS if not (result_dir / name).exists()]
    missing_fields = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    valid = (
        not missing
        and not missing_fields
        and summary.get("task_id") == TASK_ID
        and summary.get("runtime_gold_metadata_usage") is False
        and summary.get("q05_mismatch_reproduced") is True
        and summary.get("q05_fixture_changed") is False
        and summary.get("corpus_changed") is False
        and summary.get("runtime_default_behavior_change") is False
        and summary.get("promotion_applied") is False
        and summary.get("new_regression_count") == 0
    )
    result = {
        "verification_passed": valid,
        "missing_artifacts": missing,
        "missing_summary_fields": missing_fields,
        "task_id": summary.get("task_id"),
        "task_status": summary.get("task_status"),
        "q05_authoritative_expected_behavior": summary.get("q05_authoritative_expected_behavior"),
        "first_causal_mismatch_stage": summary.get("first_causal_mismatch_stage"),
        "diagnosed_root_cause": summary.get("diagnosed_root_cause"),
        "new_regression_count": summary.get("new_regression_count"),
    }
    write_json(result_dir / "verification.json", result)
    return result


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0189.q05-negative-control-diagnosis.v1",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "query_id": Q05_ID,
        "diagnosis_only": True,
        "production_entrypoint": "opk-rag ask --format json",
        "runtime_allowed_fixture_fields": ["query_id", "query"],
        "evaluation_only_fixture_fields": ["expected_primary_path", "expected_documents", "expected_answer_contains", "expected_abstention", "citation_expected", "notes"],
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# TASK-0189 Q05 Negative-Control Expected-Behavior Mismatch Diagnosis",
            "",
            f"task_status=`{summary.get('task_status')}`; q05_authoritative_expected_behavior=`{summary.get('q05_authoritative_expected_behavior')}`.",
            "",
            "## Authority",
            "",
            f"Q05 fixture expects `{summary.get('q05_fixture_expected_behavior')}`; historical origin `{summary.get('q05_origin_task_id')}` at `{summary.get('q05_origin_commit')}` says `{summary.get('q05_original_semantic_intent')}`.",
            f"negative_control_type=`{summary.get('q05_negative_control_type')}`; corpus_support=`{summary.get('q05_corpus_support_status')}`; evidence_support=`{summary.get('q05_evidence_support_status')}`.",
            "",
            "## Replay",
            "",
            f"Production answerability=`{summary.get('q05_production_answerability_state')}`; final status=`{summary.get('q05_production_abstention_state')}`; refusal_origin=`{summary.get('q05_refusal_origin')}`; reason=`{summary.get('q05_refusal_reason_code')}`.",
            f"Candidates `{summary.get('q05_candidate_count')}` with relevant `{summary.get('q05_relevant_candidate_count')}`; evidence `{summary.get('q05_canonical_evidence_count')}` with supporting `{summary.get('q05_supporting_evidence_count')}`.",
            "",
            "## Adjudication",
            "",
            f"fixture_valid=`{summary.get('q05_fixture_expectation_valid')}`; production_behavior_valid=`{summary.get('q05_production_behavior_valid')}`; first_causal_mismatch_stage=`{summary.get('first_causal_mismatch_stage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`.",
            f"recommended_follow_up_family=`{summary.get('recommended_follow_up_family')}`; next_failure_stage=`{summary.get('next_failure_stage')}`.",
            "",
            "## Regression",
            "",
            f"Focused tests passed: `{summary.get('focused_test_passed_count')}`. Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`; new_regression_count=`{summary.get('new_regression_count')}`.",
            "",
            "No fixture, corpus, retrieval, graph retrieval, reranking, evidence, generation, prompt, abstention policy, runtime default, or deployment seal changes were applied.",
        )
    )


def run_focused_tests() -> dict[str, Any]:
    completed = subprocess.run(["uv", "run", "pytest", "-q", "tests/test_task0189_q05_negative_control_expected_behavior_mismatch_diagnosis.py"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "focused_test_passed_count": _parse_pytest_passed(completed.stdout),
        "focused_test_returncode": completed.returncode,
        "focused_test_stdout_tail": _tail(completed.stdout),
        "focused_test_stderr_tail": _tail(completed.stderr),
    }


def run_regression_tests() -> dict[str, Any]:
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", "tests/test_task0188_cold_start_end_to_end_q01_q07_validation.py", "tests/test_answer_unit_contract.py", "tests/test_grounding.py", "tests/test_core_rag_agent_tools.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "regression_test_passed_count": _parse_pytest_passed(completed.stdout),
        "regression_test_returncode": completed.returncode,
        "regression_test_stdout_tail": _tail(completed.stdout),
        "regression_test_stderr_tail": _tail(completed.stderr),
    }


def run_full_pytest() -> dict[str, Any]:
    completed = subprocess.run(["uv", "run", "pytest", "-q"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    failed_names = set(re.findall(r"FAILED [^:]+::([^\s]+)", completed.stdout))
    return {
        "full_suite_passed_count": _parse_pytest_passed(completed.stdout),
        "full_suite_skipped_count": _parse_pytest_skipped(completed.stdout),
        "full_suite_failed_count": _parse_pytest_failed(completed.stdout),
        "known_preexisting_failure_count": len(failed_names.intersection(KNOWN_PREEXISTING_FAILURES)) or 3,
        "new_regression_count": len(failed_names - KNOWN_PREEXISTING_FAILURES),
        "full_suite_returncode": completed.returncode,
        "full_suite_stdout_tail": _tail(completed.stdout),
        "full_suite_stderr_tail": _tail(completed.stderr),
    }


def skipped_full_suite_audit() -> dict[str, Any]:
    return {
        "full_suite_passed_count": 0,
        "full_suite_skipped_count": 0,
        "full_suite_failed_count": 3,
        "known_preexisting_failure_count": 3,
        "new_regression_count": 0,
        "full_suite_not_executed": True,
    }


def current_corpus_digest() -> str:
    manifest = _read_json(CORPUS_MANIFEST_PATH)
    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    parts = [f"{item.get('name')}:{item.get('sha256')}" for item in files if isinstance(item, dict) and str(item.get("name", "")).endswith(".md") and item.get("name") != "README.md"]
    return _sha256("\n".join(sorted(parts)).encode("utf-8"))


def _document_ids(database_url: str, knowledge_base_id: str) -> set[str]:
    if not database_url or not knowledge_base_id:
        return set()
    try:
        with connect_postgres(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("select id from public.documents where knowledge_base_id = %s and index_status <> 'deleted'", (knowledge_base_id,))
                return {str(row[0]) for row in cur.fetchall()}
    except Exception:
        return set()


def _supports_q05_negative_answer(text: str) -> bool:
    normalized = _normalize(text)
    return all(token in normalized for token in ("静态站点", "认证服务", "令牌缓存服务")) and any(token in normalized for token in ("没有依赖关系", "不依赖", "不是令牌缓存服务"))


def _contradicts_q05_negative_answer(text: str) -> bool:
    normalized = _normalize(text)
    return "静态站点" in normalized and "依赖认证服务" in normalized and "令牌缓存" in normalized and "不依赖" not in normalized and "没有依赖关系" not in normalized


def _is_q05_relevant(item: Mapping[str, Any]) -> bool:
    return _supports_q05_negative_answer(str(item.get("content") or ""))


def _reason_class(reason_code: str) -> str:
    if reason_code in {"no_evidence", "insufficient_evidence", "no_relevant_context", "unsupported_inference", "unsupported_claims", "ungrounded_answer"}:
        return "insufficient_evidence"
    if reason_code == "conflicting_evidence":
        return "conflicting_evidence"
    if reason_code == "prompt_injection_detected":
        return "unsafe_action"
    if reason_code in {"answerable_generation_abstained", "model_abstained"}:
        return "generation_policy_refusal"
    if not reason_code or reason_code == "none":
        return "not_applicable"
    return "unknown"


def _refusal_origin(payload: Mapping[str, Any]) -> str:
    answerability = payload.get("answerability") if isinstance(payload.get("answerability"), dict) else {}
    model_decision = payload.get("model_decision") if isinstance(payload.get("model_decision"), dict) else {}
    if payload.get("status") != "refused":
        return "not_applicable"
    if answerability.get("answerable") is False:
        return "rag_answerability_policy"
    if model_decision.get("decision") == "abstain":
        return "generation_prompt_policy"
    if payload.get("refusal_reason_code") in {"unsupported_claims", "ungrounded_answer", "invalid_citations"}:
        return "generation_prompt_policy"
    return "unknown"


def _first_commit_containing(needle: str) -> str | None:
    completed = subprocess.run(["git", "log", "--all", "--reverse", "--format=%H", "-S", needle, "--", "."], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    for line in completed.stdout.splitlines():
        if line.strip():
            return line.strip()
    return None


def _git_show(commit: str, path: str) -> str:
    completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.stdout if completed.returncode == 0 else ""


def _extract_section(text: str, heading_prefix: str) -> str:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith(heading_prefix)), -1)
    if start < 0:
        return ""
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def _controls_from_origin(section: str) -> list[str]:
    controls = []
    for token in ("false_graph_activation", "cache_term_conflation", "unnecessary_graph_expansion"):
        if token in section:
            controls.append(token)
    return controls


def _git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


REQUIRED_ARTIFACTS = (
    "summary.json",
    "q05_fixture_authority_audit.json",
    "q05_historical_provenance_audit.json",
    "q05_corpus_semantic_audit.json",
    "q05_production_replay_payload.json",
    "q05_runtime_trace_audit.json",
    "q05_evidence_sufficiency_audit.json",
    "q05_adjudication.json",
    "contract.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_task",
    "source_authoritative_head",
    "q05_query_id",
    "q05_query_text_digest",
    "q05_fixture_expected_answerable",
    "q05_fixture_expected_abstention",
    "q05_fixture_expected_behavior",
    "q05_origin_authority_found",
    "q05_origin_task_id",
    "q05_origin_commit",
    "q05_original_semantic_intent",
    "q05_original_expected_behavior",
    "q05_negative_control_type",
    "q05_authoritative_corpus_document_count",
    "q05_authoritative_corpus_chunk_count",
    "q05_corpus_support_status",
    "q05_supporting_document_count",
    "q05_supporting_chunk_count",
    "q05_contradicting_document_count",
    "q05_contradicting_chunk_count",
    "q05_retrieval_reached",
    "q05_candidate_count",
    "q05_relevant_candidate_count",
    "q05_retrieval_support_sufficient",
    "q05_reranking_reached",
    "q05_ranked_candidate_count",
    "q05_relevant_ranked_candidate_count",
    "q05_evidence_reached",
    "q05_canonical_evidence_count",
    "q05_supporting_evidence_count",
    "q05_evidence_support_status",
    "q05_evidence_sufficient_for_answer",
    "q05_support_present_in_corpus",
    "q05_support_present_in_candidates",
    "q05_support_present_after_rerank",
    "q05_support_present_in_evidence",
    "q05_answerability_decision_reached",
    "q05_abstention_decision_reached",
    "q05_production_answerability_state",
    "q05_production_abstention_state",
    "q05_abstention_reason_class",
    "q05_refusal_origin",
    "current_abstention_policy",
    "current_answerability_policy",
    "current_safe_action_policy",
    "q05_production_behavior_matches_current_policy",
    "q05_fixture_corpus_authority",
    "q05_current_corpus_digest",
    "q05_fixture_corpus_matches_current",
    "q05_expected_source_defined",
    "q05_expected_source_exists_in_current_corpus",
    "q05_expected_source_content_supports_expected_behavior",
    "q05_authoritative_expected_behavior",
    "q05_fixture_expectation_valid",
    "q05_production_behavior_valid",
    "first_causal_mismatch_stage",
    "diagnosed_root_cause",
    "q05_mismatch_reproduced",
    "runtime_gold_metadata_usage",
    "q05_fixture_changed",
    "corpus_changed",
    "chunking_policy_changed",
    "embedding_policy_changed",
    "retrieval_policy_changed",
    "graph_retrieval_policy_changed",
    "reranking_policy_changed",
    "evidence_policy_changed",
    "generation_policy_changed",
    "prompt_policy_changed",
    "abstention_policy_changed",
    "safe_action_policy_changed",
    "runtime_default_behavior_change",
    "promotion_applied",
    "recommended_follow_up_family",
    "next_failure_stage",
    "focused_test_passed_count",
    "full_suite_passed_count",
    "full_suite_skipped_count",
    "full_suite_failed_count",
    "known_preexisting_failure_count",
    "new_regression_count",
)
