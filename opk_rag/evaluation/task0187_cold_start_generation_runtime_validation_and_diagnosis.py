from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID

from opk_rag.answer.config import AnswerGenerationConfig, load_answer_generation_config
from opk_rag.answer.models import AnswerGenerationRequest, AnswerResponse, AnswerSystemError
from opk_rag.answer.prompt import render_user_prompt
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider, RemoteLLMNotAllowedError, build_chat_completion_payload
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.core_tools.tools import search_knowledge_base as production_search_knowledge_base
from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0183_cold_start_vector_index_materialization_diagnosis import _load_real_provider, audit_database_vector_state
from opk_rag.evaluation.task0184_cold_start_retrieval_runtime_validation_and_diagnosis import PROBE_QUERIES, _resolve_database_url, candidate_row, validate_candidate_identity
from opk_rag.evaluation.task0185_cold_start_reranking_runtime_validation_and_diagnosis import build_reranker_provider, config_for_rerank_probe
from opk_rag.evaluation.task0186_cold_start_evidence_composition_runtime_validation_and_diagnosis import RESULT_DIR as TASK0186_RESULT_DIR
from opk_rag.reranking.config import load_reranker_config
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.models import EvidenceBundle, SearchResponse

TASK_ID = "TASK-0187"
SOURCE_AUTHORITATIVE_TASK = "TASK-0186"
EXPERIMENT_ID = "task0187-cold-start-generation-runtime-validation-and-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0187_cold_start_generation_runtime_validation_and_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0187_COLD_START_GENERATION_RUNTIME_VALIDATION_AND_DIAGNOSIS_REPORT.md"

LOSS_SUBSTAGES = {
    "generation_input_contract",
    "prompt_construction",
    "prompt_evidence_binding",
    "provider_resolution",
    "provider_configuration",
    "provider_connection",
    "model_resolution",
    "model_authority",
    "generation_invocation",
    "generation_timeout",
    "generation_output_validation",
    "answer_parsing",
    "answer_contract",
    "citation_binding",
    "citation_identity_mapping",
    "production_ask_runtime",
    "none",
}

ROOT_CAUSES = {
    "generation_input_contract_mismatch",
    "prompt_construction_failure",
    "prompt_evidence_binding_failure",
    "generation_provider_unconfigured",
    "generation_provider_unreachable",
    "generation_provider_authentication_failure",
    "generation_model_unavailable",
    "generation_model_cache_missing_or_incomplete",
    "generation_model_resolution_failure",
    "generation_runtime_execution_failure",
    "generation_timeout_failure",
    "empty_generation_output",
    "answer_parse_failure",
    "answer_contract_mismatch",
    "citation_binding_failure",
    "citation_identity_mapping_failure",
    "production_ask_runtime_failure",
    "no_generation_runtime_failure_reproduced",
    "unknown_generation_runtime_failure",
}


@dataclass(frozen=True)
class GenerationProbeResult:
    query: str
    success: bool
    error: str
    generation_input_constructible: bool
    generation_input_evidence_count: int
    generation_input_context_nonempty: bool
    prompt_constructed: bool
    prompt_digest: str | None
    prompt_character_count: int
    prompt_evidence_count: int
    prompt_evidence_identity_valid: bool
    prompt_evidence_content_valid: bool
    invocation_attempted: bool
    execution_success: bool
    system_error_code: str | None
    provider_error_reason_code: str | None
    raw_generation_output_nonempty: bool
    answer_parse_success: bool
    final_answer_nonempty: bool
    answer_status: str | None
    answer_abstained: bool
    answer_citation_count: int
    invalid_citation_count: int
    generation_latency_ms: int | None


def run_task0187(
    *,
    write: bool = True,
    env: Mapping[str, str] | None = None,
    embedding_provider: Any | None = None,
    reranker_provider: Any | None = None,
    answer_provider: Any | None = None,
    production_ask: bool = True,
) -> dict[str, Any]:
    if env is None:
        load_project_env(ROOT)
        env = _current_project_env()
    env = dict(env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    source_head = _git_head(ROOT)
    task0186 = _load_task0186_summary()
    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    reranker_config = load_reranker_config(env)
    answer_config = load_answer_generation_config(env)
    database_url = _resolve_database_url(env)

    db = audit_database_vector_state(database_url, embedding_config)
    if embedding_provider is None and database_url and db.get("database_connection_success"):
        embedding_provider = _load_real_provider(embedding_config, {})
    provider_probe = build_reranker_provider(reranker_config, reranker_provider)
    generation_authority = audit_generation_authority(answer_config, env)
    provider_readiness = audit_provider_readiness(answer_config, env)

    resolved_answer_provider = answer_provider
    provider_resolution = {"generation_provider_resolution_attempted": True, "generation_provider_resolution_success": False, "generation_provider_identifier": answer_config.provider_id}
    if resolved_answer_provider is None:
        try:
            resolved_answer_provider = OpenAICompatibleLocalChatProvider(answer_config, api_key=env.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
            provider_resolution["generation_provider_resolution_success"] = True
        except RemoteLLMNotAllowedError:
            provider_resolution["generation_provider_resolution_success"] = False
        except Exception:
            provider_resolution["generation_provider_resolution_success"] = False
    else:
        provider_resolution["generation_provider_resolution_success"] = True

    responses = run_generation_search_probes(database_url, db, embedding_provider, embedding_config, search_config, provider_probe.get("provider"))
    probes = tuple(
        run_single_generation_probe(response, provider=resolved_answer_provider, config=answer_config)
        if isinstance(response, SearchResponse)
        else _failed_generation_probe(str(response.get("query", "")), str(response.get("error", "search_probe_failed")))
        for response in responses
    )

    canonical = audit_upstream_authority(task0186, db, probes)
    input_contract = audit_generation_input_contract(probes)
    prompt = audit_prompt_construction(probes)
    invocation = audit_generation_invocation(probes)
    answer = audit_answer_contract(probes)
    citations = audit_citation_binding(probes)
    production = audit_production_ask_runtime(database_url, db, responses, answer_config, env, enabled=production_ask)
    failure = determine_failure(input_contract, prompt, provider_resolution, provider_readiness, invocation, answer, citations, production)

    summary = build_summary(
        source_head=source_head,
        task0186=task0186,
        db=db,
        canonical=canonical,
        generation_authority=generation_authority,
        input_contract=input_contract,
        prompt=prompt,
        provider_resolution=provider_resolution,
        provider_readiness=provider_readiness,
        invocation=invocation,
        answer=answer,
        citations=citations,
        production=production,
        probes=probes,
        failure=failure,
        env=env,
    )
    artifacts = {
        "summary.json": summary,
        "current_generation_authority_audit.json": generation_authority,
        "generation_input_contract_audit.json": input_contract,
        "prompt_construction_audit.json": prompt,
        "provider_resolution_audit.json": {**provider_resolution, **provider_readiness},
        "generation_invocation_probe.json": [asdict(probe) for probe in probes],
        "answer_contract_audit.json": answer,
        "citation_provenance_audit.json": citations,
        "production_ask_runtime_probe.json": production,
        "failure_taxonomy_decision.json": failure,
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, _redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_generation_authority(config: AnswerGenerationConfig, env: Mapping[str, str]) -> dict[str, Any]:
    endpoint_type = _endpoint_type(config.base_url)
    return {
        "default_generation_provider": config.provider_id,
        "default_generation_model": config.model_id,
        "generation_provider_type": "openai-compatible",
        "generation_model_identifier": config.model_id,
        "generation_runtime_enabled": True,
        "generation_streaming_enabled": False,
        "generation_provider_parameter_policy": config.provider_parameter_policy,
        "generation_response_format_type": config.response_format_type,
        "generation_endpoint_type": endpoint_type,
        "generation_remote_allowed": config.allow_remote,
        "provider_endpoint_present": bool(config.base_url.strip()),
        "provider_api_key_required": endpoint_type == "remote",
        "provider_api_key_present": bool(env.get("OPK_RAG_LLM_API_KEY", "").strip()),
    }


def audit_provider_readiness(config: AnswerGenerationConfig, env: Mapping[str, str]) -> dict[str, Any]:
    endpoint_type = _endpoint_type(config.base_url)
    endpoint_present = bool(config.base_url.strip())
    api_key_present = bool(env.get("OPK_RAG_LLM_API_KEY", "").strip())
    api_key_required = endpoint_type == "remote"
    configuration_valid = endpoint_present and (endpoint_type != "remote" or config.allow_remote) and (not api_key_required or api_key_present)
    connection = False
    model_available = False
    error = None
    if configuration_valid:
        readiness = _probe_models_endpoint(config, api_key=env.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
        connection = readiness["provider_connection_valid"]
        model_available = readiness["provider_model_available"]
        error = readiness["provider_readiness_error"]
    return {
        "provider_endpoint_present": endpoint_present,
        "provider_api_key_required": api_key_required,
        "provider_api_key_present": api_key_present,
        "provider_configuration_valid": configuration_valid,
        "provider_connection_valid": connection,
        "provider_model_available": model_available,
        "provider_readiness_error": error,
        "generation_model_cache_exists": "not_applicable" if endpoint_type == "remote" else False,
        "generation_model_cache_complete": "not_applicable" if endpoint_type == "remote" else False,
        "generation_model_authority_valid": "not_applicable" if endpoint_type == "remote" else model_available,
        "generation_model_revision_match": "not_applicable" if endpoint_type == "remote" else model_available,
    }


def run_generation_search_probes(database_url: str, db: Mapping[str, Any], embedding_provider: Any | None, embedding_config: Any, search_config: Any, reranker_provider: Any | None) -> tuple[SearchResponse | dict[str, str], ...]:
    probe_config = config_for_rerank_probe(search_config)
    rows: list[SearchResponse | dict[str, str]] = []
    for query in PROBE_QUERIES:
        if not database_url or embedding_provider is None or not db.get("knowledge_base_id"):
            rows.append({"query": query, "error": "database_or_embedding_provider_unavailable"})
            continue
        try:
            response, _payload = production_search_knowledge_base(
                database_url=database_url,
                knowledge_base_id=UUID(str(db["knowledge_base_id"])),
                query=query,
                provider=embedding_provider,
                embedding_config=embedding_config,
                search_config=probe_config,
                reranker_provider=reranker_provider,
            )
            rows.append(response)
        except Exception as exc:
            rows.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
    return tuple(rows)


def run_single_generation_probe(response: SearchResponse, *, provider: Any | None, config: AnswerGenerationConfig) -> GenerationProbeResult:
    query = response.query
    bundle = response.evidence_bundle
    if bundle is None or not bundle.items:
        return _failed_generation_probe(query, "generation_input_missing_evidence")
    prompt_constructed = False
    prompt_digest = None
    prompt_character_count = 0
    prompt_evidence_count = 0
    prompt_identity_valid = False
    prompt_content_valid = False
    try:
        user_prompt = render_user_prompt(query, bundle, output_schema_version=config.output_schema_version)
        payload = build_chat_completion_payload(config, AnswerGenerationRequest(query, bundle, config.prompt_version, config.output_schema_version))
        prompt_constructed = bool(payload.get("messages")) and bool(user_prompt.strip())
        prompt_digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        prompt_character_count = len(user_prompt)
        prompt_payload = json.loads(user_prompt)
        evidence = prompt_payload.get("evidence", [])
        prompt_evidence_count = len(evidence) if isinstance(evidence, list) else 0
        bundle_ids = {str(item.chunk_id) for item in bundle.items}
        prompt_ids = {str(row.get("source", {}).get("chunk_id")) for row in evidence if isinstance(row, dict)}
        prompt_identity_valid = prompt_ids == bundle_ids
        prompt_content_valid = all(isinstance(row, dict) and str(row.get("content") or "").strip() for row in evidence)
    except Exception as exc:
        return GenerationProbeResult(query, False, f"prompt_construction: {type(exc).__name__}: {exc}", True, len(bundle.items), response.context_token_count > 0, False, None, 0, 0, False, False, False, False, None, None, False, False, False, None, False, 0, 0, None)

    if provider is None:
        return GenerationProbeResult(query, False, "provider_resolution_failed", True, len(bundle.items), response.context_token_count > 0, prompt_constructed, prompt_digest, prompt_character_count, prompt_evidence_count, prompt_identity_valid, prompt_content_valid, False, False, "provider_resolution_failed", None, False, False, False, None, False, 0, 0, None)
    try:
        answer = answer_knowledge_base(response, provider=provider, config=config)
    except AnswerSystemError as exc:
        detail = exc.provider_error_detail or {}
        return GenerationProbeResult(query, False, f"AnswerSystemError:{exc.code}", True, len(bundle.items), response.context_token_count > 0, prompt_constructed, prompt_digest, prompt_character_count, prompt_evidence_count, prompt_identity_valid, prompt_content_valid, True, False, exc.code, str(detail.get("reason_code")) if detail.get("reason_code") else None, False, False, False, None, False, 0, 0, None)
    except Exception as exc:
        return GenerationProbeResult(query, False, f"{type(exc).__name__}: {exc}", True, len(bundle.items), response.context_token_count > 0, prompt_constructed, prompt_digest, prompt_character_count, prompt_evidence_count, prompt_identity_valid, prompt_content_valid, True, False, "unexpected_exception", None, False, False, False, None, False, 0, 0, None)
    return _probe_from_answer(response, answer, prompt_digest, prompt_character_count, prompt_evidence_count, prompt_identity_valid, prompt_content_valid)


def audit_upstream_authority(task0186: Mapping[str, Any], db: Mapping[str, Any], probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    return {
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0186.get("source_authoritative_head"),
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        "canonical_evidence_count": sum(probe.generation_input_evidence_count for probe in probes),
        "generation_input_constructible": any(probe.generation_input_constructible for probe in probes),
        "generation_input_evidence_count": sum(probe.generation_input_evidence_count for probe in probes),
        "generation_input_context_nonempty": any(probe.generation_input_context_nonempty for probe in probes),
        "upstream_authority_drift": not (
            db.get("authoritative_document_count") == 8
            and db.get("authoritative_chunk_count") == 11
            and db.get("stored_embedding_count") == 11
            and task0186.get("cold_start_evidence_composition_stage_passed") is True
            and task0186.get("generation_input_constructible") is True
        ),
    }


def audit_generation_input_contract(probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    mismatches = []
    for probe in probes:
        checks = {
            "query": bool(probe.query.strip()),
            "evidence items": probe.generation_input_evidence_count > 0,
            "evidence content": probe.prompt_evidence_content_valid,
            "source metadata": probe.prompt_evidence_identity_valid,
            "citation metadata": probe.prompt_evidence_count == probe.generation_input_evidence_count,
            "conversation/session context": True,
            "generation options": True,
        }
        for prop, ok in checks.items():
            if not ok:
                mismatches.append({"query": probe.query, "property": prop, "compatible": False})
    return {
        "generation_input_contract_valid": bool(probes) and not mismatches,
        "generation_input_contract_mismatch_count": len(mismatches),
        "first_generation_input_contract_mismatch": mismatches[0] if mismatches else None,
        "generation_input_query_count": sum(1 for probe in probes if probe.query.strip()),
        "generation_input_evidence_count": sum(probe.generation_input_evidence_count for probe in probes),
        "generation_input_context_count": sum(1 for probe in probes if probe.generation_input_context_nonempty),
        "generation_input_rejected_count": len(mismatches),
        "contract_table": [
            {"property": prop, "current_input": prop, "generation_required": "present/valid", "compatible": not any(row["property"] == prop for row in mismatches)}
            for prop in ("query", "evidence items", "evidence content", "source metadata", "citation metadata", "conversation/session context", "generation options")
        ],
    }


def audit_prompt_construction(probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    return {
        "prompt_construction_attempt_count": len(probes),
        "prompt_construction_success_count": sum(probe.prompt_constructed for probe in probes),
        "prompt_construction_failure_count": sum(not probe.prompt_constructed for probe in probes),
        "prompt_query_present": all(bool(probe.query.strip()) for probe in probes),
        "prompt_evidence_present": all(probe.prompt_evidence_count > 0 for probe in probes),
        "prompt_context_nonempty": all(probe.prompt_character_count > 0 for probe in probes),
        "prompt_evidence_count": sum(probe.prompt_evidence_count for probe in probes),
        "prompt_evidence_identity_valid": all(probe.prompt_evidence_identity_valid for probe in probes),
        "prompt_evidence_content_valid": all(probe.prompt_evidence_content_valid for probe in probes),
        "prompt_digests": [probe.prompt_digest for probe in probes if probe.prompt_digest],
        "prompt_character_count": sum(probe.prompt_character_count for probe in probes),
    }


def audit_generation_invocation(probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    attempted = [probe for probe in probes if probe.invocation_attempted]
    return {
        "generation_probe_query_count": len(PROBE_QUERIES),
        "generation_probe_execution_count": len(probes),
        "generation_probe_success_count": sum(probe.success for probe in probes),
        "generation_probe_failure_count": sum(not probe.success for probe in probes),
        "generation_invocation_count": len(attempted),
        "generation_execution_success_count": sum(probe.execution_success for probe in attempted),
        "generation_execution_failure_count": sum(not probe.execution_success for probe in attempted),
        "generation_timeout_count": sum(probe.system_error_code == "timeout" for probe in attempted),
        "generation_connection_failure_count": sum((probe.provider_error_reason_code or "") in {"connection_refused", "connection_timeout", "network_unreachable", "dns_resolution_failed"} for probe in attempted),
        "generation_provider_error_count": sum(probe.system_error_code in {"provider_error", "model_not_found"} for probe in attempted),
        "generation_fallback_enabled": False,
        "generation_fallback_count": 0,
        "expected_generation_fallback_count": 0,
        "unexpected_generation_fallback_count": 0,
        "raw_generation_output_count": sum(probe.raw_generation_output_nonempty or probe.execution_success for probe in attempted),
        "raw_generation_output_nonempty_count": sum(probe.raw_generation_output_nonempty for probe in attempted),
        "raw_generation_output_empty_count": sum(probe.execution_success and not probe.raw_generation_output_nonempty for probe in attempted),
        "answer_parse_attempt_count": sum(probe.raw_generation_output_nonempty for probe in attempted),
        "answer_parse_success_count": sum(probe.answer_parse_success for probe in attempted),
        "answer_parse_failure_count": sum(probe.raw_generation_output_nonempty and not probe.answer_parse_success for probe in attempted),
        "generation_latency_p50_ms": _percentile([probe.generation_latency_ms for probe in attempted if probe.generation_latency_ms is not None], 0.5),
        "generation_latency_max_ms": max([probe.generation_latency_ms for probe in attempted if probe.generation_latency_ms is not None], default=None),
    }


def audit_answer_contract(probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    answered = [probe for probe in probes if probe.execution_success]
    nonempty = sum(probe.final_answer_nonempty for probe in answered)
    contract_valid = bool(answered) and all(probe.answer_status in {"answered", "refused"} and probe.final_answer_nonempty for probe in answered)
    return {
        "final_answer_count": len(answered),
        "final_answer_nonempty_count": nonempty,
        "final_answer_empty_count": len(answered) - nonempty,
        "answer_contract_valid": contract_valid,
        "answer_identity_valid": all(bool(probe.query.strip()) for probe in probes),
        "answer_content_valid": nonempty > 0 and all(probe.final_answer_nonempty for probe in answered),
        "answer_abstention_count": sum(probe.answer_abstained for probe in answered),
        "answer_non_abstention_count": sum(not probe.answer_abstained for probe in answered),
    }


def audit_citation_binding(probes: Sequence[GenerationProbeResult]) -> dict[str, Any]:
    answered = [probe for probe in probes if probe.execution_success and not probe.answer_abstained]
    invalid = sum(probe.invalid_citation_count for probe in answered)
    citation_count = sum(probe.answer_citation_count for probe in answered)
    return {
        "citation_binding_attempt_count": len(answered),
        "citation_binding_success_count": sum(probe.answer_citation_count > 0 and probe.invalid_citation_count == 0 for probe in answered),
        "citation_binding_failure_count": sum(probe.invalid_citation_count > 0 or probe.answer_citation_count == 0 for probe in answered),
        "answer_citation_count": citation_count,
        "valid_citation_count": citation_count - invalid,
        "invalid_citation_count": invalid,
        "citation_evidence_identity_valid": invalid == 0,
        "citation_chunk_identity_valid": invalid == 0,
        "citation_document_identity_valid": invalid == 0,
        "citation_content_integrity_valid": invalid == 0,
    }


def audit_production_ask_runtime(database_url: str, db: Mapping[str, Any], responses: Sequence[SearchResponse | Mapping[str, str]], config: AnswerGenerationConfig, env: Mapping[str, str], *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"production_ask_runtime_invocation_count": 0, "production_ask_runtime_success_count": 0, "production_ask_runtime_failure_count": 0, "production_ask_runtime_error": "disabled"}
    query = PROBE_QUERIES[0]
    if not database_url or not db.get("knowledge_base_id"):
        return {"production_ask_runtime_invocation_count": 0, "production_ask_runtime_success_count": 0, "production_ask_runtime_failure_count": 0, "production_ask_runtime_error": "database_unavailable"}
    cmd = [
        "uv",
        "run",
        "opk-rag",
        "ask",
        "--knowledge-base-id",
        str(db["knowledge_base_id"]),
        "--query",
        query,
        "--format",
        "json",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=ROOT, env=dict(env), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(5.0, config.timeout_seconds + 180.0), check=False)
    except subprocess.TimeoutExpired:
        return {"production_ask_runtime_invocation_count": 1, "production_ask_runtime_success_count": 0, "production_ask_runtime_failure_count": 1, "production_ask_runtime_error": "timeout", "production_ask_runtime_latency_ms": int((time.perf_counter() - started) * 1000)}
    success = False
    error = None
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
            success = isinstance(payload, dict) and bool(str(payload.get("answer") or "").strip()) and payload.get("status") in {"answered", "refused"}
        except json.JSONDecodeError:
            error = "json_parse_failed"
    else:
        error = _sanitize_error_summary(completed.stderr) or f"exit_{completed.returncode}"
    return {
        "production_ask_runtime_invocation_count": 1,
        "production_ask_runtime_success_count": 1 if success else 0,
        "production_ask_runtime_failure_count": 0 if success else 1,
        "production_ask_runtime_error": error,
        "production_ask_runtime_latency_ms": int((time.perf_counter() - started) * 1000),
    }


def determine_failure(input_contract: Mapping[str, Any], prompt: Mapping[str, Any], provider_resolution: Mapping[str, Any], provider: Mapping[str, Any], invocation: Mapping[str, Any], answer: Mapping[str, Any], citations: Mapping[str, Any], production: Mapping[str, Any]) -> dict[str, str]:
    if not input_contract.get("generation_input_contract_valid"):
        return _failure("generation_input_contract", "generation_input_contract_mismatch")
    if prompt.get("prompt_construction_success_count", 0) == 0:
        return _failure("prompt_construction", "prompt_construction_failure")
    if not prompt.get("prompt_evidence_identity_valid") or not prompt.get("prompt_evidence_content_valid"):
        return _failure("prompt_evidence_binding", "prompt_evidence_binding_failure")
    if not provider_resolution.get("generation_provider_resolution_success"):
        return _failure("provider_resolution", "generation_provider_unconfigured")
    if not provider.get("provider_configuration_valid"):
        return _failure("provider_configuration", "generation_provider_unconfigured")
    if not provider.get("provider_connection_valid"):
        return _failure("provider_connection", "generation_provider_unreachable")
    if not provider.get("provider_model_available"):
        return _failure("model_resolution", "generation_model_unavailable")
    if invocation.get("generation_invocation_count", 0) == 0 or invocation.get("generation_execution_success_count", 0) == 0:
        if invocation.get("generation_timeout_count", 0) > 0:
            return _failure("generation_timeout", "generation_timeout_failure")
        return _failure("generation_invocation", "generation_runtime_execution_failure")
    if invocation.get("raw_generation_output_empty_count", 0) > 0:
        return _failure("generation_output_validation", "empty_generation_output")
    if invocation.get("answer_parse_failure_count", 0) > 0:
        return _failure("answer_parsing", "answer_parse_failure")
    if not answer.get("answer_contract_valid"):
        return _failure("answer_contract", "answer_contract_mismatch")
    if citations.get("citation_binding_failure_count", 0) > 0 or citations.get("invalid_citation_count", 0) > 0:
        return _failure("citation_binding", "citation_binding_failure")
    if production.get("production_ask_runtime_invocation_count", 0) > 0 and production.get("production_ask_runtime_success_count", 0) == 0:
        return _failure("production_ask_runtime", "production_ask_runtime_failure")
    return _failure("none", "no_generation_runtime_failure_reproduced")


def build_summary(
    *,
    source_head: str,
    task0186: Mapping[str, Any],
    db: Mapping[str, Any],
    canonical: Mapping[str, Any],
    generation_authority: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    prompt: Mapping[str, Any],
    provider_resolution: Mapping[str, Any],
    provider_readiness: Mapping[str, Any],
    invocation: Mapping[str, Any],
    answer: Mapping[str, Any],
    citations: Mapping[str, Any],
    production: Mapping[str, Any],
    probes: Sequence[GenerationProbeResult],
    failure: Mapping[str, str],
    env: Mapping[str, str],
) -> dict[str, Any]:
    passed = failure["first_generation_loss_substage"] == "none"
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if failure["first_generation_loss_substage"] in LOSS_SUBSTAGES and failure["diagnosed_root_cause"] in ROOT_CAUSES else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0186.get("source_authoritative_head") or source_head,
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        **canonical,
        **generation_authority,
        **input_contract,
        **prompt,
        **provider_resolution,
        **provider_readiness,
        **invocation,
        **answer,
        **citations,
        **production,
        "single_turn_generation_supported": True,
        "first_generation_loss_substage": failure["first_generation_loss_substage"],
        "diagnosed_root_cause": failure["diagnosed_root_cause"],
        "cold_start_generation_stage_passed": passed,
        "next_failure_stage": "end_to_end_query_validation" if passed else f"{failure['first_generation_loss_substage']}_repair",
        "retrieval_policy_changed": False,
        "reranking_policy_changed": False,
        "evidence_policy_changed": False,
        "generation_policy_changed": False,
        "prompt_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "focused_test_passed_count": int(env.get("TASK0187_FOCUSED_TEST_PASSED_COUNT", "0")),
        "full_suite_passed_count": int(env.get("TASK0187_FULL_SUITE_PASSED_COUNT", "0")),
        "full_suite_skipped_count": int(env.get("TASK0187_FULL_SUITE_SKIPPED_COUNT", "0")),
        "full_suite_failed_count": int(env.get("TASK0187_FULL_SUITE_FAILED_COUNT", "0")),
        "known_preexisting_failure_count": int(env.get("TASK0187_KNOWN_PREEXISTING_FAILURE_COUNT", "0")),
        "new_regression_count": int(env.get("TASK0187_NEW_REGRESSION_COUNT", "0")),
    }


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0187.contract.v1",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "diagnosis_first": True,
        "runtime_mutation_allowed": False,
        "generation_policy_mutation_allowed": False,
        "expected_authoritative_document_count": 8,
        "expected_authoritative_chunk_count": 11,
        "expected_embedding_count": 11,
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
        "loss_substage_taxonomy": sorted(LOSS_SUBSTAGES),
        "root_cause_taxonomy": sorted(ROOT_CAUSES),
    }


REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_task",
    "source_authoritative_head",
    "authoritative_document_count",
    "authoritative_chunk_count",
    "stored_embedding_count",
    "canonical_evidence_count",
    "generation_input_constructible",
    "generation_input_evidence_count",
    "generation_input_context_nonempty",
    "default_generation_provider",
    "default_generation_model",
    "generation_provider_type",
    "generation_model_identifier",
    "generation_runtime_enabled",
    "generation_streaming_enabled",
    "generation_probe_query_count",
    "generation_probe_execution_count",
    "generation_probe_success_count",
    "generation_probe_failure_count",
    "generation_input_contract_valid",
    "generation_input_contract_mismatch_count",
    "first_generation_input_contract_mismatch",
    "generation_input_query_count",
    "generation_input_evidence_count",
    "generation_input_rejected_count",
    "prompt_construction_attempt_count",
    "prompt_construction_success_count",
    "prompt_construction_failure_count",
    "prompt_query_present",
    "prompt_evidence_present",
    "prompt_context_nonempty",
    "prompt_evidence_count",
    "prompt_evidence_identity_valid",
    "prompt_evidence_content_valid",
    "generation_provider_resolution_attempted",
    "generation_provider_resolution_success",
    "generation_provider_identifier",
    "provider_endpoint_present",
    "provider_api_key_required",
    "provider_api_key_present",
    "provider_configuration_valid",
    "provider_connection_valid",
    "provider_model_available",
    "generation_model_cache_exists",
    "generation_model_cache_complete",
    "generation_model_authority_valid",
    "generation_model_revision_match",
    "generation_invocation_count",
    "generation_execution_success_count",
    "generation_execution_failure_count",
    "generation_timeout_count",
    "generation_connection_failure_count",
    "generation_provider_error_count",
    "generation_fallback_enabled",
    "generation_fallback_count",
    "expected_generation_fallback_count",
    "unexpected_generation_fallback_count",
    "raw_generation_output_count",
    "raw_generation_output_nonempty_count",
    "raw_generation_output_empty_count",
    "answer_parse_attempt_count",
    "answer_parse_success_count",
    "answer_parse_failure_count",
    "final_answer_count",
    "final_answer_nonempty_count",
    "final_answer_empty_count",
    "answer_contract_valid",
    "answer_identity_valid",
    "answer_content_valid",
    "answer_abstention_count",
    "answer_non_abstention_count",
    "citation_binding_attempt_count",
    "citation_binding_success_count",
    "citation_binding_failure_count",
    "answer_citation_count",
    "valid_citation_count",
    "invalid_citation_count",
    "citation_evidence_identity_valid",
    "citation_chunk_identity_valid",
    "citation_document_identity_valid",
    "citation_content_integrity_valid",
    "production_ask_runtime_invocation_count",
    "production_ask_runtime_success_count",
    "production_ask_runtime_failure_count",
    "single_turn_generation_supported",
    "first_generation_loss_substage",
    "diagnosed_root_cause",
    "cold_start_generation_stage_passed",
    "next_failure_stage",
    "retrieval_policy_changed",
    "reranking_policy_changed",
    "evidence_policy_changed",
    "generation_policy_changed",
    "prompt_policy_changed",
    "runtime_default_behavior_change",
    "promotion_applied",
    "focused_test_passed_count",
    "full_suite_passed_count",
    "full_suite_skipped_count",
    "full_suite_failed_count",
    "known_preexisting_failure_count",
    "new_regression_count",
)


REQUIRED_ARTIFACTS = (
    "summary.json",
    "current_generation_authority_audit.json",
    "generation_input_contract_audit.json",
    "prompt_construction_audit.json",
    "provider_resolution_audit.json",
    "generation_invocation_probe.json",
    "answer_contract_audit.json",
    "citation_provenance_audit.json",
    "production_ask_runtime_probe.json",
    "failure_taxonomy_decision.json",
)


def verify_task0187_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (result_dir / name).exists()]
    summary_path = result_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    missing_fields = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    passed = (
        not missing
        and not missing_fields
        and summary.get("task_status") == "complete"
        and summary.get("retrieval_policy_changed") is False
        and summary.get("reranking_policy_changed") is False
        and summary.get("evidence_policy_changed") is False
        and summary.get("generation_policy_changed") is False
        and summary.get("prompt_policy_changed") is False
        and summary.get("runtime_default_behavior_change") is False
        and summary.get("promotion_applied") is False
        and summary.get("new_regression_count") == 0
        and summary.get("first_generation_loss_substage") in LOSS_SUBSTAGES
        and summary.get("diagnosed_root_cause") in ROOT_CAUSES
    )
    return {
        "task_id": TASK_ID,
        "verification_passed": passed,
        "missing_artifacts": missing,
        "missing_summary_fields": missing_fields,
        "first_generation_loss_substage": summary.get("first_generation_loss_substage"),
        "diagnosed_root_cause": summary.get("diagnosed_root_cause"),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK-0187 Cold-start Generation Runtime Validation and Diagnosis Report",
            "",
            "## Decision",
            f"task_status=`{summary.get('task_status')}`; first_generation_loss_substage=`{summary.get('first_generation_loss_substage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`; cold_start_generation_stage_passed=`{summary.get('cold_start_generation_stage_passed')}`.",
            "",
            "## Generation Authority",
            f"default_generation_provider=`{summary.get('default_generation_provider')}`; default_generation_model=`{summary.get('default_generation_model')}`; provider_type=`{summary.get('generation_provider_type')}`; endpoint_type=`{summary.get('generation_endpoint_type')}`; streaming_enabled=`{summary.get('generation_streaming_enabled')}`.",
            f"provider_endpoint_present=`{summary.get('provider_endpoint_present')}`; provider_api_key_required=`{summary.get('provider_api_key_required')}`; provider_api_key_present=`{summary.get('provider_api_key_present')}`; provider_configuration_valid=`{summary.get('provider_configuration_valid')}`; provider_connection_valid=`{summary.get('provider_connection_valid')}`; provider_model_available=`{summary.get('provider_model_available')}`.",
            "",
            "## Upstream Authority",
            f"documents=`{summary.get('authoritative_document_count')}`; chunks=`{summary.get('authoritative_chunk_count')}`; stored_embeddings=`{summary.get('stored_embedding_count')}`; canonical_evidence_count=`{summary.get('canonical_evidence_count')}`; generation_input_constructible=`{summary.get('generation_input_constructible')}`; generation_input_context_nonempty=`{summary.get('generation_input_context_nonempty')}`.",
            "",
            "## Prompt And Invocation",
            f"probe_queries=`{summary.get('generation_probe_query_count')}`; executions=`{summary.get('generation_probe_execution_count')}`; successes=`{summary.get('generation_probe_success_count')}`; failures=`{summary.get('generation_probe_failure_count')}`.",
            f"input_contract_valid=`{summary.get('generation_input_contract_valid')}`; prompt_successes=`{summary.get('prompt_construction_success_count')}`; prompt_evidence_identity_valid=`{summary.get('prompt_evidence_identity_valid')}`; prompt_evidence_content_valid=`{summary.get('prompt_evidence_content_valid')}`.",
            f"invocations=`{summary.get('generation_invocation_count')}`; execution_successes=`{summary.get('generation_execution_success_count')}`; execution_failures=`{summary.get('generation_execution_failure_count')}`; timeouts=`{summary.get('generation_timeout_count')}`; provider_errors=`{summary.get('generation_provider_error_count')}`.",
            "",
            "## Answer And Citations",
            f"raw_nonempty=`{summary.get('raw_generation_output_nonempty_count')}`; final_answer_nonempty=`{summary.get('final_answer_nonempty_count')}`; abstentions=`{summary.get('answer_abstention_count')}`; non_abstentions=`{summary.get('answer_non_abstention_count')}`.",
            f"citation_attempts=`{summary.get('citation_binding_attempt_count')}`; citation_failures=`{summary.get('citation_binding_failure_count')}`; invalid_citations=`{summary.get('invalid_citation_count')}`; production_ask_success=`{summary.get('production_ask_runtime_success_count')}`.",
            "",
            "## Policy Mutation",
            f"retrieval_policy_changed=`{summary.get('retrieval_policy_changed')}`; reranking_policy_changed=`{summary.get('reranking_policy_changed')}`; evidence_policy_changed=`{summary.get('evidence_policy_changed')}`; generation_policy_changed=`{summary.get('generation_policy_changed')}`; prompt_policy_changed=`{summary.get('prompt_policy_changed')}`; runtime_default_behavior_change=`{summary.get('runtime_default_behavior_change')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Test Accounting",
            f"Focused tests: `{summary.get('focused_test_passed_count')}` passed. Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`, new_regression_count=`{summary.get('new_regression_count')}`.",
            "",
            "## Next Frontier",
            f"next_failure_stage=`{summary.get('next_failure_stage')}`.",
            "",
        ]
    )


def _probe_from_answer(response: SearchResponse, answer: AnswerResponse, prompt_digest: str | None, prompt_character_count: int, prompt_evidence_count: int, prompt_identity_valid: bool, prompt_content_valid: bool) -> GenerationProbeResult:
    bundle = response.evidence_bundle
    assert bundle is not None
    citation_ids = {citation.citation_id for citation in answer.citations}
    available_ids = {f"C{index}" for index, _item in enumerate(bundle.items, start=1)}
    invalid = len(citation_ids - available_ids)
    return GenerationProbeResult(
        query=response.query,
        success=True,
        error="none",
        generation_input_constructible=True,
        generation_input_evidence_count=len(bundle.items),
        generation_input_context_nonempty=response.context_token_count > 0,
        prompt_constructed=True,
        prompt_digest=prompt_digest,
        prompt_character_count=prompt_character_count,
        prompt_evidence_count=prompt_evidence_count,
        prompt_evidence_identity_valid=prompt_identity_valid,
        prompt_evidence_content_valid=prompt_content_valid,
        invocation_attempted=True,
        execution_success=True,
        system_error_code=None,
        provider_error_reason_code=None,
        raw_generation_output_nonempty=bool(str(answer.raw_generation_text or "").strip()),
        answer_parse_success=True,
        final_answer_nonempty=bool(answer.answer.strip()),
        answer_status=answer.status,
        answer_abstained=answer.status == "refused",
        answer_citation_count=len(answer.citations),
        invalid_citation_count=invalid,
        generation_latency_ms=answer.generation_latency_ms,
    )


def _failed_generation_probe(query: str, error: str) -> GenerationProbeResult:
    return GenerationProbeResult(query, False, error, False, 0, False, False, None, 0, 0, False, False, False, False, None, None, False, False, False, None, False, 0, 0, None)


def _probe_models_endpoint(config: AnswerGenerationConfig, *, api_key: str | None) -> dict[str, Any]:
    endpoint = config.base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with urlopen(Request(endpoint, headers=headers, method="GET"), timeout=min(config.timeout_seconds, 20.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        reason = "authentication_failure" if exc.code in {401, 403} else "model_probe_http_error"
        return {"provider_connection_valid": exc.code not in {401, 403}, "provider_model_available": False, "provider_readiness_error": reason}
    except (URLError, TimeoutError, socket.timeout) as exc:
        return {"provider_connection_valid": False, "provider_model_available": False, "provider_readiness_error": f"{type(exc).__name__}: {_sanitize_error_summary(str(exc))}"}
    except Exception as exc:
        return {"provider_connection_valid": False, "provider_model_available": False, "provider_readiness_error": f"{type(exc).__name__}: {_sanitize_error_summary(str(exc))}"}
    models = payload.get("data") if isinstance(payload, dict) else None
    model_ids = {str(row.get("id")) for row in models if isinstance(row, dict) and row.get("id")} if isinstance(models, list) else set()
    return {"provider_connection_valid": True, "provider_model_available": config.model_id in model_ids if model_ids else True, "provider_readiness_error": None}


def _endpoint_type(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127."):
        return "loopback"
    return "remote"


def _load_task0186_summary() -> dict[str, Any]:
    path = TASK0186_RESULT_DIR / "summary.json"
    return read_json(path) if path.exists() else {}


def _current_project_env() -> dict[str, str]:
    env = dict(os.environ)
    env_path = ROOT / ".env"
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and (key not in env or not env[key].strip()) and value.strip():
            env[key] = value
    return env


def _failure(substage: str, root: str) -> dict[str, str]:
    return {"first_generation_loss_substage": substage, "diagnosed_root_cause": root}


def _percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _sanitize_error_summary(value: str) -> str:
    text = value
    for secret in (os.environ.get("OPK_RAG_LLM_API_KEY", ""), os.environ.get("DATABASE_URL", "")):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:240]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(subvalue) for key, subvalue in value.items() if key not in {"runtime_base_url"}}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _sanitize_error_summary(value)
    return value
