from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import KnowledgeBaseRepository
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import load_reranker_config
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.runtime_trace import RuntimeTraceContext
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "evaluation-data/showcase/showcase_manifest_v1.json"
TASK0218_SUMMARY_PATH = ROOT / "evaluation-data/results/task0218-showcase-scenario-and-demo-query-freeze/summary.json"
SHOWCASE_ROOT = (ROOT / "source-documents").resolve()
SHOWCASE_SCHEMA_VERSION = "opk-rag.showcase.demo.v1"
MANIFEST_SCHEMA_VERSION = "opk-rag.showcase-manifest.v1"
EXPECTED_STAGE = "project_showcase_delivery"
HISTORICAL_GRAPH_QUERY = "academic-docx-polisher 的技术路线、最小测试和 Skill 化路线之间是什么关系？"


class ShowcaseAuthorityError(RuntimeError):
    pass


class ShowcaseRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShowcaseAuthority:
    manifest: dict[str, Any]
    scenarios: tuple[dict[str, Any], ...]
    query_set_digest: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_query_payload(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"scenario_id": str(item["scenario_id"]), "query": str(item["query"])}
        for item in sorted(scenarios, key=lambda row: str(row["scenario_id"]))
        if item.get("approved") is True
    ]


def compute_query_set_digest(scenarios: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(canonical_query_payload(scenarios), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_showcase_authority(
    manifest_path: Path = MANIFEST_PATH,
    task0218_summary_path: Path = TASK0218_SUMMARY_PATH,
) -> ShowcaseAuthority:
    if not manifest_path.is_file():
        raise ShowcaseAuthorityError(f"Showcase manifest is missing: {manifest_path}")
    if not task0218_summary_path.is_file():
        raise ShowcaseAuthorityError(f"TASK-0218 authority summary is missing: {task0218_summary_path}")
    manifest = _read_json(manifest_path)
    summary = _read_json(task0218_summary_path)
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest_schema_version_mismatch")
    if manifest.get("active_project_stage") != EXPECTED_STAGE:
        errors.append("project_stage_mismatch")
    raw_scenarios = manifest.get("scenarios")
    if not isinstance(raw_scenarios, list):
        errors.append("scenarios_missing")
        raw_scenarios = []
    scenarios = tuple(dict(item) for item in raw_scenarios if isinstance(item, dict) and item.get("approved") is True)
    ids = [str(item.get("scenario_id") or "") for item in scenarios]
    if not scenarios:
        errors.append("approved_scenarios_missing")
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        errors.append("scenario_ids_invalid_or_duplicate")
    if len(scenarios) != int(manifest.get("scenario_count") or 0):
        errors.append("scenario_count_mismatch")
    if any(not str(item.get("query") or "").strip() for item in scenarios):
        errors.append("scenario_query_missing")
    if any(item.get("runtime_mutation_required") is not False for item in scenarios):
        errors.append("runtime_mutation_required")
    computed = compute_query_set_digest(scenarios)
    manifest_digest = str(manifest.get("showcase_demo_query_set_v1_digest") or "")
    summary_digest = str(summary.get("showcase_demo_query_set_v1_digest") or "")
    if not manifest_digest or computed != manifest_digest:
        errors.append("manifest_query_set_digest_mismatch")
    if not summary_digest or manifest_digest != summary_digest:
        errors.append("task0218_query_set_digest_mismatch")
    if summary.get("showcase_scenario_set_frozen") is not True or summary.get("showcase_demo_query_set_frozen") is not True:
        errors.append("task0218_showcase_authority_not_frozen")
    if any(str(item.get("query")) == HISTORICAL_GRAPH_QUERY and item.get("scenario_id") == "S03" for item in scenarios):
        errors.append("historical_graph_query_promoted")
    if errors:
        raise ShowcaseAuthorityError(",".join(errors))
    return ShowcaseAuthority(manifest=manifest, scenarios=scenarios, query_set_digest=manifest_digest)


def list_showcase_scenarios() -> dict[str, Any]:
    authority = load_showcase_authority()
    return {
        "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
        "manifest_version": authority.manifest.get("manifest_version"),
        "query_set_digest": authority.query_set_digest,
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "name": item["name"],
                "command": item["command"],
                "query": item["query"],
                "capabilities": list(item.get("capabilities") or []),
            }
            for item in authority.scenarios
        ],
    }


def preflight(authority: ShowcaseAuthority | None = None) -> dict[str, Any]:
    authority = authority or load_showcase_authority()
    database_url = os.environ.get("DATABASE_URL", "").strip()
    backend = os.environ.get("OPK_RAG_VECTOR_BACKEND", "qdrant").strip() or "qdrant"
    checks: dict[str, Any] = {
        "repository_root_valid": (ROOT / "pyproject.toml").is_file(),
        "showcase_manifest_exists": MANIFEST_PATH.is_file(),
        "showcase_manifest_valid": True,
        "showcase_digest_valid": compute_query_set_digest(authority.scenarios) == authority.query_set_digest,
        "approved_scenario_count_valid": len(authority.scenarios) == 4,
        "production_vector_backend": backend,
        "production_vector_backend_valid": backend == "qdrant",
        "database_url_available": bool(database_url),
        "showcase_knowledge_base_resolvable": False,
        "knowledge_base_id": None,
        "qdrant_reachable": False,
        "qdrant_collection_exists": False,
        "required_runtime_configuration_available": False,
        "production_embedding_model": None,
        "production_reranker_model": None,
        "default_initial_retrieval_policy": authority.manifest.get("default_initial_retrieval_policy"),
        "graph_runtime_hop_depth": authority.manifest.get("graph_runtime_hop_depth"),
    }
    if database_url:
        try:
            with connect_postgres(database_url) as connection:
                kb = KnowledgeBaseRepository(connection).get_by_root_path(str(SHOWCASE_ROOT))
                if kb is not None:
                    checks["showcase_knowledge_base_resolvable"] = True
                    checks["knowledge_base_id"] = str(kb.id)
        except Exception as exc:
            checks["database_error"] = f"{type(exc).__name__}: {exc}"
    try:
        qdrant = QdrantVectorBackend(load_qdrant_config())
        health = dict(qdrant.health_check())
        checks["qdrant_reachable"] = health.get("qdrant_server_reachable") is True
        if checks["qdrant_reachable"]:
            checks["qdrant_collection_exists"] = qdrant.collection_exists()
    except Exception as exc:
        checks["qdrant_error"] = f"{type(exc).__name__}: {exc}"
    try:
        embedding = load_embedding_config()
        reranker = load_reranker_config()
        checks["production_embedding_model"] = embedding.model_name
        checks["production_reranker_model"] = reranker.model_name
        checks["required_runtime_configuration_available"] = True
    except Exception as exc:
        checks["configuration_error"] = f"{type(exc).__name__}: {exc}"
    required = (
        "repository_root_valid",
        "showcase_manifest_valid",
        "showcase_digest_valid",
        "approved_scenario_count_valid",
        "production_vector_backend_valid",
        "database_url_available",
        "showcase_knowledge_base_resolvable",
        "qdrant_reachable",
        "qdrant_collection_exists",
        "required_runtime_configuration_available",
    )
    checks["preflight_valid"] = all(checks.get(key) is True for key in required)
    return checks


class ShowcaseRunner:
    """Presentation-layer orchestrator over the existing production Search/Ask services."""

    def __init__(self, authority: ShowcaseAuthority, preflight_result: Mapping[str, Any]) -> None:
        if preflight_result.get("preflight_valid") is not True:
            raise ShowcaseRuntimeError("showcase_preflight_failed")
        self.authority = authority
        self.knowledge_base_id = str(preflight_result["knowledge_base_id"])
        self.database_url = os.environ["DATABASE_URL"].strip()
        self.search_config = load_vector_search_config()
        self.embedding_config = load_embedding_config()
        self.reranker_config = load_reranker_config()
        self.embedding_provider = QwenLocalEmbeddingProvider(self.embedding_config) if self.search_config.mode in {"vector", "hybrid"} else None
        self.reranker_provider = BgeLocalRerankerProvider(self.reranker_config) if self.search_config.rerank_enabled else None
        self.context_token_counter = QwenContextTokenCounter(self.embedding_config)
        self._answer_provider = None
        self._answer_config = None

    def _get_answer_runtime(self):
        if self._answer_provider is None:
            self._answer_config = load_answer_generation_config()
            self._answer_provider = OpenAICompatibleLocalChatProvider(
                self._answer_config,
                api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None,
            )
        return self._answer_provider, self._answer_config

    def run_scenario(self, scenario: Mapping[str, Any], *, trace: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        command = str(scenario["command"])
        trace_context = RuntimeTraceContext(
            query_text=str(scenario["query"]),
            execution_scope="ask" if command == "ask" else "search",
            query_id=f"scenario:{scenario['scenario_id']}",
            scenario_id=str(scenario["scenario_id"]),
            enabled=trace,
        ) if trace else None
        response = search_knowledge_base(
            self.database_url,
            knowledge_base_id=_uuid(self.knowledge_base_id),
            query=str(scenario["query"]),
            provider=self.embedding_provider,
            embedding_config=self.embedding_config,
            search_config=self.search_config,
            reranker_provider=self.reranker_provider,
            context_token_counter=self.context_token_counter,
            execution_scope="ask" if command == "ask" else "search",
            runtime_trace_context=trace_context,
        )
        search = _normalize_search(response)
        answer_payload: dict[str, Any] = {}
        if command == "ask":
            provider, config = self._get_answer_runtime()
            answer = answer_knowledge_base(response, provider=provider, config=config, runtime_trace_context=trace_context)
            answer_payload = {
                "status": answer.status,
                "answerable": answer.answerable,
                "answer": answer.answer,
                "refusal_reason_code": answer.refusal_reason_code,
                "grounding_valid": answer.grounding.valid,
                "grounding_reason_code": answer.grounding.reason_code,
                "citation_count": len(answer.citations),
            }
            guard = dict(search.get("guard") or {})
            guard.update(
                {
                    "final_decision": "answer" if answer.status == "answered" else "fail_closed",
                    "answer_status": answer.status,
                    "fail_closed": answer.status == "refused",
                    "refusal_reason_code": answer.refusal_reason_code,
                }
            )
            search["guard"] = guard
        envelope = {
            "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
            "manifest_version": self.authority.manifest.get("manifest_version"),
            "query_set_digest": self.authority.query_set_digest,
            "scenario_id": scenario["scenario_id"],
            "scenario_name": scenario["name"],
            "query": scenario["query"],
            "command": command,
            "execution_status": "passed",
            "runtime": {
                "vector_backend": "qdrant",
                "knowledge_base_id": self.knowledge_base_id,
                "retrieval_policy": self.authority.manifest.get("default_initial_retrieval_policy"),
                "embedding_model": self.embedding_config.model_name,
                "reranker": self.reranker_config.model_name,
                "graph_hop_depth": self.authority.manifest.get("graph_runtime_hop_depth"),
            },
            "retrieval": search["retrieval"],
            "guard": search["guard"],
            "graph": search["graph"],
            "evidence": search["evidence"],
            "answer": answer_payload,
            "top_candidates": search["top_candidates"],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        if trace:
            envelope["runtime_trace"] = answer.runtime_trace if command == "ask" and "answer" in locals() else response.runtime_trace
        envelope["showcase_validation"] = validate_scenario(envelope)
        if envelope["showcase_validation"]["valid"] is not True:
            envelope["execution_status"] = "showcase_drift"
        return envelope


def _uuid(value: str):
    from uuid import UUID

    return UUID(value)


def _normalize_search(response) -> dict[str, Any]:
    graph = dict(response.graph_trace or {})
    guard = dict(response.guard_trace or {})
    evidence_items = []
    if response.evidence_bundle is not None:
        for item in response.evidence_bundle.items:
            evidence_items.append(
                {
                    "chunk_id": str(item.chunk_id),
                    "document_id": str(item.document_id),
                    "relative_path": item.relative_path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "retrieval_sources": list(item.retrieval_sources),
                }
            )
    candidates = [
        {
            "rank": item.rank,
            "chunk_id": str(item.chunk_id),
            "document_id": str(item.document_id),
            "relative_path": item.relative_path,
            "retrieval_sources": list(item.retrieval_sources),
            "selected_for_context": item.selected_for_context,
        }
        for item in response.results[:8]
    ]
    return {
        "retrieval": {
            "retrieval_mode": response.retrieval_mode,
            "candidate_count": response.candidate_count,
            "result_count": response.result_count,
            "vector_candidate_count": response.vector_candidate_count,
            "bm25_candidate_count": response.bm25_candidate_count,
            "reranker_active": response.reranker_enabled,
            "reranker_model_id": response.reranker_model_id,
        },
        "guard": guard,
        "graph": graph,
        "evidence": {
            "selected_evidence": evidence_items,
            "selected_evidence_count": len(evidence_items),
            "provenance_available": bool(evidence_items),
        },
        "top_candidates": candidates,
    }


def validate_scenario(envelope: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(envelope.get("scenario_id"))
    retrieval = envelope.get("retrieval") or {}
    guard = envelope.get("guard") or {}
    graph = envelope.get("graph") or {}
    evidence = envelope.get("evidence") or {}
    answer = envelope.get("answer") or {}
    checks: dict[str, bool] = {}
    if sid == "S01":
        checks = {
            "retrieval_result_available": int(retrieval.get("result_count") or 0) > 0,
            "reranker_active": retrieval.get("reranker_active") is True,
            "evidence_provenance_available": evidence.get("provenance_available") is True,
        }
    elif sid == "S02":
        checks = {
            "guard_evaluated": guard.get("guard_evaluated") is True,
            "structure_lane_invoked": guard.get("structure_lane_invoked") is True,
            "bounded_recovery": int(guard.get("recovery_attempt_count") or 0) <= 1,
            "recovery_activated": guard.get("recovery_activated") is True,
        }
    elif sid == "S03":
        checks = {
            "graph_activated": graph.get("graph_activated") is True,
            "graph_hop_depth_one": int(graph.get("hop_depth") or 0) == 1,
            "expanded_candidate_present": int(graph.get("expanded_candidate_count") or 0) >= 1,
            "graph_candidate_provenance_available": bool(graph.get("candidate_provenance")),
            "runtime_gold_metadata_usage_false": graph.get("runtime_gold_metadata_usage") is False,
        }
    elif sid == "S04":
        checks = {
            "fail_closed": guard.get("fail_closed") is True,
            "answer_status_refused": answer.get("status") == "refused",
            "refusal_reason_available": bool(answer.get("refusal_reason_code")),
        }
    else:
        checks = {"known_scenario": False}
    return {"valid": all(checks.values()), "checks": checks}


def select_scenarios(authority: ShowcaseAuthority, scenario_id: str | None = None) -> list[dict[str, Any]]:
    scenarios = list(authority.scenarios)
    if scenario_id is None:
        return scenarios
    normalized = scenario_id.upper()
    selected = [item for item in scenarios if str(item["scenario_id"]).upper() == normalized]
    if not selected:
        raise ShowcaseAuthorityError(f"unknown_scenario:{scenario_id}")
    return selected


def execute_showcase(scenario_id: str | None = None, *, trace: bool = False) -> dict[str, Any]:
    try:
        authority = load_showcase_authority()
        scenarios = select_scenarios(authority, scenario_id)
    except Exception as exc:
        return {
            "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
            "demo_status": "authority_mismatch",
            "execution_status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
        }
    preflight_result = preflight(authority)
    if preflight_result.get("preflight_valid") is not True:
        return {
            "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
            "manifest_version": authority.manifest.get("manifest_version"),
            "query_set_digest": authority.query_set_digest,
            "demo_status": "blocked",
            "execution_status": "blocked",
            "blocker": "showcase_preflight_failed",
            "preflight": preflight_result,
        }
    try:
        runner = ShowcaseRunner(authority, preflight_result)
        outputs = [runner.run_scenario(item, trace=trace) for item in scenarios]
    except Exception as exc:
        return {
            "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
            "manifest_version": authority.manifest.get("manifest_version"),
            "query_set_digest": authority.query_set_digest,
            "demo_status": "blocked",
            "execution_status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
            "preflight": preflight_result,
        }
    passed = all(item.get("execution_status") == "passed" and (item.get("showcase_validation") or {}).get("valid") is True for item in outputs)
    return {
        "showcase_schema_version": SHOWCASE_SCHEMA_VERSION,
        "manifest_version": authority.manifest.get("manifest_version"),
        "query_set_digest": authority.query_set_digest,
        "demo_status": "passed" if passed else "showcase_drift",
        "execution_status": "passed" if passed else "showcase_drift",
        "scenario_order": [item["scenario_id"] for item in outputs],
        "scenario_count": len(outputs),
        "preflight": preflight_result,
        "scenarios": outputs,
        "runtime_gold_metadata_usage": False,
        "query_specific_hardcoding": False,
        "demo_fixture_manipulation": False,
        "demo_only_runtime_policy": False,
        "runtime_traces": [item.get("runtime_trace") for item in outputs if item.get("runtime_trace")] if trace else [],
    }


def render_scenario_text(item: Mapping[str, Any]) -> str:
    runtime = item.get("runtime") or {}
    retrieval = item.get("retrieval") or {}
    guard = item.get("guard") or {}
    graph = item.get("graph") or {}
    evidence = item.get("evidence") or {}
    answer = item.get("answer") or {}
    candidates = item.get("top_candidates") or []
    lines = [
        "────────────────────────────────────────────────────────",
        f"OPK-RAG Showcase — {item.get('scenario_id')}  {item.get('scenario_name')}",
        "────────────────────────────────────────────────────────",
        "",
        "Query",
        str(item.get("query") or ""),
        "",
        "Runtime",
        f"Retrieval Policy : {runtime.get('retrieval_policy')}",
        f"Vector Backend   : {runtime.get('vector_backend')}",
        f"Reranker         : {runtime.get('reranker')}",
        f"Graph Hop Depth  : {runtime.get('graph_hop_depth')}",
        "",
        "Retrieval",
        f"Mode             : {retrieval.get('retrieval_mode')}",
        f"Candidates       : {retrieval.get('candidate_count')}",
        f"Results          : {retrieval.get('result_count')}",
    ]
    if candidates:
        lines.extend(["", "Top Candidates"])
        for candidate in candidates[:5]:
            sources = ",".join(candidate.get("retrieval_sources") or [])
            lines.append(f"{candidate.get('rank')}. {candidate.get('relative_path')} [{sources}]")
    lines.extend(
        [
            "",
            "Guarded Agent",
            f"Evaluated        : {guard.get('guard_evaluated')}",
            f"Decision         : {guard.get('recovery_decision') or guard.get('final_decision')}",
            f"Recovery         : {guard.get('recovery_attempt_count', 0)} / {guard.get('maximum_recovery_attempt_count', 1)}",
            f"Structure Lane   : {guard.get('structure_lane_invoked')}",
        ]
    )
    if graph.get("graph_activation_evaluated") is not None:
        lines.extend(
            [
                "",
                "Graph Retrieval",
                f"Activated        : {graph.get('graph_activated')}",
                f"Reason           : {graph.get('graph_activation_reason')}",
                f"Expanded         : {graph.get('expanded_candidate_count', 0)}",
            ]
        )
        relations = graph.get("relations") or []
        for relation in relations[:3]:
            lines.append(
                f"Relation         : {relation.get('source_document_id')} --{relation.get('relation_type')}--> {relation.get('target_document_id')}"
            )
    lines.extend(["", "Final Evidence"])
    for index, row in enumerate(evidence.get("selected_evidence") or [], start=1):
        lines.append(f"{index}. {row.get('relative_path')}:{row.get('start_line')}-{row.get('end_line')}")
    if answer:
        lines.extend(
            [
                "",
                "Answer Safety",
                f"Status           : {answer.get('status')}",
                f"Refusal Reason   : {answer.get('refusal_reason_code')}",
            ]
        )
    validation = item.get("showcase_validation") or {}
    lines.extend(
        [
            "",
            "Showcase Result",
            f"Status           : {item.get('execution_status')}",
            f"Validated        : {validation.get('valid')}",
            f"Latency          : {item.get('latency_ms')} ms",
            "────────────────────────────────────────────────────────",
        ]
    )
    return "\n".join(lines)


def render_showcase_text(payload: Mapping[str, Any]) -> str:
    if payload.get("execution_status") != "passed" and not payload.get("scenarios"):
        return "\n".join(
            [
                "OPK-RAG Showcase blocked",
                f"status: {payload.get('demo_status')}",
                f"blocker: {payload.get('blocker')}",
            ]
        )
    blocks = [render_scenario_text(item) for item in payload.get("scenarios") or []]
    header = (
        f"OPK-RAG Unified Showcase Demo\n"
        f"Query Set Digest: {payload.get('query_set_digest')}\n"
        f"Scenario Order: {' → '.join(payload.get('scenario_order') or [])}\n"
    )
    return header + "\n\n" + "\n\n".join(blocks)


def render_scenario_list_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "OPK-RAG Showcase Scenarios",
        f"Query Set Digest: {payload.get('query_set_digest')}",
        "",
    ]
    for item in payload.get("scenarios") or []:
        lines.append(f"{item['scenario_id']}  {item['name']}  ({item['command']})")
        lines.append(f"  {item['query']}")
    return "\n".join(lines)
