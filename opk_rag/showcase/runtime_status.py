from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import KnowledgeBaseRepository
from opk_rag.embedding.config import load_embedding_config
from opk_rag.reranking.config import load_reranker_config
from opk_rag.showcase.demo import SHOWCASE_ROOT
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config

ROOT = Path(__file__).resolve().parents[2]
GRAPH_AUTHORITY = ROOT / "evaluation-data/results/task0169-graph-lifecycle-v1-freeze-and-engineering-authority-closeout/summary.json"
SCHEMA_VERSION = "opk-rag.control-center-runtime-status.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def _kb_status() -> dict[str, Any]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return {**_safe_error("database_url_not_configured"), "path_visibility_scope": "local_operator_only"}
    try:
        with connect_postgres(database_url) as conn:
            kb = KnowledgeBaseRepository(conn).get_by_root_path(str(SHOWCASE_ROOT))
            if kb is None:
                return {**_safe_error("knowledge_base_not_registered"), "path_visibility_scope": "local_operator_only"}
            with conn.cursor() as cur:
                cur.execute("select count(*), count(*) filter (where index_status = 'indexed') from public.documents where knowledge_base_id = %s", (kb.id,))
                source_docs, indexed_docs = cur.fetchone()
                cur.execute("select count(*) from public.chunks c join public.documents d on d.id=c.document_id where d.knowledge_base_id=%s", (kb.id,))
                chunks = cur.fetchone()[0]
                cur.execute("select status, started_at, finished_at from public.index_runs where knowledge_base_id=%s order by started_at desc limit 1", (kb.id,))
                last = cur.fetchone()
            return {
                "available": True,
                "reason": None,
                "knowledge_base_id": str(kb.id),
                "knowledge_base_name": kb.name,
                "knowledge_base_path": kb.root_path,
                "path_visibility_scope": "local_operator_only",
                "source_document_count": int(source_docs),
                "indexed_document_count": int(indexed_docs),
                "chunk_count": int(chunks),
                "index_status": last[0] if last else None,
                "last_indexed_at": (last[2] or last[1]).isoformat() if last else None,
            }
    except Exception as exc:
        return {**_safe_error(type(exc).__name__), "path_visibility_scope": "local_operator_only"}


def _qdrant_status() -> dict[str, Any]:
    config = load_qdrant_config()
    backend = QdrantVectorBackend(config)
    base = {"backend": "qdrant", "collection_name": config.collection, "vector_size": config.vector_size, "distance": config.distance}
    try:
        health = backend.health_check()
        if health.get("qdrant_server_reachable") is not True:
            return {**base, "server_reachable": False, "collection_exists": False, "points_count": None, "collection_status": "unavailable", "reason": "qdrant_unreachable"}
        exists = backend.collection_exists()
        if not exists:
            return {**base, "server_reachable": True, "collection_exists": False, "points_count": 0, "collection_status": "missing", "reason": "collection_missing"}
        info = backend.client.get_collection(config.collection)
        return {**base, "server_reachable": True, "collection_exists": True, "points_count": backend.count(), "collection_status": str(getattr(info, "status", "available")), "reason": None}
    except Exception as exc:
        return {**base, "server_reachable": False, "collection_exists": False, "points_count": None, "collection_status": "unavailable", "reason": type(exc).__name__}
    finally:
        backend.close()


def _graph_status() -> dict[str, Any]:
    if not GRAPH_AUTHORITY.is_file():
        return {**_safe_error("graph_authority_missing"), "graph_hop_limit": 1, "live_repair_enabled": False}
    try:
        row = json.loads(GRAPH_AUTHORITY.read_text(encoding="utf-8"))
        graph_digest = row.get("active_authoritative_graph_digest")
        corpus_digest = row.get("current_corpus_digest")
        source_digest = row.get("active_graph_source_corpus_digest")
        return {
            "available": True,
            "reason": None,
            "graph_snapshot_status": "available",
            "graph_freshness": row.get("freshness_state") or ("fresh" if row.get("active_graph_freshness_valid") else "stale"),
            "graph_digest": graph_digest,
            "corpus_digest": corpus_digest,
            "graph_matches_corpus": bool(source_digest and source_digest == corpus_digest),
            "graph_hop_limit": int(row.get("graph_runtime_hop_depth") or 1),
            "live_repair_enabled": False,
            "runtime_freshness_check_enabled": bool(row.get("runtime_freshness_check_enabled")),
            "authority_source": str(GRAPH_AUTHORITY.relative_to(ROOT)),
        }
    except Exception as exc:
        return {**_safe_error(type(exc).__name__), "graph_hop_limit": 1, "live_repair_enabled": False}


def _models_status() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        e = load_embedding_config()
        out["embedding"] = {"configured": True, "model": e.model_name, "dimension": e.dimension, "configured_device": e.device, "resolved_device": None, "runtime_loaded_state": "not_probed", "dtype": None}
    except Exception as exc:
        out["embedding"] = {"configured": False, "reason": type(exc).__name__}
    try:
        r = load_reranker_config()
        out["reranker"] = {"configured": True, "model": r.model_name, "configured_device": r.device, "resolved_device": None, "runtime_loaded_state": "not_probed", "search_precision": r.search_precision, "ask_precision": r.ask_precision}
    except Exception as exc:
        out["reranker"] = {"configured": False, "reason": type(exc).__name__}
    try:
        g = load_answer_generation_config()
        location = "remote" if g.allow_remote else "local_or_operator_configured"
        out["generation"] = {"configured": True, "provider": g.provider_id, "model": g.model_id, "location": location, "provider_reachable": None, "health": "not_probed"}
    except Exception as exc:
        out["generation"] = {"configured": False, "reason": type(exc).__name__}
    return out


def _gpu_status() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, check=True)
        line = next((x.strip() for x in proc.stdout.splitlines() if x.strip()), "")
        if not line:
            return {"gpu_available": False, "reason": "no_gpu_rows"}
        parts = [x.strip() for x in line.split(",")]
        return {"gpu_available": True, "reason": None, "device_index": int(parts[0]), "device_name": parts[1], "total_vram_mb": int(parts[2]), "used_vram_mb": int(parts[3]), "free_vram_mb": int(parts[4]), "utilization_percent": int(parts[5]), "temperature_c": int(parts[6]), "source": "nvidia-smi", "polling_policy": "manual_or_low_frequency_5_10s"}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, StopIteration) as exc:
        return {"gpu_available": False, "reason": type(exc).__name__, "source": "nvidia-smi", "polling_policy": "manual_or_low_frequency_5_10s"}


def _health(payload: dict[str, Any]) -> dict[str, Any]:
    kb = payload["knowledge_base"]
    q = payload["qdrant"]
    models = payload["models"]
    graph = payload["graph"]
    search_ready = bool(kb.get("available") and q.get("server_reachable") and q.get("collection_exists") and models.get("embedding", {}).get("configured") and models.get("reranker", {}).get("configured"))
    ask_ready = bool(search_ready and models.get("generation", {}).get("configured"))
    status = "healthy" if ask_ready and graph.get("available") else "degraded" if search_ready else "unavailable"
    return {"system_status": status, "search_ready": search_ready, "ask_ready": ask_ready, "generation_health": models.get("generation", {}).get("health", "unavailable")}


def collect_runtime_status() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "observed_at": _now(),
        "authority_kind": "runtime_observed_and_configured",
        "runtime_trace_authority_changed": False,
        "ui_decision_authority": False,
        "rag_backend_architecture_changed": False,
        "production_agent_authority_changed": False,
        "graph_max_hop": 1,
        "recovery_max_attempts": 1,
        "knowledge_base": _kb_status(),
        "qdrant": _qdrant_status(),
        "graph": _graph_status(),
        "models": _models_status(),
        "gpu": _gpu_status(),
    }
    payload["health"] = _health(payload)
    return payload
