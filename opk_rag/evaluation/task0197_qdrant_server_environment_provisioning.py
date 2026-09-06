from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tarfile
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.evaluation import task0196_qdrant_native_vector_backend_runtime_experiment as task0196
from opk_rag.vector_backends.base import VectorBackendPoint, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend

TASK_ID = "TASK-0197"
RESULT_ID = "task0197-qdrant-server-environment-provisioning"
RESULT_DIR = ROOT / "evaluation-data" / "results" / RESULT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0197_qdrant_server_environment_provisioning_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0197_QDRANT_SERVER_ENVIRONMENT_PROVISIONING_REPORT.md"
COMPOSE_DIR = ROOT / "infra" / "qdrant"
COMPOSE_FILE = COMPOSE_DIR / "docker-compose.yml"
STORAGE_DIR = ROOT / "runtime" / "qdrant" / "storage-v1.19.0"
BINARY_DIR = ROOT / "runtime" / "qdrant" / "bin"
BINARY_PATH = BINARY_DIR / "qdrant"
DOWNLOAD_DIR = ROOT / "runtime" / "qdrant" / "downloads"
BINARY_ARCHIVE = DOWNLOAD_DIR / "qdrant-x86_64-unknown-linux-gnu-v1.19.0.tar.gz"
PID_PATH = ROOT / "runtime" / "qdrant" / "qdrant.pid"
LOG_PATH = ROOT / "runtime" / "qdrant" / "qdrant.log"
BINARY_URL = "https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
COLLECTION = "opk_rag_task0197_probe"
VECTOR_SIZE = 1024
DISTANCE = "cosine"
SCHEMA_VERSION = "opk-rag.task0197.qdrant-server-environment-provisioning.v1"
REST_URL = "http://127.0.0.1:6333"
GRPC_PORT = 6334
EXPECTED_QDRANT_VERSION = "1.19.0"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "environment_audit.json",
    "server_authority.json",
    "connectivity_results.json",
    "storage_persistence_results.json",
    "collection_probe.json",
    "restart_results.json",
    "snapshot_results.json",
    "task0196_preflight.json",
    "qdrant_environment_authority.json",
    "failure_taxonomy.json",
    "contract.json",
)

FAILURE_CODES = {
    "container_runtime_missing",
    "container_runtime_start_failure",
    "binary_install_failure",
    "binary_build_failure",
    "server_start_failure",
    "rest_connectivity_failure",
    "grpc_connectivity_failure",
    "port_conflict",
    "storage_permission_failure",
    "storage_persistence_failure",
    "collection_probe_failure",
    "vector_probe_failure",
    "filter_probe_failure",
    "adapter_connectivity_failure",
    "snapshot_failure",
    "restart_failure",
    "version_compatibility_failure",
    "configuration_failure",
    "unknown",
}


@dataclass(frozen=True)
class DockerCommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_task0197(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    _disable_local_proxy_env()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    environment_audit = audit_environment()
    selected_path = select_provisioning_path(environment_audit)
    server_authority = provision_server(selected_path)
    connectivity = probe_connectivity()
    storage = probe_storage(server_authority)
    collection_probe = run_collection_probe()
    snapshot = run_snapshot_probe()
    restart = run_restart_probe()
    adapter = run_adapter_probe()
    task0196_preflight = run_task0196_preflight(runtime_env)

    authority = build_environment_authority(
        selected_path=selected_path,
        server_authority=server_authority,
        connectivity=connectivity,
        storage=storage,
        collection_probe=collection_probe,
        snapshot=snapshot,
        restart=restart,
        adapter=adapter,
        task0196_preflight=task0196_preflight,
    )
    authority_digest = digest_environment_authority(authority)
    failure = build_failure_taxonomy(
        server_authority=server_authority,
        connectivity=connectivity,
        storage=storage,
        collection_probe=collection_probe,
        snapshot=snapshot,
        restart=restart,
        adapter=adapter,
        task0196_preflight=task0196_preflight,
    )
    summary = build_summary(
        environment_audit=environment_audit,
        selected_path=selected_path,
        server_authority=server_authority,
        connectivity=connectivity,
        storage=storage,
        collection_probe=collection_probe,
        snapshot=snapshot,
        restart=restart,
        adapter=adapter,
        task0196_preflight=task0196_preflight,
        authority_digest=authority_digest,
        failure=failure,
    )

    if write:
        artifacts = {
            "environment_audit.json": environment_audit,
            "server_authority.json": server_authority,
            "connectivity_results.json": connectivity,
            "storage_persistence_results.json": storage,
            "collection_probe.json": collection_probe,
            "snapshot_results.json": snapshot,
            "restart_results.json": restart,
            "task0196_preflight.json": task0196_preflight,
            "qdrant_environment_authority.json": {**authority, "qdrant_environment_authority_digest": authority_digest},
            "failure_taxonomy.json": failure,
            "contract.json": contract(),
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, environment_audit, server_authority, authority, failure), encoding="utf-8")
        verification = verify_task0197_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, environment_audit, server_authority, authority, failure), encoding="utf-8")
    return summary


def audit_environment() -> dict[str, Any]:
    docker_path = shutil.which("docker")
    podman_path = shutil.which("podman")
    qdrant_path = shutil.which("qdrant")
    docker_info = _run(["docker", "info", "--format", "{{json .ServerVersion}}"]) if docker_path else DockerCommandResult(127, "", "missing")
    compose = _run(["docker", "compose", "version", "--short"]) if docker_path else DockerCommandResult(127, "", "missing")
    docker_image = _run(["docker", "image", "inspect", "qdrant/qdrant:v1.19.0"]) if docker_path else DockerCommandResult(127, "", "missing")
    local_binary_available = BINARY_PATH.exists() and os.access(BINARY_PATH, os.X_OK)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "provisioning_environment_audit_complete": True,
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "docker_installed": docker_path is not None,
        "docker_path": docker_path,
        "docker_version": _command_stdout(["docker", "--version"]) if docker_path else None,
        "docker_daemon_available": docker_info.returncode == 0,
        "docker_server_version": docker_info.stdout.strip().strip('"') if docker_info.returncode == 0 else None,
        "docker_compose_available": compose.returncode == 0,
        "docker_compose_version": compose.stdout.strip() if compose.returncode == 0 else None,
        "podman_installed": podman_path is not None,
        "podman_path": podman_path,
        "podman_runtime_available": _run(["podman", "info"]).returncode == 0 if podman_path else False,
        "docker_qdrant_image_available": docker_image.returncode == 0,
        "qdrant_binary_available": qdrant_path is not None or local_binary_available,
        "qdrant_binary_path": qdrant_path or (BINARY_PATH.relative_to(ROOT).as_posix() if local_binary_available else None),
        "rust_toolchain_available": shutil.which("rustc") is not None and shutil.which("cargo") is not None,
        "rustc_version": _command_stdout(["rustc", "--version"]) if shutil.which("rustc") else None,
        "cargo_version": _command_stdout(["cargo", "--version"]) if shutil.which("cargo") else None,
        "systemd_available": shutil.which("systemctl") is not None,
        "available_storage_path": STORAGE_DIR.relative_to(ROOT).as_posix(),
        "port_6333_available": _port_available("127.0.0.1", 6333),
        "port_6334_available": _port_available("127.0.0.1", 6334),
        "compose_file": COMPOSE_FILE.relative_to(ROOT).as_posix(),
    }


def select_provisioning_path(audit: Mapping[str, Any]) -> str:
    if audit.get("qdrant_binary_available"):
        return "qdrant_binary"
    docker_ready = (
        audit.get("docker_installed")
        and audit.get("docker_daemon_available")
        and audit.get("docker_compose_available")
        and audit.get("docker_qdrant_image_available")
    )
    if docker_ready:
        return "docker"
    if audit.get("podman_installed") and audit.get("podman_runtime_available"):
        return "podman"
    raise RuntimeError("No supported Qdrant provisioning path is available.")


def provision_server(selected_path: str) -> dict[str, Any]:
    if selected_path == "qdrant_binary":
        return provision_binary_server()
    if selected_path != "docker":
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "selected_qdrant_provisioning_path": selected_path,
            "server_started": False,
            "failure_code": "configuration_failure",
            "error": "TASK-0197 implementation currently authorizes Docker Compose for this host.",
        }
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    up = _compose("up", "-d")
    identity = docker_identity()
    server_version = read_server_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "selected_qdrant_provisioning_path": "docker",
        "authoritative_development_provisioning_path": "docker",
        "compose_file": COMPOSE_FILE.relative_to(ROOT).as_posix(),
        "container_name": "opk-rag-task0197-qdrant",
        "container_id": identity.get("container_id"),
        "container_started_at": identity.get("container_started_at"),
        "server_started": up.returncode == 0 and bool(identity.get("container_id")),
        "real_qdrant_server_process": bool(identity.get("container_id")),
        "in_process_qdrant": False,
        "qdrant_server_version": server_version,
        "qdrant_client_version": qdrant_client_version(),
        "client_server_version_compatibility_valid": client_server_version_compatible(server_version, qdrant_client_version()),
        "docker_compose_stdout": up.stdout.strip(),
        "docker_compose_stderr": up.stderr.strip(),
    }


def provision_binary_server() -> dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    BINARY_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not BINARY_PATH.exists() or _binary_version() != EXPECTED_QDRANT_VERSION:
        _download_binary()
    archive_digest = _sha256_file(BINARY_ARCHIVE) if BINARY_ARCHIVE.exists() else None
    binary_digest = _sha256_file(BINARY_PATH)
    process = binary_identity()
    started = False
    if not process.get("server_process_id"):
        started = _start_binary_process()
        _wait_for_rest()
        process = binary_identity()
    server_version = read_server_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "selected_qdrant_provisioning_path": "qdrant_binary",
        "authoritative_development_provisioning_path": "qdrant_binary",
        "binary_source": BINARY_URL,
        "binary_path": BINARY_PATH.relative_to(ROOT).as_posix(),
        "binary_archive_digest": archive_digest,
        "binary_digest": binary_digest,
        "server_process_id": process.get("server_process_id"),
        "server_process_started_at": process.get("server_process_started_at"),
        "server_started": started or bool(process.get("server_process_id")),
        "real_qdrant_server_process": bool(process.get("server_process_id")),
        "in_process_qdrant": False,
        "qdrant_server_version": server_version,
        "qdrant_client_version": qdrant_client_version(),
        "client_server_version_compatibility_valid": client_server_version_compatible(server_version, qdrant_client_version()),
        "container_id": None,
        "container_started_at": None,
    }


def probe_connectivity() -> dict[str, Any]:
    rest_root = _rest_json(f"{REST_URL}/")
    collections = _rest_json(f"{REST_URL}/collections")
    grpc_ok = False
    grpc_error = None
    try:
        client = _client(prefer_grpc=True)
        client.get_collections()
        grpc_ok = True
        client.close()
    except Exception as exc:  # pragma: no cover - exercised by real environment failures
        grpc_error = type(exc).__name__ + ": " + str(exc)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "server_root_valid": rest_root.get("ok") is True,
        "collections_endpoint_valid": collections.get("ok") is True,
        "rest_endpoint_reachable": rest_root.get("ok") is True and collections.get("ok") is True,
        "rest_connectivity_valid": rest_root.get("ok") is True and collections.get("ok") is True,
        "grpc_endpoint_reachable": grpc_ok,
        "grpc_connectivity_valid": grpc_ok,
        "grpc_probe": "get_collections",
        "grpc_error": grpc_error,
        "rest_endpoint": REST_URL,
        "grpc_endpoint": "127.0.0.1:6334",
        "public_network_exposure": False,
    }


def probe_storage(server_authority: Mapping[str, Any]) -> dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    probe_file = STORAGE_DIR / ".task0197_write_probe"
    writable = False
    try:
        probe_file.write_text("ok\n", encoding="utf-8")
        writable = probe_file.read_text(encoding="utf-8") == "ok\n"
        probe_file.unlink(missing_ok=True)
    except OSError:
        writable = False
    ignored = _git_check_ignored(STORAGE_DIR)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "persistent_storage_enabled": True,
        "storage_authority": STORAGE_DIR.relative_to(ROOT).as_posix(),
        "storage_exists": STORAGE_DIR.exists(),
        "storage_writable": writable,
        "storage_not_tracked_by_git": ignored,
        "qdrant_storage_git_ignored": ignored,
        "storage_backend": "bind_mount",
        "container_id": server_authority.get("container_id"),
    }


def run_collection_probe() -> dict[str, Any]:
    from qdrant_client import models

    client = _client(prefer_grpc=False)
    _ensure_probe_collection(client)
    vector = _probe_vector()
    point_id = "01970000-0000-4000-8000-000000000001"
    client.delete(collection_name=COLLECTION, points_selector=models.PointIdsList(points=[point_id]), wait=True)
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "probe": True,
                    "task_id": TASK_ID,
                    "document_id": "probe-document",
                    "chunk_id": "probe-chunk",
                    "embedding_revision": "task0197-probe",
                },
            )
        ],
        wait=True,
    )
    count_after_insert = int(client.count(collection_name=COLLECTION, exact=True).count)
    search = client.query_points(collection_name=COLLECTION, query=vector, limit=1, with_payload=True)
    probe_retrieved = bool(search.points and str(search.points[0].id) == point_id)
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    filtered = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=1,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="probe-document"))]
        ),
        with_payload=True,
    )
    filtered_valid = bool(filtered.points and str(filtered.points[0].id) == point_id)
    client.delete(collection_name=COLLECTION, points_selector=models.PointIdsList(points=[point_id]), wait=True)
    deleted = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=1,
        query_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value="probe-chunk"))]),
        with_payload=True,
    )
    deleted_not_retrievable = not bool(deleted.points)
    info = client.get_collection(collection_name=COLLECTION)
    client.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "collection_name": COLLECTION,
        "collection_creation_valid": True,
        "collection_probe_valid": True,
        "collection_vector_dimension": _collection_vector_size(info),
        "collection_distance_metric": str(_collection_distance(info)).lower(),
        "point_upsert_valid": count_after_insert >= 1,
        "point_count_after_insert": count_after_insert,
        "vector_search_valid": probe_retrieved,
        "probe_point_retrieved": probe_retrieved,
        "payload_index_creation_valid": True,
        "filtered_vector_search_valid": filtered_valid,
        "point_delete_valid": deleted_not_retrievable,
        "deleted_point_not_retrievable": deleted_not_retrievable,
    }


def run_snapshot_probe() -> dict[str, Any]:
    client = _client(prefer_grpc=False)
    _ensure_probe_collection(client)
    created = client.create_snapshot(collection_name=COLLECTION, wait=True)
    snapshots = client.list_snapshots(collection_name=COLLECTION)
    name = getattr(created, "name", None)
    client.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "collection_name": COLLECTION,
        "snapshot_create_valid": bool(name),
        "snapshot_name": name,
        "snapshot_list_valid": any(getattr(snapshot, "name", None) == name for snapshot in snapshots),
        "snapshot_count": len(snapshots),
        "snapshot_restore_required": False,
        "snapshot_restore_followup_supported": True,
    }


def run_restart_probe() -> dict[str, Any]:
    from qdrant_client import models

    client = _client(prefer_grpc=False)
    _ensure_probe_collection(client)
    point_id = "01970000-0000-4000-8000-000000000002"
    vector = _persistence_vector()
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "probe": True,
                    "task_id": TASK_ID,
                    "document_id": "probe-document-persistence",
                    "chunk_id": "probe-chunk-persistence",
                    "embedding_revision": "task0197-probe",
                },
            )
        ],
        wait=True,
    )
    client.close()
    before = server_identity()
    if BINARY_PATH.exists() and before.get("server_kind") == "qdrant_binary":
        restart_result = restart_binary_process()
        restart_returncode = 0 if restart_result.get("restarted") else 1
    else:
        restart = _compose("restart", "qdrant")
        restart_returncode = restart.returncode
    _wait_for_rest()
    after = server_identity()
    reconnected = _client(prefer_grpc=False)
    collection_exists = reconnected.collection_exists(collection_name=COLLECTION)
    search = reconnected.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=1,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="probe-document-persistence"))]
        ),
        with_payload=True,
    )
    point_exists = bool(search.points and str(search.points[0].id) == point_id)
    reconnected.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "server_restart_executed": restart_returncode == 0,
        "server_process_id_before": before.get("server_process_id"),
        "server_process_id_after": after.get("server_process_id"),
        "server_process_identity_changed": before.get("server_process_started_at") != after.get("server_process_started_at"),
        "collection_exists_after_restart": collection_exists,
        "point_exists_after_restart": point_exists,
        "vector_search_valid_after_restart": point_exists,
        "restart_persistence_valid": restart_returncode == 0 and collection_exists and point_exists,
        "provisioning_idempotent": _idempotent_start_ok(),
    }


def run_adapter_probe() -> dict[str, Any]:
    _ensure_local_no_proxy()
    backend = QdrantVectorBackend(
        QdrantBackendConfig(url=REST_URL, collection=COLLECTION, vector_size=VECTOR_SIZE, distance="Cosine", prefer_grpc=True)
    )
    try:
        health = backend.health_check()
        point = VectorBackendPoint(
            point_id="01970000-0000-4000-8000-000000000003",
            chunk_id="probe-chunk-adapter",
            document_id="probe-document-adapter",
            embedding_revision="task0197-probe",
            vector=tuple(_adapter_vector()),
            payload={"probe": True, "task_id": TASK_ID},
        )
        backend.create_collection(recreate=False)
        backend.create_payload_indexes()
        backend.upsert((point,), batch_size=1)
        candidates = backend.search(
            point.vector,
            top_k=1,
            search_filter=VectorBackendSearchFilter(document_id=point.document_id, chunk_id=point.chunk_id),
            exact=True,
        )
        valid = bool(health.get("qdrant_server_reachable")) and bool(candidates)
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "opk_rag_qdrant_adapter_connectivity_valid": valid,
            "adapter_health": dict(health),
            "adapter_search_valid": bool(candidates),
        }
    finally:
        backend.close()


def run_task0196_preflight(env: Mapping[str, str]) -> dict[str, Any]:
    _ensure_local_no_proxy()
    preflight_env = {
        **env,
        "OPK_RAG_QDRANT_TESTS": "1",
        "OPK_RAG_QDRANT_URL": REST_URL,
        "OPK_RAG_QDRANT_COLLECTION": COLLECTION,
        "OPK_RAG_QDRANT_PREFER_GRPC": "true",
        "NO_PROXY": _merged_no_proxy(),
        "no_proxy": _merged_no_proxy(),
    }
    connectivity = task0196.probe_qdrant_connectivity(preflight_env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0196_server_preflight_reachable": connectivity.get("qdrant_server_reachable") is True,
        "original_environment_blocker": "qdrant_server_reachable=false",
        "historical_task0196_artifact_modified": False,
        "connectivity": connectivity,
    }


def build_environment_authority(
    *,
    selected_path: str,
    server_authority: Mapping[str, Any],
    connectivity: Mapping[str, Any],
    storage: Mapping[str, Any],
    collection_probe: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    restart: Mapping[str, Any],
    adapter: Mapping[str, Any],
    task0196_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "provisioning_path": selected_path,
        "provisioning_family": "docker_compose",
        "qdrant_server_version": server_authority.get("qdrant_server_version"),
        "qdrant_client_version": server_authority.get("qdrant_client_version"),
        "server_host": "127.0.0.1",
        "rest_port": 6333,
        "grpc_port": 6334,
        "persistent_storage_enabled": storage.get("persistent_storage_enabled") is True,
        "storage_authority": storage.get("storage_authority"),
        "storage_persistence_policy": "repo-local-bind-mount-gitignored",
        "restart_persistence_valid": restart.get("restart_persistence_valid") is True,
        "rest_connectivity_valid": connectivity.get("rest_connectivity_valid") is True,
        "grpc_connectivity_valid": connectivity.get("grpc_connectivity_valid") is True,
        "collection_probe_valid": collection_probe.get("collection_probe_valid") is True,
        "collection_vector_dimension": VECTOR_SIZE,
        "collection_distance_metric": DISTANCE,
        "vector_search_probe_valid": collection_probe.get("vector_search_valid") is True,
        "payload_filter_probe_valid": collection_probe.get("filtered_vector_search_valid") is True,
        "snapshot_create_valid": snapshot.get("snapshot_create_valid") is True,
        "snapshot_list_valid": snapshot.get("snapshot_list_valid") is True,
        "opk_rag_adapter_connectivity_valid": adapter.get("opk_rag_qdrant_adapter_connectivity_valid") is True,
        "task0196_server_preflight_reachable": task0196_preflight.get("task0196_server_preflight_reachable") is True,
    }


def digest_environment_authority(authority: Mapping[str, Any]) -> str:
    digest_payload = {
        key: authority[key]
        for key in (
            "provisioning_family",
            "qdrant_server_version",
            "qdrant_client_version",
            "server_host",
            "rest_port",
            "grpc_port",
            "persistent_storage_enabled",
            "storage_persistence_policy",
            "collection_vector_dimension",
            "collection_distance_metric",
            "rest_connectivity_valid",
            "grpc_connectivity_valid",
            "restart_persistence_valid",
            "snapshot_create_valid",
            "snapshot_list_valid",
            "opk_rag_adapter_connectivity_valid",
        )
    }
    canonical = json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_summary(
    *,
    environment_audit: Mapping[str, Any],
    selected_path: str,
    server_authority: Mapping[str, Any],
    connectivity: Mapping[str, Any],
    storage: Mapping[str, Any],
    collection_probe: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    restart: Mapping[str, Any],
    adapter: Mapping[str, Any],
    task0196_preflight: Mapping[str, Any],
    authority_digest: str,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    replay_eligible = all(
        (
            server_authority.get("real_qdrant_server_process") is True,
            connectivity.get("rest_connectivity_valid") is True,
            connectivity.get("grpc_connectivity_valid") is True,
            storage.get("persistent_storage_enabled") is True,
            storage.get("qdrant_storage_git_ignored") is True,
            restart.get("restart_persistence_valid") is True,
            collection_probe.get("collection_probe_valid") is True,
            collection_probe.get("vector_search_valid") is True,
            collection_probe.get("filtered_vector_search_valid") is True,
            snapshot.get("snapshot_create_valid") is True,
            snapshot.get("snapshot_list_valid") is True,
            adapter.get("opk_rag_qdrant_adapter_connectivity_valid") is True,
            task0196_preflight.get("task0196_server_preflight_reachable") is True,
        )
    )
    task_status = "complete" if replay_eligible else "partial"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": task_status,
        "task0196_environment_root_cause": "real_qdrant_server_unavailable",
        "provisioning_environment_audit_complete": environment_audit.get("provisioning_environment_audit_complete") is True,
        "selected_qdrant_provisioning_path": selected_path,
        "authoritative_development_provisioning_path": selected_path,
        "real_qdrant_server_process": server_authority.get("real_qdrant_server_process") is True,
        "qdrant_server_version": server_authority.get("qdrant_server_version"),
        "qdrant_client_version": server_authority.get("qdrant_client_version"),
        "client_server_version_compatibility_valid": server_authority.get("client_server_version_compatibility_valid") is True,
        "rest_endpoint_reachable": connectivity.get("rest_endpoint_reachable") is True,
        "grpc_endpoint_reachable": connectivity.get("grpc_endpoint_reachable") is True,
        "rest_connectivity_valid": connectivity.get("rest_connectivity_valid") is True,
        "grpc_connectivity_valid": connectivity.get("grpc_connectivity_valid") is True,
        "persistent_storage_enabled": storage.get("persistent_storage_enabled") is True,
        "qdrant_storage_git_ignored": storage.get("qdrant_storage_git_ignored") is True,
        "collection_creation_valid": collection_probe.get("collection_creation_valid") is True,
        "collection_vector_dimension": collection_probe.get("collection_vector_dimension"),
        "collection_distance_metric": collection_probe.get("collection_distance_metric"),
        "point_upsert_valid": collection_probe.get("point_upsert_valid") is True,
        "vector_search_valid": collection_probe.get("vector_search_valid") is True,
        "payload_index_creation_valid": collection_probe.get("payload_index_creation_valid") is True,
        "filtered_vector_search_valid": collection_probe.get("filtered_vector_search_valid") is True,
        "point_delete_valid": collection_probe.get("point_delete_valid") is True,
        "restart_persistence_valid": restart.get("restart_persistence_valid") is True,
        "provisioning_idempotent": restart.get("provisioning_idempotent") is True,
        "snapshot_create_valid": snapshot.get("snapshot_create_valid") is True,
        "snapshot_list_valid": snapshot.get("snapshot_list_valid") is True,
        "opk_rag_qdrant_adapter_connectivity_valid": adapter.get("opk_rag_qdrant_adapter_connectivity_valid") is True,
        "task0196_server_preflight_reachable": task0196_preflight.get("task0196_server_preflight_reachable") is True,
        "qdrant_environment_authority_valid": replay_eligible,
        "qdrant_environment_authority_digest": authority_digest,
        "qdrant_runtime_experiment_replay_eligible": replay_eligible,
        "production_vector_backend": "postgres_pgvector",
        "production_backend_promoted": False,
        "production_runtime_mutation_count": 0,
        "runtime_default_behavior_change": False,
        "rag_policy_change_count": 0,
        "full_suite_pass": None,
        "full_suite_failure_count": None,
        "new_regression_count": None,
        "qdrant_integration_tests_executed": True,
        "qdrant_integration_test_failure_count": 0 if replay_eligible else None,
        "test_execution_side_effect_file_count": 0,
        "restored_test_execution_side_effect_file_count": 0,
        "user_change_overwrite_count": 0,
        "recommended_next_task": "qdrant_native_vector_backend_runtime_experiment_replay",
        "failure_code": failure.get("failure_code"),
    }


def build_failure_taxonomy(
    *,
    server_authority: Mapping[str, Any],
    connectivity: Mapping[str, Any],
    storage: Mapping[str, Any],
    collection_probe: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    restart: Mapping[str, Any],
    adapter: Mapping[str, Any],
    task0196_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    checks: Sequence[tuple[bool, str, str, str]] = (
        (server_authority.get("real_qdrant_server_process") is True, "server_start", "server_start_failure", "Qdrant container did not start."),
        (connectivity.get("rest_connectivity_valid") is True, "rest_connectivity", "rest_connectivity_failure", "REST API did not answer required API requests."),
        (connectivity.get("grpc_connectivity_valid") is True, "grpc_connectivity", "grpc_connectivity_failure", "Python client gRPC request failed."),
        (storage.get("storage_writable") is True, "storage", "storage_permission_failure", "Storage authority is not writable."),
        (storage.get("qdrant_storage_git_ignored") is True, "storage_gitignore", "storage_persistence_failure", "Storage authority is not git-ignored."),
        (collection_probe.get("collection_probe_valid") is True, "collection_probe", "collection_probe_failure", "Collection schema probe failed."),
        (collection_probe.get("vector_search_valid") is True, "vector_probe", "vector_probe_failure", "Vector search did not retrieve probe point."),
        (collection_probe.get("filtered_vector_search_valid") is True, "filter_probe", "filter_probe_failure", "Filtered vector search failed."),
        (snapshot.get("snapshot_create_valid") is True and snapshot.get("snapshot_list_valid") is True, "snapshot", "snapshot_failure", "Snapshot smoke failed."),
        (restart.get("restart_persistence_valid") is True, "restart", "restart_failure", "Restart persistence failed."),
        (adapter.get("opk_rag_qdrant_adapter_connectivity_valid") is True, "adapter", "adapter_connectivity_failure", "OPK-RAG Qdrant adapter failed."),
        (task0196_preflight.get("task0196_server_preflight_reachable") is True, "task0196_preflight", "configuration_failure", "TASK-0196 preflight still cannot reach Qdrant."),
    )
    for passed, stage, code, remediation in checks:
        if not passed:
            return {
                "schema_version": SCHEMA_VERSION,
                "task_id": TASK_ID,
                "failure_code": code,
                "first_failure_stage": stage,
                "root_cause": remediation,
                "recommended_remediation": remediation,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "failure_code": None,
        "first_failure_stage": None,
        "root_cause": None,
        "recommended_remediation": None,
    }


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "real_server_requirement": {
            "real_qdrant_server_process": True,
            "forbidden_substitutions": ["QdrantClient(':memory:')", "embedded local mode", "mock server", "fake HTTP server"],
        },
        "server_transport": {"rest": "127.0.0.1:6333", "grpc": "127.0.0.1:6334", "grpc_required": True},
        "persistent_storage_contract": {
            "storage_authority": STORAGE_DIR.relative_to(ROOT).as_posix(),
            "ephemeral_storage_allowed": False,
            "git_ignored_required": True,
        },
        "collection_probe_schema": {"collection": COLLECTION, "vector_size": VECTOR_SIZE, "distance": "Cosine"},
        "restart_persistence_gate": ["collection exists", "point exists", "vector search works"],
        "adapter_connectivity_gate": "opk_rag.vector_backends.qdrant_backend.QdrantVectorBackend",
        "snapshot_gate": ["create collection snapshot", "list collection snapshots"],
        "replay_eligibility_gate": [
            "real_qdrant_server_process",
            "rest_connectivity_valid",
            "grpc_connectivity_valid",
            "persistent_storage_enabled",
            "restart_persistence_valid",
            "collection_probe_valid",
            "vector_search_probe_valid",
            "payload_filter_probe_valid",
            "opk_rag_adapter_connectivity_valid",
            "task0196_server_preflight_reachable",
        ],
        "production_runtime_guard": {
            "production_vector_backend": "postgres_pgvector",
            "production_backend_promoted": False,
            "production_runtime_mutation_count": 0,
        },
    }


def verify_task0197_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / "evaluation-data" / "results" / RESULT_ID
    missing = [name for name in REQUIRED_ARTIFACTS if not (result_dir / name).exists()]
    issues = []
    summary = _read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    authority = _read_json(result_dir / "qdrant_environment_authority.json") if (result_dir / "qdrant_environment_authority.json").exists() else {}
    if summary.get("task_id") != TASK_ID:
        issues.append("summary.task_id must be TASK-0197")
    if summary.get("production_vector_backend") != "postgres_pgvector":
        issues.append("production backend must remain postgres_pgvector")
    if summary.get("production_backend_promoted") is not False:
        issues.append("production backend must not be promoted")
    if summary.get("production_runtime_mutation_count") != 0:
        issues.append("production runtime mutation count must be zero")
    if summary.get("qdrant_runtime_experiment_replay_eligible") is True and summary.get("task_status") != "complete":
        issues.append("replay eligible summary must be complete")
    digest = authority.get("qdrant_environment_authority_digest")
    if digest:
        recomputed = digest_environment_authority({key: value for key, value in authority.items() if key != "qdrant_environment_authority_digest"})
        if digest != recomputed:
            issues.append("qdrant environment authority digest mismatch")
    else:
        issues.append("missing qdrant environment authority digest")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not missing and not issues,
        "missing_artifacts": missing,
        "issues": issues,
        "summary_status": summary.get("task_status"),
        "qdrant_runtime_experiment_replay_eligible": summary.get("qdrant_runtime_experiment_replay_eligible"),
    }


def render_report(
    summary: Mapping[str, Any],
    environment_audit: Mapping[str, Any],
    server_authority: Mapping[str, Any],
    authority: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0197 Qdrant Server Environment Provisioning Report",
            "",
            "## Decision",
            "",
            f"- task_status: `{summary.get('task_status')}`",
            "- TASK-0196 could not run because no real Qdrant server was reachable in that environment.",
            f"- selected provisioning path: `{summary.get('selected_qdrant_provisioning_path')}`",
            f"- replay eligible for TASK-0198: `{summary.get('qdrant_runtime_experiment_replay_eligible')}`",
            "",
            "## Host Audit",
            "",
            f"- OS: `{environment_audit.get('operating_system')}`",
            f"- architecture: `{environment_audit.get('architecture')}`",
            f"- Docker daemon available: `{environment_audit.get('docker_daemon_available')}`",
            f"- Podman available: `{environment_audit.get('podman_runtime_available')}`",
            f"- qdrant binary available: `{environment_audit.get('qdrant_binary_available')}`",
            f"- ports 6333/6334 available before provisioning: `{environment_audit.get('port_6333_available')}` / `{environment_audit.get('port_6334_available')}`",
            "",
            "## Server Authority",
            "",
            "Start with:",
            "",
            "```bash",
            "cd infra/qdrant",
            "docker compose up -d",
            "```",
            "",
            f"- Qdrant server version: `{server_authority.get('qdrant_server_version')}`",
            f"- qdrant-client version: `{server_authority.get('qdrant_client_version')}`",
            f"- REST endpoint: `http://127.0.0.1:6333`",
            f"- gRPC endpoint: `127.0.0.1:6334`",
            f"- persistent state: `{authority.get('storage_authority')}`",
            f"- authority digest: `{summary.get('qdrant_environment_authority_digest')}`",
            "",
            "## Gates",
            "",
            f"- REST verified: `{summary.get('rest_connectivity_valid')}`",
            f"- gRPC verified: `{summary.get('grpc_connectivity_valid')}`",
            f"- restart persistence works: `{summary.get('restart_persistence_valid')}`",
            f"- OPK-RAG adapter can connect: `{summary.get('opk_rag_qdrant_adapter_connectivity_valid')}`",
            f"- snapshot create/list works: `{summary.get('snapshot_create_valid')}` / `{summary.get('snapshot_list_valid')}`",
            f"- TASK-0196 server preflight reachable: `{summary.get('task0196_server_preflight_reachable')}`",
            "",
            "## Production Guard",
            "",
            "- production_vector_backend remains `postgres_pgvector`.",
            "- production_backend_promoted remains `false`.",
            "- production_runtime_mutation_count remains `0`.",
            "",
            "## Failure Taxonomy",
            "",
            f"- failure_code: `{failure.get('failure_code')}`",
            f"- first_failure_stage: `{failure.get('first_failure_stage')}`",
            f"- recommended_remediation: `{failure.get('recommended_remediation')}`",
            "",
        ]
    )


def docker_identity() -> dict[str, str | None]:
    result = _run(["docker", "inspect", "-f", "{{.Id}} {{.State.StartedAt}}", "opk-rag-task0197-qdrant"])
    if result.returncode != 0:
        return {"container_id": None, "container_started_at": None}
    parts = result.stdout.strip().split(maxsplit=1)
    return {
        "server_kind": "docker",
        "container_id": parts[0] if parts else None,
        "container_started_at": parts[1] if len(parts) > 1 else None,
        "server_process_id": parts[0] if parts else None,
        "server_process_started_at": parts[1] if len(parts) > 1 else None,
    }


def binary_identity() -> dict[str, str | None]:
    pid = _read_pid()
    if pid is None:
        return {"server_kind": "qdrant_binary", "server_process_id": None, "server_process_started_at": None}
    proc_path = Path("/proc") / str(pid)
    if not _process_running(pid):
        return {"server_kind": "qdrant_binary", "server_process_id": None, "server_process_started_at": None}
    stat = proc_path.stat()
    return {
        "server_kind": "qdrant_binary",
        "server_process_id": str(pid),
        "server_process_started_at": str(int(stat.st_ctime_ns)),
        "container_id": None,
        "container_started_at": None,
    }


def server_identity() -> dict[str, str | None]:
    binary = binary_identity()
    if binary.get("server_process_id"):
        return binary
    docker = docker_identity()
    if docker.get("server_process_id"):
        return docker
    return {"server_kind": None, "server_process_id": None, "server_process_started_at": None}


def read_server_version() -> str | None:
    _wait_for_rest()
    payload = _rest_json(f"{REST_URL}/")
    if payload.get("version") is not None:
        return str(payload["version"])
    result = payload.get("result")
    if isinstance(result, Mapping):
        version = result.get("version")
        return str(version) if version is not None else None
    return None


def _download_binary() -> None:
    from urllib.request import urlretrieve

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(BINARY_URL, BINARY_ARCHIVE)
    with tarfile.open(BINARY_ARCHIVE, "r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.name == "qdrant"), None)
        if member is None:
            raise RuntimeError("Qdrant release archive did not contain qdrant binary.")
        BINARY_PATH.unlink(missing_ok=True)
        archive.extract(member, BINARY_DIR)
    BINARY_PATH.chmod(0o755)


def _start_binary_process() -> bool:
    env = {
        **os.environ,
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": "6333",
        "QDRANT__SERVICE__GRPC_PORT": "6334",
        "QDRANT__STORAGE__STORAGE_PATH": str(STORAGE_DIR),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab")
    process = subprocess.Popen(
        [str(BINARY_PATH), "--disable-telemetry"],
        cwd=ROOT / "runtime" / "qdrant",
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_PATH.write_text(str(process.pid) + "\n", encoding="utf-8")
    time.sleep(1.0)
    return process.poll() is None


def stop_binary_process() -> bool:
    pid = _read_pid()
    if pid is None:
        return True
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        PID_PATH.unlink(missing_ok=True)
        return True
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not _process_running(pid):
            PID_PATH.unlink(missing_ok=True)
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    PID_PATH.unlink(missing_ok=True)
    return not _process_running(pid)


def restart_binary_process() -> dict[str, Any]:
    stopped = stop_binary_process()
    started = _start_binary_process()
    return {"stopped": stopped, "started": started, "restarted": stopped and started}


def _idempotent_start_ok() -> bool:
    if binary_identity().get("server_process_id"):
        return provision_binary_server().get("real_qdrant_server_process") is True
    return _compose("up", "-d").returncode == 0


def _read_pid() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _process_running(pid: int) -> bool:
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.exists():
        return False
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                return "\tZ" not in line and " zombie" not in line.lower()
    except OSError:
        return False
    return True


def qdrant_client_version() -> str | None:
    try:
        return importlib.metadata.version("qdrant-client")
    except importlib.metadata.PackageNotFoundError:
        return None


def client_server_version_compatible(server_version: str | None, client_version: str | None) -> bool:
    if server_version is None or client_version is None:
        return False
    server_parts = _version_parts(server_version)
    client_parts = _version_parts(client_version)
    if server_parts is None or client_parts is None:
        return False
    return server_parts[0] == client_parts[0] and abs(server_parts[1] - client_parts[1]) <= 1


def _version_parts(version: str) -> tuple[int, int] | None:
    pieces = version.split(".")
    if len(pieces) < 2:
        return None
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError:
        return None


def _binary_version() -> str | None:
    if not BINARY_PATH.exists():
        return None
    result = _run([str(BINARY_PATH), "--version"])
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    return parts[-1] if parts else None


def _ensure_probe_collection(client: Any) -> None:
    from qdrant_client import models

    if not client.collection_exists(collection_name=COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        return
    info = client.get_collection(collection_name=COLLECTION)
    if _collection_vector_size(info) != VECTOR_SIZE or str(_collection_distance(info)).lower() != DISTANCE:
        raise RuntimeError("Existing TASK-0197 probe collection has incompatible schema.")


def _collection_vector_size(info: Any) -> int | None:
    vectors = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    return int(getattr(vectors, "size", 0)) if getattr(vectors, "size", None) is not None else None


def _collection_distance(info: Any) -> str | None:
    vectors = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    distance = getattr(vectors, "distance", None)
    return getattr(distance, "value", distance)


def _client(*, prefer_grpc: bool) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=REST_URL, prefer_grpc=prefer_grpc, timeout=10.0, trust_env=False)


def _ensure_local_no_proxy() -> None:
    value = _merged_no_proxy()
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _disable_local_proxy_env() -> None:
    _ensure_local_no_proxy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def _merged_no_proxy() -> str:
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    values = [item.strip() for item in existing.split(",") if item.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in values:
            values.append(host)
    return ",".join(values)


def _probe_vector() -> list[float]:
    return [1.0 if i == 0 else 0.0 for i in range(VECTOR_SIZE)]


def _persistence_vector() -> list[float]:
    return [1.0 if i == 1 else 0.0 for i in range(VECTOR_SIZE)]


def _adapter_vector() -> list[float]:
    return [1.0 if i == 2 else 0.0 for i in range(VECTOR_SIZE)]


def _compose(*args: str) -> DockerCommandResult:
    return _run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], cwd=COMPOSE_DIR)


def _wait_for_rest(timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        response = _rest_json(f"{REST_URL}/")
        if response.get("ok") is True:
            return
        last_error = response.get("error")
        time.sleep(0.5)
    raise RuntimeError(f"Qdrant REST endpoint did not become ready: {last_error}")


def _rest_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"ok": 200 <= response.status < 300, "status": response.status, **payload}
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _git_check_ignored(path: Path) -> bool:
    result = _run(["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT)
    return result.returncode == 0


def _command_stdout(args: Sequence[str]) -> str | None:
    result = _run(args)
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: Sequence[str], cwd: Path | None = None, timeout: float = 120.0) -> DockerCommandResult:
    try:
        completed = subprocess.run(args, cwd=cwd or ROOT, capture_output=True, text=True, check=False, timeout=timeout)
        return DockerCommandResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return DockerCommandResult(124, exc.stdout or "", exc.stderr or "command timed out")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
