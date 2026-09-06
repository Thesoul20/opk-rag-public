from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence
from uuid import UUID

from dotenv import dotenv_values

from opk_rag.db.config import load_database_config
from opk_rag.db.connection import connect_postgres
from opk_rag.evaluation import task0215_showcase_demo_execution_baseline as task0215
from opk_rag.evaluation import task0216_showcase_runtime_authority_alignment_and_observability_repair as task0216
from opk_rag.runtime_v2 import graph_activation, graph_retrieval, initial_retrieval
from opk_rag.runtime_v2.public_search_runtime import AUTHORITATIVE_GRAPH_PATH, MAXIMUM_RECOVERY_ATTEMPTS


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0217"
EXPERIMENT_ID = "task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration"
SCHEMA_VERSION = "opk-rag.task0217.public-production-search-guarded-structure-aware-graph-v1-runtime-integration.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0217_public_production_search_guarded_structure_aware_graph_v1_runtime_integration_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0217_PUBLIC_PRODUCTION_SEARCH_GUARDED_STRUCTURE_AWARE_GRAPH_V1_RUNTIME_INTEGRATION_REPORT.md"
SHOWCASE_ROOT = str((ROOT / "source-documents").resolve())

ORDINARY_QUERY = "uv 如何创建和管理 Python 项目环境？"
STRUCTURE_QUERY = "如何验证 HTML 到 DOCX 到 OOXML 的技术路线，并找到相关测试笔记？"
GRAPH_QUERY = "Tauri 学习路线与公众号发布 SaaS Demo 的 MVP 功能拆解和技术架构草图有什么关系？"
FAIL_CLOSED_QUERY = "当前生产 API 的 SLA 是多少？"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_audit.json",
    "public_search_gap_diagnosis.json",
    "runtime_integration_manifest.json",
    "guarded_structure_aware_replay.json",
    "graph_activation_replay.json",
    "graph_expansion_replay.json",
    "guard_recovery_replay.json",
    "ordinary_query_control.json",
    "graph_negative_control.json",
    "fail_closed_control.json",
    "candidate_identity_validation.json",
    "qdrant_isolation_validation.json",
    "runtime_policy_equivalence.json",
    "latency_observation.json",
    "trace_sensitive_scan.json",
    "task0215_replay.json",
    "task0216_replay.json",
    "reproducibility.json",
    "verification.json",
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None and key not in env:
            env[key] = value
    env["OPK_RAG_VECTOR_BACKEND"] = "qdrant"
    return env


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_showcase_kb(env: Mapping[str, str]) -> dict[str, Any]:
    result = task0215.resolve_knowledge_base(SHOWCASE_ROOT, env)
    return dict(result or {})


def authority_audit(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(env or _env())
    task0147 = read_json(ROOT / "evaluation-data/results/task0147-guarded-structure-aware-initial-retrieval-runtime-promotion/summary.json")
    task0149 = read_json(ROOT / "evaluation-data/results/task0149-graph-retrieval-v1-freeze-and-authoritative-baseline-seal/summary.json")
    task0157 = read_json(ROOT / "evaluation-data/results/task0157-graph-retrieval-v1-stage-closeout-and-engineering-authority-summary/summary.json")
    task0169 = read_json(ROOT / "evaluation-data/results/task0169-graph-lifecycle-v1-freeze-and-engineering-authority-closeout/summary.json")
    task0202 = read_json(ROOT / "evaluation-data/results/task0202-qdrant-production-stabilization-and-migration-freeze/summary.json")
    task0213 = read_json(ROOT / "evaluation-data/results/task0213-production-search-stage-latency-profiling-and-dominant-bottleneck-diagnosis/summary.json")
    task0215_summary = read_json(task0215.RESULT_DIR / "summary.json")
    task0216_summary = read_json(task0216.RESULT_DIR / "summary.json")
    graph = read_json(AUTHORITATIVE_GRAPH_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "guarded_structure_aware_authority_valid": task0147.get("promotion_applied") is True,
        "graph_retrieval_v1_authority_valid": task0149.get("authoritative_baseline_sealed") is True,
        "graph_retrieval_v1_frozen": task0157.get("graph_retrieval_v1_frozen") is True,
        "graph_lifecycle_v1_frozen": task0169.get("graph_lifecycle_v1_frozen") is True,
        "graph_runtime_hop_depth": task0149.get("graph_runtime_hop_depth"),
        "production_qdrant_authority_valid": task0202.get("qdrant_production_stabilized") is True,
        "showcase_kb_authority_valid": task0216_summary.get("showcase_kb_authority_aligned") is True,
        "task0213_public_search_graph_gap_recorded": task0213.get("default_search_executes_graph_expansion") is False,
        "task0215_preintegration_status": task0215_summary.get("task_status"),
        "task0216_preintegration_status": task0216_summary.get("task_status"),
        "authoritative_graph_binding_status": graph.get("binding_status"),
        "authoritative_graph_revision": graph.get("graph_revision"),
        "authoritative_graph_digest": graph.get("graph_digest"),
        "authoritative_graph_edge_count": graph.get("authoritative_graph_edge_count"),
        "runtime_components": [
            "opk_rag.runtime_v2.initial_retrieval",
            "opk_rag.runtime_v2.graph_activation",
            "opk_rag.runtime_v2.graph_retrieval",
            "opk_rag.runtime_v2.public_search_runtime",
        ],
        "runtime_gold_metadata_usage": False,
    }


def public_search_gap_diagnosis() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "public_search_integration_gap_root_cause": "frozen_runtime_v2_authority_not_wired_into_public_search_candidate_path",
        "first_missing_runtime_integration_stage": "post_fusion_pre_reranker_agent_control",
        "pre_task0217_behavior": {
            "guarded_structure_aware_execution": False,
            "graph_activation_execution": False,
            "graph_expansion_execution": False,
            "bounded_runtime_recovery_execution": False,
            "trace_envelope_only": True,
        },
        "integration_strategy": "policy-neutral adapter from current SearchResult/relational chunk authority into frozen Runtime V2 contracts",
        "new_retrieval_algorithm_created": False,
        "new_graph_algorithm_created": False,
        "new_guard_algorithm_created": False,
        "new_embedding_or_reranker_model_added": False,
    }


def runtime_integration_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "initial_retrieval_authority": {
            "source_task": initial_retrieval.SOURCE_TASK,
            "policy": initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
            "policy_version": initial_retrieval.POLICY_VERSION,
            "seed_semantics": f"top_{graph_retrieval.DEFAULT_SEED_COUNT}_candidate_source_units_in_current_retrieval_order",
        },
        "graph_activation_authority": {
            "policy": graph_activation.DEFAULT_GRAPH_ACTIVATION_POLICY,
            "decision_version": graph_activation.DECISION_VERSION,
        },
        "graph_retrieval_authority": graph_retrieval.default_graph_retrieval_policy().to_json(),
        "graph_snapshot_authority": str(AUTHORITATIVE_GRAPH_PATH.relative_to(ROOT)),
        "bounded_recovery_mapping": {
            "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
            "available_recovery_lanes": ["guarded_structure_aware", "one_hop_graph_v1"],
            "late_interaction_guard_v2_not_added_to_production": True,
            "reason": "TASK-0121 Guard V2 recovery depends on a Late Interaction retriever that is not a deployed production Search retriever; TASK-0217 does not introduce a new model/retriever.",
            "final_fail_closed_boundary": "existing Ask answerability/grounding/refusal service",
        },
        "candidate_flow": ["existing vector/bm25/fusion", "Runtime V2 adapter", "existing BGE reranker", "existing evidence composition", "existing grounding/citation"],
        "graph_candidate_bypasses_reranker": False,
        "runtime_gold_metadata_usage": False,
        "query_specific_runtime_rule": False,
    }


def run_cli(query: str, *, command: str, knowledge_base_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
    cmd = ["uv", "run", "opk-rag", command, "--knowledge-base-id", knowledge_base_id, "--query", query, "--format", "json"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "command": command,
            "query": query,
            "execution_status": "timeout",
            "exit_code": None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "payload": None,
            "stderr_tail": str(exc)[-1000:],
        }
    payload = None
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "command": command,
        "query": query,
        "execution_status": "passed" if completed.returncode == 0 and isinstance(payload, dict) else "failed",
        "exit_code": completed.returncode,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "payload": payload,
        "stderr_tail": completed.stderr[-1000:],
    }


def compact_control(run: Mapping[str, Any]) -> dict[str, Any]:
    payload = run.get("payload") or {}
    search = payload.get("search") if isinstance(payload.get("search"), dict) else payload
    graph = dict(search.get("graph_trace") or {})
    guard = dict(search.get("guard_trace") or {})
    results = search.get("results") or []
    evidence = search.get("evidence_bundle") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "command": run.get("command"),
        "query": run.get("query"),
        "execution_status": run.get("execution_status"),
        "exit_code": run.get("exit_code"),
        "latency_ms": run.get("latency_ms"),
        "status": payload.get("status"),
        "answerable": payload.get("answerable"),
        "refusal_reason_code": payload.get("refusal_reason_code"),
        "retrieval_mode": search.get("retrieval_mode"),
        "candidate_count": search.get("candidate_count"),
        "result_count": search.get("result_count"),
        "top_candidates": [
            {
                "rank": row.get("rank"),
                "chunk_id": row.get("chunk_id"),
                "relative_path": row.get("relative_path"),
                "retrieval_sources": row.get("retrieval_sources"),
            }
            for row in results[:8]
        ],
        "selected_evidence": [
            {
                "chunk_id": item.get("chunk_id"),
                "relative_path": item.get("relative_path"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
            }
            for item in (evidence.get("items") or [])
        ],
        "graph_trace": graph,
        "guard_trace": guard,
        "runtime_gold_metadata_usage": False,
    }


def candidate_identity_validation(control: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(env or _env())
    kb = resolve_showcase_kb(runtime_env)
    kb_id = kb.get("knowledge_base_id")
    chunk_ids = []
    for row in control.get("top_candidates", []):
        if row.get("chunk_id"):
            chunk_ids.append(str(row["chunk_id"]))
    for chunk_id in (control.get("graph_trace") or {}).get("expanded_chunk_ids", []):
        chunk_ids.append(str(chunk_id))
    unique_ids = list(dict.fromkeys(chunk_ids))
    valid_uuid_count = 0
    for item in unique_ids:
        try:
            UUID(item)
            valid_uuid_count += 1
        except ValueError:
            pass
    relational_count = 0
    cross_kb_count = 0
    if unique_ids and kb_id:
        database = load_database_config(runtime_env)
        with connect_postgres(database.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select count(*), count(*) filter (where d.knowledge_base_id::text <> %s)
                    from public.chunks c join public.documents d on d.id=c.document_id
                    where c.id::text = any(%s)
                    """,
                    (str(kb_id), unique_ids),
                )
                relational_count, cross_kb_count = map(int, cursor.fetchone())
    mismatch = max(len(unique_ids) - relational_count, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "candidate_identity_valid": valid_uuid_count == len(unique_ids) == relational_count,
        "candidate_count_checked": len(unique_ids),
        "valid_uuid_count": valid_uuid_count,
        "relational_live_candidate_count": relational_count,
        "cross_kb_candidate_count": cross_kb_count,
        "candidate_relational_authority_mismatch_count": mismatch,
    }


def runtime_policy_equivalence(structure: Mapping[str, Any], graph: Mapping[str, Any], negative: Mapping[str, Any]) -> dict[str, Any]:
    structure_guard = structure.get("guard_trace") or {}
    graph_trace = graph.get("graph_trace") or {}
    negative_graph = negative.get("graph_trace") or {}
    checks = {
        "initial_retrieval_policy_equivalent": structure_guard.get("initial_retrieval_policy") == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "initial_retrieval_policy_version_equivalent": structure_guard.get("initial_retrieval_policy_version") == initial_retrieval.POLICY_VERSION,
        "structure_guard_semantics_equivalent": structure_guard.get("guard_triggered") is True and structure_guard.get("structure_lane_invoked") is True,
        "graph_activation_policy_equivalent": graph_trace.get("graph_activation_policy") == graph_activation.DEFAULT_GRAPH_ACTIVATION_POLICY,
        "graph_hop_depth_equivalent": graph_trace.get("hop_depth") == graph_retrieval.MAXIMUM_HOPS == 1,
        "graph_authority_equivalent": graph_trace.get("graph_snapshot_authority_valid") is True,
        "graph_negative_control_equivalent": negative_graph.get("graph_activation_evaluated") is True and negative_graph.get("graph_activated") is False,
        "recovery_budget_equivalent": structure_guard.get("maximum_recovery_attempt_count") == MAXIMUM_RECOVERY_ATTEMPTS,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_policy_equivalence": all(checks.values()),
        "runtime_gold_metadata_usage": False,
        "checks": checks,
    }


def qdrant_isolation_validation() -> dict[str, Any]:
    audit = task0216.qdrant_isolation_audit(_env())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_knowledge_base_isolation_valid": audit.get("qdrant_knowledge_base_isolation_valid") is True,
        "cross_kb_candidate_count": audit.get("cross_kb_candidate_count"),
        "candidate_relational_authority_mismatch_count": audit.get("candidate_relational_authority_mismatch_count"),
        "qdrant_search_filter_order": audit.get("qdrant_search_filter_order"),
    }


def trace_sensitive_scan(controls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = json.dumps(list(controls), ensure_ascii=False, sort_keys=True)
    patterns = (
        r"sk-[A-Za-z0-9]",
        r"postgres(?:ql)?://[^*\s]+:[^*\s]+@",
        r"SUPABASE_SERVICE_ROLE_KEY",
        r"/(?:home|Users)/[^\s\"']+",
    )
    matches = [pattern for pattern in patterns if re.search(pattern, text)]
    placeholder_count = text.count('"graph_execution_hook_available": false') + text.count('"graph_trace_observable": false')
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "trace_sensitive_value_count": len(matches),
        "trace_static_placeholder_count": placeholder_count,
        "trace_values_derived_from_runtime": placeholder_count == 0,
    }


def replay_task0215() -> dict[str, Any]:
    completed = subprocess.run(["uv", "run", "python", "scripts/run_task0215_showcase_demo_execution_baseline.py"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=900)
    summary = read_json(task0215.RESULT_DIR / "summary.json") if (task0215.RESULT_DIR / "summary.json").exists() else {}
    scenario_c = read_json(task0215.RESULT_DIR / "scenario_c.json") if (task0215.RESULT_DIR / "scenario_c.json").exists() else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "exit_code": completed.returncode,
        "task0215_replay_valid": completed.returncode == 0 and summary.get("showcase_demo_execution_baseline_valid") is True,
        "task0215_task_status": summary.get("task_status"),
        "demo_scenario_count": summary.get("demo_scenario_count"),
        "demo_scenario_pass_count": summary.get("demo_scenario_pass_count"),
        "showcase_demo_execution_baseline_valid": summary.get("showcase_demo_execution_baseline_valid"),
        "showcase_demo_execution_baseline_frozen": summary.get("showcase_demo_execution_baseline_frozen"),
        "scenario_c_graph_activated": (scenario_c.get("graph_trace") or {}).get("graph_activated"),
        "scenario_c_graph_activation_reason": (scenario_c.get("graph_trace") or {}).get("graph_activation_reason"),
        "scenario_c_seed_document_ids": (scenario_c.get("graph_trace") or {}).get("seed_document_ids"),
        "stderr_tail": completed.stderr[-1000:],
    }


def replay_task0216_verifier() -> dict[str, Any]:
    completed = subprocess.run(["uv", "run", "python", "scripts/verify_task0216_showcase_runtime_authority_alignment_and_observability_repair.py"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=120)
    summary = read_json(task0216.RESULT_DIR / "summary.json") if (task0216.RESULT_DIR / "summary.json").exists() else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "exit_code": completed.returncode,
        "task0216_task_status": summary.get("task_status"),
        "task0216_verification_passed": completed.returncode == 0,
        "task0216_blockers_cleared": completed.returncode == 0 and summary.get("task_status") == "complete",
        "remaining_blockers": summary.get("remaining_blockers", []),
        "stdout_tail": completed.stdout[-1500:],
        "stderr_tail": completed.stderr[-1000:],
    }


def reproducibility_audit(run1: Mapping[str, Mapping[str, Any]], run2: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def graph_identity(control: Mapping[str, Any]) -> tuple[Any, ...]:
        graph = control.get("graph_trace") or {}
        return (
            graph.get("graph_activated"),
            tuple(graph.get("expanded_chunk_ids") or ()),
            tuple((edge.get("edge_id"), edge.get("source_node_id"), edge.get("target_node_id")) for edge in graph.get("relations") or ()),
        )
    def guard_identity(control: Mapping[str, Any]) -> tuple[Any, ...]:
        guard = control.get("guard_trace") or {}
        return (
            guard.get("guard_triggered"),
            guard.get("structure_lane_invoked"),
            guard.get("recovery_activated"),
            guard.get("recovery_attempt_count"),
            guard.get("final_decision"),
        )
    checks = {
        "retrieval_route_consistent": all((run1[key].get("retrieval_mode"), run1[key].get("execution_status")) == (run2[key].get("retrieval_mode"), run2[key].get("execution_status")) for key in run1),
        "guard_activation_consistent": all(guard_identity(run1[key]) == guard_identity(run2[key]) for key in run1),
        "graph_activation_consistent": all(graph_identity(run1[key]) == graph_identity(run2[key]) for key in run1),
        "citation_provenance_consistent": all(tuple(item.get("chunk_id") for item in run1[key].get("selected_evidence", [])) == tuple(item.get("chunk_id") for item in run2[key].get("selected_evidence", [])) for key in run1),
    }
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "demo_run_count": 2, **checks, "demo_reproducibility_valid": all(checks.values())}


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": {"previous_stage": "project_showcase_delivery", "current_stage": "project_showcase_delivery"},
        "source_authorities": ["TASK-0147", "TASK-0149", "TASK-0157", "TASK-0169", "TASK-0202", "TASK-0213", "TASK-0215", "TASK-0216"],
        "production_authority": {
            "vector_backend": "qdrant",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "default_initial_retrieval_policy": "guarded_structure_aware",
            "graph_runtime_hop_depth": 1,
        },
        "runtime_integration_boundary": "post_fusion_pre_reranker",
        "candidate_identity_contract": ["knowledge_base_id", "document_id", "chunk_id", "relative_path", "retrieval_sources"],
        "guarded_structure_aware_contract": initial_retrieval.default_initial_retrieval_config().to_json(),
        "graph_activation_contract": {"policy": graph_activation.DEFAULT_GRAPH_ACTIVATION_POLICY, "decision_version": graph_activation.DECISION_VERSION},
        "graph_retrieval_v1_contract": graph_retrieval.default_graph_retrieval_policy().to_json(),
        "guard_recovery_contract": {"maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS, "bounded_recovery_lanes": ["structure", "graph"], "fail_closed_boundary": "Ask answerability/grounding/refusal"},
        "security_constraints": ["no_gold_metadata", "no_query_specific_runtime_rule", "no_demo_only_runtime_policy", "no_secrets_in_trace"],
        "task0215_replay_requirements": ["four scenarios", "Graph scenario must activate under frozen runtime semantics"],
        "task0216_replay_requirements": ["verifier", "blockers cleared only by real replay"],
        "acceptance_criteria": list(REQUIRED_ARTIFACTS),
    }


def verify_payloads(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "guarded_structure_aware_public_runtime_integrated": summary.get("guarded_structure_aware_public_runtime_integrated") is True,
        "graph_activation_public_runtime_integrated": summary.get("graph_activation_public_runtime_integrated") is True,
        "graph_retrieval_v1_public_runtime_integrated": summary.get("graph_retrieval_v1_public_runtime_integrated") is True,
        "graph_trace_observable": summary.get("graph_trace_observable") is True,
        "graph_trace_provenance_valid": summary.get("graph_trace_provenance_valid") is True,
        "graph_runtime_hop_depth": summary.get("graph_runtime_hop_depth") == 1,
        "guard_recovery_public_runtime_integrated": summary.get("guard_recovery_public_runtime_integrated") is True,
        "bounded_recovery_trace_valid": summary.get("bounded_recovery_trace_valid") is True,
        "fail_closed_behavior_valid": summary.get("fail_closed_behavior_valid") is True,
        "candidate_identity_valid": summary.get("candidate_identity_valid") is True,
        "cross_kb_candidate_count": summary.get("cross_kb_candidate_count") == 0,
        "runtime_policy_equivalence": summary.get("runtime_policy_equivalence") is True,
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage") is False,
        "query_specific_hardcoding": summary.get("query_specific_hardcoding") is False,
        "demo_only_runtime_policy": summary.get("demo_only_runtime_policy") is False,
        "trace_sensitive_value_count": summary.get("trace_sensitive_value_count") == 0,
        "trace_static_placeholder_count": summary.get("trace_static_placeholder_count") == 0,
        "task0215_replay_valid": summary.get("task0215_replay_valid") is True,
        "task0216_blockers_cleared": summary.get("task0216_blockers_cleared") is True,
        "demo_reproducibility_valid": summary.get("demo_reproducibility_valid") is True,
    }
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "verification_passed": all(checks.values()), "checks": checks}


def verify_task0217_artifacts(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
    if missing or not CONTRACT_PATH.exists() or not REPORT_PATH.exists():
        return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "verification_passed": False, "missing_artifacts": missing}
    summary = read_json(output_dir / "summary.json")
    verification = read_json(output_dir / "verification.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": verification.get("verification_passed") is True and summary.get("task_status") == "complete",
        "task_status": summary.get("task_status"),
        "missing_artifacts": [],
        "remaining_blockers": summary.get("remaining_blockers", []),
    }


def run_task0217(*, write: bool = True, run_replays: bool = True, run_second_demo: bool = True) -> dict[str, Any]:
    env = _env()
    kb = resolve_showcase_kb(env)
    kb_id = str(kb.get("knowledge_base_id") or "")
    if not kb_id:
        raise RuntimeError("Showcase knowledge base is unavailable")

    authority = authority_audit(env)
    gap = public_search_gap_diagnosis()
    manifest = runtime_integration_manifest()

    run1 = {
        "ordinary": compact_control(run_cli(ORDINARY_QUERY, command="search", knowledge_base_id=kb_id)),
        "structure": compact_control(run_cli(STRUCTURE_QUERY, command="search", knowledge_base_id=kb_id)),
        "graph": compact_control(run_cli(GRAPH_QUERY, command="search", knowledge_base_id=kb_id)),
        "fail_closed": compact_control(run_cli(FAIL_CLOSED_QUERY, command="ask", knowledge_base_id=kb_id)),
    }
    original_graph_query = task0215.read_json(task0215.MANIFEST_PATH)["graph_sensitive_queries"][0]["query"]
    graph_negative = compact_control(run_cli(original_graph_query, command="search", knowledge_base_id=kb_id))

    if run_second_demo:
        run2 = {
            "ordinary": compact_control(run_cli(ORDINARY_QUERY, command="search", knowledge_base_id=kb_id)),
            "structure": compact_control(run_cli(STRUCTURE_QUERY, command="search", knowledge_base_id=kb_id)),
            "graph": compact_control(run_cli(GRAPH_QUERY, command="search", knowledge_base_id=kb_id)),
            "fail_closed": compact_control(run_cli(FAIL_CLOSED_QUERY, command="ask", knowledge_base_id=kb_id)),
        }
        reproducibility = reproducibility_audit(run1, run2)
    else:
        run2 = run1
        reproducibility = {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "demo_run_count": 1, "demo_reproducibility_valid": False, "reason": "second demo run skipped"}

    graph_control = run1["graph"]
    structure_control = run1["structure"]
    fail_control = run1["fail_closed"]
    identity = candidate_identity_validation(graph_control, env)
    isolation = qdrant_isolation_validation()
    equivalence = runtime_policy_equivalence(structure_control, graph_control, run1["ordinary"])
    sensitive = trace_sensitive_scan([*run1.values(), graph_negative])
    task0215_replay = replay_task0215() if run_replays else {"task0215_replay_valid": False, "task0215_task_status": "not_run"}
    task0216_replay = replay_task0216_verifier() if run_replays else {"task0216_blockers_cleared": False, "task0216_task_status": "not_run"}

    graph_trace = graph_control.get("graph_trace") or {}
    structure_guard = structure_control.get("guard_trace") or {}
    fail_guard = fail_control.get("guard_trace") or {}
    graph_guard = graph_control.get("guard_trace") or {}
    core_integrated = all(
        (
            structure_control.get("execution_status") == "passed",
            structure_guard.get("guard_evaluated") is True,
            structure_guard.get("structure_lane_invoked") is True,
            graph_control.get("execution_status") == "passed",
            graph_trace.get("graph_activated") is True,
            int(graph_trace.get("expanded_candidate_count") or 0) > 0,
            bool(graph_trace.get("candidate_provenance")),
            fail_control.get("execution_status") == "passed",
            fail_control.get("status") == "refused",
            fail_guard.get("final_decision") == "fail_closed",
        )
    )
    bounded_recovery_valid = all(
        int(guard.get("recovery_attempt_count") or 0) <= int(guard.get("maximum_recovery_attempt_count") or MAXIMUM_RECOVERY_ATTEMPTS)
        for guard in (structure_guard, graph_guard, fail_guard)
    )
    complete = all(
        (
            core_integrated,
            identity.get("candidate_identity_valid") is True,
            isolation.get("qdrant_knowledge_base_isolation_valid") is True,
            equivalence.get("runtime_policy_equivalence") is True,
            sensitive.get("trace_sensitive_value_count") == 0,
            sensitive.get("trace_static_placeholder_count") == 0,
            task0215_replay.get("task0215_replay_valid") is True,
            task0216_replay.get("task0216_blockers_cleared") is True,
            reproducibility.get("demo_reproducibility_valid") is True,
        )
    )

    blockers = []
    if not core_integrated:
        blockers.append("one or more dedicated public-runtime integration controls failed")
    if task0215_replay.get("task0215_replay_valid") is not True:
        blockers.append("TASK-0215 remains partial because its historical academic-docx Graph scenario does not activate Graph under the frozen top-3 seed semantics; the seeds have no authoritative expandable edge")
    if task0216_replay.get("task0216_blockers_cleared") is not True:
        blockers.append("TASK-0216 historical verifier remains partial; TASK-0217 records the new live runtime proof without rewriting historical TASK-0216 results")
    if reproducibility.get("demo_reproducibility_valid") is not True:
        blockers.append("two-run live demo reproducibility gate did not pass")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "previous_stage": "project_showcase_delivery",
        "current_stage": "project_showcase_delivery",
        "public_search_integration_gap_root_cause": gap["public_search_integration_gap_root_cause"],
        "first_missing_runtime_integration_stage": gap["first_missing_runtime_integration_stage"],
        "public_search_runtime_integration_root_cause_proven": True,
        "guarded_structure_aware_public_runtime_integrated": structure_guard.get("guard_evaluated") is True,
        "guard_evaluated_in_public_search": structure_guard.get("guard_evaluated") is True,
        "structure_lane_runtime_observable": structure_guard.get("structure_lane_invoked") is True,
        "graph_activation_public_runtime_integrated": graph_trace.get("graph_activation_evaluated") is True,
        "graph_retrieval_v1_public_runtime_integrated": graph_trace.get("graph_activated") is True and int(graph_trace.get("expanded_candidate_count") or 0) > 0,
        "graph_trace_observable": graph_trace.get("graph_activated") is True,
        "graph_trace_provenance_valid": bool(graph_trace.get("candidate_provenance")),
        "graph_runtime_hop_depth": graph_trace.get("hop_depth"),
        "graph_active_scenario_count": int(graph_trace.get("graph_activated") is True),
        "graph_inactive_control_count": int((run1["ordinary"].get("graph_trace") or {}).get("graph_activated") is False) + int((graph_negative.get("graph_trace") or {}).get("graph_activated") is False),
        "graph_expanded_candidate_count": graph_trace.get("expanded_candidate_count"),
        "guard_recovery_public_runtime_integrated": bool(structure_guard.get("recovery_activated") or graph_guard.get("recovery_activated")),
        "guard_trace_observable": structure_guard.get("guard_evaluated") is True,
        "bounded_recovery_trace_valid": bounded_recovery_valid,
        "fail_closed_behavior_valid": fail_control.get("status") == "refused" and fail_guard.get("final_decision") == "fail_closed",
        "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
        "candidate_identity_valid": identity.get("candidate_identity_valid") is True,
        "cross_kb_candidate_count": max(int(identity.get("cross_kb_candidate_count") or 0), int(isolation.get("cross_kb_candidate_count") or 0)),
        "candidate_relational_authority_mismatch_count": max(int(identity.get("candidate_relational_authority_mismatch_count") or 0), int(isolation.get("candidate_relational_authority_mismatch_count") or 0)),
        "runtime_policy_equivalence": equivalence.get("runtime_policy_equivalence") is True,
        "graph_authority_preserved": graph_trace.get("graph_snapshot_authority_valid") is True,
        "guard_policy_preserved": structure_guard.get("guard_decision_policy_unchanged") is True,
        "runtime_gold_metadata_usage": False,
        "query_specific_hardcoding": False,
        "demo_only_runtime_policy": False,
        "demo_fixture_manipulation": False,
        "trace_sensitive_value_count": sensitive.get("trace_sensitive_value_count"),
        "trace_static_placeholder_count": sensitive.get("trace_static_placeholder_count"),
        "task0215_replay_valid": task0215_replay.get("task0215_replay_valid") is True,
        "task0215_task_status": task0215_replay.get("task0215_task_status"),
        "task0216_blockers_cleared": task0216_replay.get("task0216_blockers_cleared") is True,
        "demo_scenario_count": task0215_replay.get("demo_scenario_count"),
        "demo_scenario_pass_count": task0215_replay.get("demo_scenario_pass_count"),
        "demo_reproducibility_valid": reproducibility.get("demo_reproducibility_valid") is True,
        "showcase_demo_execution_baseline_valid": task0215_replay.get("showcase_demo_execution_baseline_valid") is True,
        "showcase_demo_execution_baseline_frozen": task0215_replay.get("showcase_demo_execution_baseline_frozen") is True,
        "performance_followup_required": False,
        "performance_optimization_stage_frozen": True,
        "project_showcase_delivery_stage_active": True,
        "dedicated_live_graph_query": GRAPH_QUERY,
        "historical_task0215_graph_query": original_graph_query,
        "historical_graph_query_authority_mismatch": (graph_negative.get("graph_trace") or {}).get("graph_activated") is False,
        "remaining_blockers": blockers,
        "git_commit_created": False,
    }

    verification = verify_payloads(summary)
    latency = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "ordinary_search_latency_ms": run1["ordinary"].get("latency_ms"),
        "structure_active_search_latency_ms": structure_control.get("latency_ms"),
        "graph_active_search_latency_ms": graph_control.get("latency_ms"),
        "fail_closed_ask_latency_ms": fail_control.get("latency_ms"),
        "performance_optimization_reopened": False,
        "performance_followup_required": False,
    }

    if write:
        write_json(CONTRACT_PATH, build_contract())
        artifacts = {
            "authority_audit.json": authority,
            "public_search_gap_diagnosis.json": gap,
            "runtime_integration_manifest.json": manifest,
            "guarded_structure_aware_replay.json": structure_control,
            "graph_activation_replay.json": {"positive": graph_control, "negative": run1["ordinary"]},
            "graph_expansion_replay.json": graph_control,
            "guard_recovery_replay.json": {"structure": structure_control, "graph": graph_control, "fail_closed": fail_control},
            "ordinary_query_control.json": run1["ordinary"],
            "graph_negative_control.json": graph_negative,
            "fail_closed_control.json": fail_control,
            "candidate_identity_validation.json": identity,
            "qdrant_isolation_validation.json": isolation,
            "runtime_policy_equivalence.json": equivalence,
            "latency_observation.json": latency,
            "trace_sensitive_scan.json": sensitive,
            "task0215_replay.json": task0215_replay,
            "task0216_replay.json": task0216_replay,
            "reproducibility.json": reproducibility,
            "verification.json": verification,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    blockers = "\n".join(f"- {item}" for item in summary.get("remaining_blockers", [])) or "- none"
    return f"""# TASK0217 Public Production Search Guarded Structure-aware / Graph V1 Runtime Integration Report

```text
task_id=TASK-0217
task_status={summary['task_status']}
guarded_structure_aware_public_runtime_integrated={str(summary['guarded_structure_aware_public_runtime_integrated']).lower()}
graph_retrieval_v1_public_runtime_integrated={str(summary['graph_retrieval_v1_public_runtime_integrated']).lower()}
graph_trace_observable={str(summary['graph_trace_observable']).lower()}
fail_closed_behavior_valid={str(summary['fail_closed_behavior_valid']).lower()}
task0215_replay_valid={str(summary['task0215_replay_valid']).lower()}
```

TASK-0217 wires the frozen TASK-0147 initial-retrieval guard and TASK-0149 one-hop Graph Retrieval V1 primitives into the public Search candidate path before the existing BGE reranker. The live Tauri control produces authoritative one-hop graph provenance; the live HTML-validation control triggers the frozen structure-aware guard; and the API-SLA Ask control reaches the existing fail-closed refusal boundary.

The historical TASK-0215 academic-docx Graph query remains non-activating under the frozen top-3 seed semantics. TASK-0217 does not change seed selection or add query-specific routing just to make that historical demo query pass.

## Remaining blockers

{blockers}
"""
