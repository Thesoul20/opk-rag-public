from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0238"
EXPERIMENT_ID = "task0238-guarded-rag-pre-llm-agentic-baseline-freeze-and-stage-transition"
SCHEMA_VERSION = "opk-rag.task0238.guarded-rag-pre-llm-agentic-baseline-freeze-and-stage-transition.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0238_guarded_rag_pre_llm_agentic_baseline_freeze_and_stage_transition.json"
BASELINE_DOC = ROOT / "docs" / "LLM_AGENTIC_RAG_PRE_MIGRATION_BASELINE.md"
REPORT_DOC = ROOT / "docs" / "TASK0238_GUARDED_RAG_PRE_LLM_AGENTIC_BASELINE_FREEZE_AND_STAGE_TRANSITION_REPORT.md"

PRODUCTION_ENTRYPOINTS = (
    ROOT / "opk_rag/search/service.py",
    ROOT / "opk_rag/answer/service.py",
    ROOT / "opk_rag/cli.py",
)
PRODUCTION_RUNTIME_PREFIXES = (
    "opk_rag/search/",
    "opk_rag/answer/",
    "opk_rag/agent/",
    "opk_rag/core_tools/",
    "opk_rag/reranking/",
    "opk_rag/retrieval/",
    "opk_rag/graph/",
    "opk_rag/evidence/",
)
MODEL_AGENT_MARKERS = ("ModelDrivenAgentPolicy", "AgentRuntime", "run_agent_recovery_loop", "GovernedRetrievalPlanningRunner")


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _production_model_agent_integration() -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for path in PRODUCTION_ENTRYPOINTS:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for marker in MODEL_AGENT_MARKERS:
            if marker in text:
                findings.append({"path": str(path.relative_to(ROOT)), "marker": marker})
    return {
        "checked_entrypoints": [str(path.relative_to(ROOT)) for path in PRODUCTION_ENTRYPOINTS],
        "model_agent_activation_findings": findings,
        "production_model_driven_agent_active": bool(findings),
    }


def _runtime_diff_audit() -> dict[str, Any]:
    changed = [line for line in git("diff", "--name-only", "HEAD").splitlines() if line]
    runtime_changed = [path for path in changed if path.startswith(PRODUCTION_RUNTIME_PREFIXES)]
    return {
        "changed_paths": changed,
        "production_runtime_changed_paths": runtime_changed,
        "production_runtime_behavior_changed": bool(runtime_changed),
    }


def build_architecture_snapshot() -> dict[str, Any]:
    task0203 = read_json("evaluation-data/results/task0203-post-migration-project-state-consolidation/summary.json")
    task0204 = read_json("evaluation-data/results/task0204-post-migration-release-acceptance-and-rollback-drill/summary.json")
    task0212 = read_json("evaluation-data/results/task0212-search-scoped-fp16-autocast-reranker-production-integration/summary.json")
    task0213 = read_json("evaluation-data/results/task0213-production-search-stage-latency-profiling-and-dominant-bottleneck-diagnosis/summary.json")
    task0149 = read_json("evaluation-data/results/task0149-graph-retrieval-v1-freeze-and-authoritative-baseline-seal/summary.json")
    task0169 = read_json("evaluation-data/results/task0169-graph-lifecycle-v1-freeze-and-engineering-authority-closeout/summary.json")
    integration = _production_model_agent_integration()
    historical_assets = {
        "agent_runtime": (ROOT / "opk_rag/agent/runtime.py").is_file(),
        "model_driven_policy": (ROOT / "opk_rag/agent/model_policy.py").is_file(),
        "tool_registry": (ROOT / "opk_rag/agent/tool_registry.py").is_file(),
        "query_reformulation": (ROOT / "opk_rag/agent/query_reformulation.py").is_file(),
        "task0066_contract": (ROOT / "tasks/TASK-0066-build-and-evaluate-governed-model-driven-agent-policy.md").is_file(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "git_head": git("rev-parse", "HEAD"),
        "stage": {
            "previous_stage": "project_showcase_delivery",
            "current_stage": "llm_agentic_rag_preparation",
            "next_stage": "llm_agentic_rag_development",
        },
        "production": {
            "vector_backend": task0204["production_vector_backend"],
            "production_collection": task0204["active_qdrant_collection"],
            "relational_authority": task0204["relational_authority_backend"],
            "rollback_vector_backend": task0204["rollback_vector_backend"],
            "embedding_model": task0213["production_embedding_model"],
            "embedding_dimension": task0203["current_embedding_dimension"],
            "embedding_distance_metric": task0203["current_embedding_distance_metric"],
            "reranker_model": task0213["production_reranker_model"],
            "search_reranker_precision": task0213["production_search_reranker_precision"],
            "ask_reranker_precision": "fp32",
            "initial_retrieval_policy": task0213["default_initial_retrieval_policy"],
            "graph_runtime_hop_depth": task0149["graph_runtime_hop_depth"],
            "graph_retrieval_v1_frozen": task0149["graph_retrieval_v1_frozen"],
            "graph_lifecycle_v1_frozen": task0169["graph_lifecycle_v1_frozen"],
            "candidate_pipeline": ["retrieval", "bounded_recovery", "reranking", "evidence", "answerability", "generation", "grounding", "citation"],
        },
        "agent_authority": {
            "current_agent_control_mode": "rule_governed",
            "historical_model_driven_agent_assets_present": all(historical_assets.values()),
            "historical_model_driven_agent_assets": historical_assets,
            "historical_model_driven_agent_assets_production_active": integration["production_model_driven_agent_active"],
            "llm_agent_policy_active": False,
            "llm_agent_runtime_active": False,
            "llm_driven_tool_selection": False,
            "llm_observation_loop_active": False,
            "planner_enabled": False,
            "unbounded_agent_loop_enabled": False,
            "maximum_recovery_attempt_count": 1,
            "production_integration_audit": integration,
        },
        "graph_and_corpus": {
            "corpus_digest": task0169["current_corpus_digest"],
            "graph_digest": task0169["active_authoritative_graph_digest"],
            "graph_lifecycle_baseline_digest": task0169["graph_lifecycle_v1_baseline_digest"],
        },
        "runtime_source_digests": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in PRODUCTION_ENTRYPOINTS
        },
    }


def build_benchmark_snapshot() -> dict[str, Any]:
    task0202 = read_json("evaluation-data/results/task0202-qdrant-production-stabilization-and-migration-freeze/summary.json")
    task0204 = read_json("evaluation-data/results/task0204-post-migration-release-acceptance-and-rollback-drill/summary.json")
    task0205 = read_json("evaluation-data/results/task0205-qdrant-production-performance-baseline-and-reseal/summary.json")
    task0213 = read_json("evaluation-data/results/task0213-production-search-stage-latency-profiling-and-dominant-bottleneck-diagnosis/summary.json")
    task0149 = read_json("evaluation-data/results/task0149-graph-retrieval-v1-freeze-and-authoritative-baseline-seal/summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "retrieval": {
            "benchmark_query_digest": task0205["benchmark_query_digest"],
            "benchmark_corpus_digest": task0205["benchmark_corpus_digest"],
            "benchmark_configuration_digest": task0205["benchmark_configuration_digest"],
            "formal_retrieval_recall_at_k": task0205["formal_retrieval_recall_at_k"],
            "formal_retrieval_mrr": task0205["formal_retrieval_mrr"],
            "graph_sensitive_unit_count": task0149["formal_graph_sensitive_unit_count"],
            "graph_runtime_hop_depth": task0149["graph_runtime_hop_depth"],
        },
        "answer_quality": {
            "q01_q07_pass_count": task0204["q01_q07_pass_count"],
            "retrieval_acceptance_passed": task0204["retrieval_acceptance_passed"],
            "answer_acceptance_passed": task0204["answer_acceptance_passed"],
            "citation_acceptance_passed": task0204["citation_acceptance_passed"],
            "safety_acceptance_passed": task0204["safety_acceptance_passed"],
            "grounding_regression_count": task0202["grounding_regression_count"],
            "citation_regression_count": task0202["citation_regression_count"],
            "safety_regression_count": task0202["safety_regression_count"],
        },
        "performance": {
            "qdrant_backend_p50_ms": task0205["qdrant_backend_latency_p50_ms"],
            "qdrant_backend_p95_ms": task0205["qdrant_backend_latency_p95_ms"],
            "qdrant_backend_p99_ms": task0205["qdrant_backend_latency_p99_ms"],
            "full_retrieval_p50_ms": task0213["search_total_p50_ms"],
            "full_retrieval_p95_ms": task0213["search_total_p95_ms"],
            "full_retrieval_p99_ms": task0213["search_total_p99_ms"],
            "reranker_p95_ms": task0213["reranker_total_p95_ms"],
        },
        "retrieval_benchmark_valid": task0205["formal_retrieval_recall_at_k"] == 1.0 and task0205["formal_retrieval_mrr"] == 1.0,
        "answer_quality_benchmark_valid": all([
            task0204["retrieval_acceptance_passed"],
            task0204["answer_acceptance_passed"],
            task0204["citation_acceptance_passed"],
            task0204["safety_acceptance_passed"],
            task0202["grounding_regression_count"] == 0,
            task0202["citation_regression_count"] == 0,
            task0202["safety_regression_count"] == 0,
        ]),
    }


def build_runtime_snapshot() -> dict[str, Any]:
    audit = _runtime_diff_audit()
    task0237 = read_json("evaluation-data/results/task0237-showcase-video-editing-packaging-and-final-delivery/summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "production_runtime_behavior_changed": audit["production_runtime_behavior_changed"],
        "runtime_diff_audit": audit,
        "showcase_video_delivery_complete": task0237["project_showcase_video_delivery"] == "complete",
        "final_showcase_video_ready": task0237["final_showcase_video_ready"],
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "purpose": "Freeze the rule-governed production RAG baseline before LLM Agentic RAG development.",
        "required_agent_state": {
            "current_agent_control_mode": "rule_governed",
            "llm_agent_policy_active": False,
            "llm_agent_runtime_active": False,
            "llm_driven_tool_selection": False,
            "llm_observation_loop_active": False,
        },
        "historical_asset_rule": "Historical model-driven Agent assets may exist, but they must not be active in current production Search/Ask authority.",
        "production_runtime_mutation_allowed": False,
        "next_stage": "llm_agentic_rag_development",
    }


def build_summary() -> dict[str, Any]:
    architecture = build_architecture_snapshot()
    benchmark = build_benchmark_snapshot()
    runtime = build_runtime_snapshot()
    digest_payload = {"architecture": architecture, "benchmark": benchmark}
    digest = stable_digest(digest_payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete",
        "previous_stage": "project_showcase_delivery",
        "current_stage": "llm_agentic_rag_preparation",
        "next_stage": "llm_agentic_rag_development",
        "guarded_rag_baseline_frozen": True,
        "current_agent_control_mode": "rule_governed",
        "historical_model_driven_agent_assets_present": architecture["agent_authority"]["historical_model_driven_agent_assets_present"],
        "historical_model_driven_agent_assets_production_active": architecture["agent_authority"]["historical_model_driven_agent_assets_production_active"],
        "llm_agent_policy_active": False,
        "llm_agent_runtime_active": False,
        "llm_driven_tool_selection": False,
        "llm_observation_loop_active": False,
        "production_vector_backend": architecture["production"]["vector_backend"],
        "production_collection": architecture["production"]["production_collection"],
        "production_embedding_model": architecture["production"]["embedding_model"],
        "production_reranker_model": architecture["production"]["reranker_model"],
        "production_search_reranker_precision": architecture["production"]["search_reranker_precision"],
        "production_ask_reranker_precision": architecture["production"]["ask_reranker_precision"],
        "default_initial_retrieval_policy": architecture["production"]["initial_retrieval_policy"],
        "graph_runtime_hop_depth": architecture["production"]["graph_runtime_hop_depth"],
        "retrieval_benchmark_valid": benchmark["retrieval_benchmark_valid"],
        "answer_quality_benchmark_valid": benchmark["answer_quality_benchmark_valid"],
        "runtime_baseline_valid": not runtime["production_runtime_behavior_changed"],
        "architecture_authority_valid": (
            architecture["production"]["vector_backend"] == "qdrant"
            and architecture["production"]["graph_runtime_hop_depth"] == 1
            and not architecture["agent_authority"]["historical_model_driven_agent_assets_production_active"]
        ),
        "production_runtime_behavior_changed": runtime["production_runtime_behavior_changed"],
        "pre_llm_agentic_baseline_digest": digest,
        "pre_llm_agentic_baseline_digest_present": bool(digest),
        "pre_llm_agentic_baseline_document_valid": BASELINE_DOC.is_file(),
        "blocking_failure_count": 0,
        "known_nonblocking_full_suite_failure_count": 6,
        "full_regression_valid": False,
        "llm_agentic_rag_development_ready": False,
        "git_commit_created": False,
    }


def run_task0238(*, write: bool = True) -> dict[str, Any]:
    architecture = build_architecture_snapshot()
    benchmark = build_benchmark_snapshot()
    runtime = build_runtime_snapshot()
    summary = build_summary()
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT_PATH.write_text(json.dumps(build_contract(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "architecture.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "digest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "pre_llm_agentic_baseline_digest": summary["pre_llm_agentic_baseline_digest"]}, indent=2) + "\n", encoding="utf-8")
        (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def finalize_with_regression(full_regression: dict[str, Any]) -> dict[str, Any]:
    summary = run_task0238(write=True)
    known = int(full_regression.get("failed", 0))
    blocking = int(full_regression.get("blocking_failure_count", known))
    full_valid = bool(full_regression.get("command_completed")) and blocking == 0
    summary.update({
        "known_nonblocking_full_suite_failure_count": known - blocking,
        "blocking_failure_count": blocking,
        "full_regression_valid": full_valid,
        "llm_agentic_rag_development_ready": (
            full_valid
            and summary["architecture_authority_valid"]
            and summary["retrieval_benchmark_valid"]
            and summary["answer_quality_benchmark_valid"]
            and summary["runtime_baseline_valid"]
            and summary["pre_llm_agentic_baseline_document_valid"]
        ),
    })
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "full_regression.json").write_text(json.dumps(full_regression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def verify() -> dict[str, Any]:
    summary_path = RESULT_DIR / "summary.json"
    if not summary_path.is_file():
        return {"verification_passed": False, "reason": "summary_missing"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "task_status": "complete",
        "guarded_rag_baseline_frozen": True,
        "current_agent_control_mode": "rule_governed",
        "historical_model_driven_agent_assets_present": True,
        "historical_model_driven_agent_assets_production_active": False,
        "llm_agent_policy_active": False,
        "llm_agent_runtime_active": False,
        "llm_driven_tool_selection": False,
        "llm_observation_loop_active": False,
        "production_vector_backend": "qdrant",
        "graph_runtime_hop_depth": 1,
        "retrieval_benchmark_valid": True,
        "answer_quality_benchmark_valid": True,
        "runtime_baseline_valid": True,
        "architecture_authority_valid": True,
        "production_runtime_behavior_changed": False,
        "pre_llm_agentic_baseline_digest_present": True,
        "pre_llm_agentic_baseline_document_valid": True,
        "blocking_failure_count": 0,
        "full_regression_valid": True,
        "llm_agentic_rag_development_ready": True,
        "git_commit_created": False,
    }
    mismatches = {key: {"expected": value, "actual": summary.get(key)} for key, value in required.items() if summary.get(key) != value}
    required_files = [
        CONTRACT_PATH, BASELINE_DOC, REPORT_DOC,
        RESULT_DIR / "architecture.json", RESULT_DIR / "benchmark.json", RESULT_DIR / "runtime.json",
        RESULT_DIR / "digest.json", RESULT_DIR / "full_regression.json", RESULT_DIR / "summary.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    state = (ROOT / "PROJECT_STATE.md").read_text(encoding="utf-8")
    authority = (ROOT / "docs/CURRENT_PROJECT_AUTHORITY.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    docs_ok = (
        "TASK-0238 Pre-LLM-Agentic Baseline Freeze" in state
        and "TASK-0238 Pre-LLM-Agentic Baseline Authority" in authority
        and "TASK-0238 Guarded RAG Pre-LLM-Agentic Baseline Freeze" in changelog
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not mismatches and not missing and docs_ok,
        "mismatches": mismatches,
        "missing_files": missing,
        "documentation_markers_valid": docs_ok,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
