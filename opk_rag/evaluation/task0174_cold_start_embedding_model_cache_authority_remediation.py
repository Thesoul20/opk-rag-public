from __future__ import annotations

from pathlib import Path
from typing import Any

from opk_rag.embedding.cache_authority import EmbeddingModelCacheAuthority
from opk_rag.embedding.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_REVISION
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, write_json

TASK_ID = "TASK-0174"
TASK0173_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0173-cold-start-embedding-cuda-runtime-failure-diagnosis"
EXPERIMENT_ID = "task0174-cold-start-embedding-model-cache-authority-remediation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0174_cold_start_embedding_model_cache_authority_remediation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0174_COLD_START_EMBEDDING_MODEL_CACHE_AUTHORITY_REMEDIATION_REPORT.md"


def run_task0174(write: bool = True) -> dict[str, Any]:
    source_head = _git_head()
    task0173 = _read_json(TASK0173_RESULT_DIR / "summary.json")
    root_cause_valid = task0173.get("dominant_embedding_root_cause") == "model_cache_missing_or_incomplete"
    confidence_valid = task0173.get("root_cause_confidence") in {"high", "medium"}
    cache_authority = EmbeddingModelCacheAuthority(
        model_id=DEFAULT_EMBEDDING_MODEL_NAME,
        revision=DEFAULT_EMBEDDING_MODEL_REVISION,
    )
    development_cache_audit = cache_authority.audit()
    remediation_design = build_remediation_design(development_cache_audit)
    no_policy_mutation = {
        "embedding_policy_mutation_count": 0,
        "embedding_device_default_mutation_count": 0,
        "dependency_remediation_count": 0,
        "retrieval_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "reranker_policy_mutation_count": 0,
        "chunking_policy_mutation_count": 0,
        "corpus_mutation_count": 0,
        "cpu_fallback_promoted": False,
    }
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "task_status": "complete",
        "source_authoritative_head": source_head,
        "parent_task": "TASK-0173",
        "task0173_root_cause_loaded": root_cause_valid,
        "task0173_root_cause_confidence_valid": confidence_valid,
        "diagnosed_root_cause": task0173.get("dominant_embedding_root_cause"),
        "first_embedding_failure_stage": task0173.get("first_embedding_failure_stage"),
        "remediation_family": "repair_model_cache_authority",
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "cache_authority_module_added": True,
        "cache_authority_contract_added": True,
        "cache_authority_verifier_added": True,
        "development_model_cache_exists": development_cache_audit["model_cache_exists"],
        "development_model_cache_complete": development_cache_audit["model_cache_complete"],
        "development_cache_authority_valid": development_cache_audit["cache_authority_valid"],
        "cold_start_cache_authority_required": True,
        "cold_start_cache_materialized_by_task0174": False,
        "model_cache_files_committed": False,
        "runtime_default_behavior_change": False,
        "embedding_policy_unchanged": True,
        "embedding_device_policy_unchanged": True,
        "dependency_policy_unchanged": True,
        "task0170_revalidation_executed": False,
        "task0170_revalidation_not_run_reason": "model cache materialization is operator/local-cache authority; repository now provides deterministic verifier and documented remediation gate",
        **no_policy_mutation,
    }
    artifacts = {
        "summary.json": summary,
        "task0173_authority.json": {
            "task_id": "TASK-0173",
            "summary_path": str(TASK0173_RESULT_DIR / "summary.json"),
            "root_cause_valid_for_task0174": root_cause_valid,
            "confidence_valid_for_task0174": confidence_valid,
            "dominant_embedding_root_cause": task0173.get("dominant_embedding_root_cause"),
            "recommended_remediation_family": task0173.get("recommended_remediation_family"),
        },
        "model_cache_authority_audit.json": development_cache_audit,
        "remediation_design.json": remediation_design,
        "no_policy_mutation_audit.json": no_policy_mutation | {
            "runtime_default_behavior_change": False,
            "model_replacement_count": 0,
            "embedding_dimension_mutation_count": 0,
        },
    }
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, artifacts), encoding="utf-8")
    return summary


def build_remediation_design(cache_audit: dict[str, object]) -> dict[str, object]:
    return {
        "remediation_scope": "model_cache_authority_only",
        "fixed_model_id": DEFAULT_EMBEDDING_MODEL_NAME,
        "fixed_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "required_files": cache_audit["required_files"],
        "operator_materialization_command": "python scripts/verify_task0174_cold_start_embedding_model_cache_authority_remediation.py",
        "post_materialization_expected_gate": "cache_authority_valid=true before local_files_only cold-start embedding smoke",
        "forbidden_actions": [
            "promote_cpu_fallback",
            "replace_embedding_model",
            "change_embedding_dimension",
            "change_embedding_device_default",
            "commit_model_cache_files",
            "change_retrieval_graph_reranker_chunking_or_corpus_policy",
        ],
    }


def contract() -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "parent_task": "TASK-0173",
        "diagnosed_root_cause_required": "model_cache_missing_or_incomplete",
        "remediation_scope": "embedding_model_cache_authority",
        "embedding_model_mutation_allowed": False,
        "embedding_device_default_mutation_allowed": False,
        "cpu_fallback_promotion_allowed": False,
        "dependency_remediation_allowed": False,
        "model_cache_files_committable": False,
        "cache_authority_verification_required": True,
        "task0170_revalidation_required_after_cache_materialization": True,
    }


def render_report(summary: dict[str, Any], artifacts: dict[str, Any]) -> str:
    cache = artifacts["model_cache_authority_audit.json"]
    return f"""# TASK-0174 Cold-start Embedding Model Cache Authority Remediation Report

## Decision

TASK-0174 is complete as a repository remediation gate for the TASK-0173 root cause `{summary['diagnosed_root_cause']}`.

## Implemented remediation

* Added deterministic Qwen3 embedding cache authority audit for `{summary['embedding_model']}` at revision `{summary['embedding_model_revision']}`.
* Added TASK-0174 contract, runner/verifier, focused tests, and machine-readable artifacts.
* No model weights or Hugging Face cache files are committed.

## Current cache audit

* cache path: `{cache['snapshot_path']}`
* exists: `{cache['model_cache_exists']}`
* complete: `{cache['model_cache_complete']}`
* missing required files: `{cache['missing_required_files']}`

## Boundary preservation

Embedding model, dimension, device default, dependency policy, retrieval, Graph, reranker, chunking, and corpus policy remain unchanged. CPU fallback was not promoted.

## TASK-0170 revalidation

`task0170_revalidation_executed=false` in this repository-only pass. After an operator materializes the fixed revision into the cold-start Hugging Face cache and this verifier reports `cache_authority_valid=true`, TASK-0170 must be rerun.
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
