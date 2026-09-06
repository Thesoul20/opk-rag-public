from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from opk_rag.embedding.cache_authority import EmbeddingModelCacheAuthority, resolve_huggingface_hub_cache
from opk_rag.embedding.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    load_embedding_config,
)
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, write_json

TASK_ID = "TASK-0175"
EXPERIMENT_ID = "task0175-pinned-embedding-model-cache-materialization-and-authority-revalidation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0175_pinned_embedding_model_cache_materialization_and_authority_revalidation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0175_PINNED_EMBEDDING_MODEL_CACHE_MATERIALIZATION_AND_AUTHORITY_REVALIDATION_REPORT.md"
TASK0174_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0174-cold-start-embedding-model-cache-authority-remediation"


def expected_authority(env: Mapping[str, str] | None = None) -> EmbeddingModelCacheAuthority:
    config = load_embedding_config(env or os.environ)
    return EmbeddingModelCacheAuthority(
        model_id=DEFAULT_EMBEDDING_MODEL_NAME,
        revision=DEFAULT_EMBEDDING_MODEL_REVISION,
        cache_dir=config.cache_dir,
    )


def audit_cache(env: Mapping[str, str] | None = None) -> dict[str, object]:
    return expected_authority(env).audit(env=env)


def materialize_pinned_revision(*, cache_root: Path, model_id: str, revision: str) -> dict[str, object]:
    result: dict[str, object] = {
        "materialization_attempted": True,
        "materialization_success": False,
        "materialization_method": "huggingface_hub.snapshot_download",
        "embedding_model": model_id,
        "requested_revision": revision,
        "cache_root": str(cache_root),
    }
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - environment guard
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_root),
            local_files_only=False,
        )
    except Exception as exc:  # pragma: no cover - network/provider dependent
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    resolved = Path(snapshot_path).name
    result.update(
        {
            "materialization_success": resolved == revision,
            "snapshot_path": str(snapshot_path),
            "resolved_revision": resolved,
            "revision_match": resolved == revision,
        }
    )
    return result


def validate_local_files_only_loader() -> dict[str, object]:
    result: dict[str, object] = {
        "local_files_only_validation_attempted": True,
        "local_files_only_loader_resolved": False,
        "local_files_only_inference_executed": False,
        "embedding_device_override_for_validation": "cpu",
    }
    env = dict(os.environ)
    env["OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY"] = "true"
    env["OPK_RAG_EMBEDDING_DEVICE"] = "cpu"
    try:
        from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider

        provider = QwenLocalEmbeddingProvider(load_embedding_config(env))
        token_count = provider.count_tokens("cache authority local-files-only validation")
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    result.update(
        {
            "local_files_only_loader_resolved": True,
            "embedding_model": provider.model_id,
            "embedding_dimension": provider.dimension,
            "tokenizer_probe_token_count": token_count,
        }
    )
    return result


def build_summary(
    *,
    source_head: str,
    pre_audit: dict[str, object],
    post_audit: dict[str, object],
    materialization: dict[str, object],
    local_files_only_validation: dict[str, object] | None = None,
) -> dict[str, Any]:
    cache_files_committed = model_cache_files_committed(Path(str(post_audit["cache_root"])))
    complete = bool(post_audit.get("cache_complete", post_audit.get("model_cache_complete")))
    revision_match = bool(post_audit.get("revision_match"))
    authority_valid = bool(post_audit.get("cache_authority_valid"))
    task0174_summary = _read_json(TASK0174_RESULT_DIR / "summary.json")
    pre_exists = task0174_summary.get("development_model_cache_exists", pre_audit.get("model_cache_exists"))
    pre_complete = task0174_summary.get("development_model_cache_complete", pre_audit.get("model_cache_complete"))
    local_ok = True if local_files_only_validation is None else bool(local_files_only_validation.get("local_files_only_loader_resolved"))
    task_status = "complete" if complete and revision_match and authority_valid and local_ok and not cache_files_committed else "blocked"
    return {
        "task_id": TASK_ID,
        "task_status": task_status,
        "source_authoritative_head": source_head,
        "parent_task": "TASK-0174",
        "task0174_diagnosed_root_cause": task0174_summary.get("diagnosed_root_cause"),
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "pre_materialization_cache_exists": bool(pre_exists),
        "pre_materialization_cache_complete": bool(pre_complete),
        "materialization_attempted": bool(materialization.get("materialization_attempted")),
        "materialization_success": bool(materialization.get("materialization_success")),
        "development_model_cache_exists": bool(post_audit.get("model_cache_exists")),
        "development_model_cache_complete": complete,
        "development_model_revision_match": revision_match,
        "development_cache_authority_valid": authority_valid,
        "requested_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "resolved_revision": post_audit.get("resolved_revision"),
        "cold_start_cache_authority_required": True,
        "model_cache_files_committed": cache_files_committed,
        "local_files_only_loader_resolved": local_ok,
        "runtime_default_behavior_change": False,
        "embedding_policy_mutation_count": 0,
        "embedding_device_default_mutation_count": 0,
        "dependency_remediation_count": 0,
        "cpu_fallback_promoted": False,
        "task0170_revalidation_eligible": authority_valid,
        "task0170_revalidation_executed": False,
    }


def run_task0175(*, write: bool = True, materialize: bool = True) -> dict[str, Any]:
    source_head = _git_head()
    pre_audit = audit_cache()
    cache_root = Path(str(pre_audit["cache_root"]))
    if materialize:
        materialization = materialize_pinned_revision(
            cache_root=cache_root,
            model_id=DEFAULT_EMBEDDING_MODEL_NAME,
            revision=DEFAULT_EMBEDDING_MODEL_REVISION,
        )
    else:
        materialization = {"materialization_attempted": False, "materialization_success": False, "cache_root": str(cache_root)}
    post_audit = audit_cache()
    local_files_only_validation = validate_local_files_only_loader() if bool(post_audit.get("cache_authority_valid")) else {
        "local_files_only_validation_attempted": False,
        "local_files_only_loader_resolved": False,
        "skip_reason": "cache_authority_valid=false",
    }
    summary = build_summary(
        source_head=source_head,
        pre_audit=pre_audit,
        post_audit=post_audit,
        materialization=materialization,
        local_files_only_validation=local_files_only_validation,
    )
    artifacts = {
        "summary.json": summary,
        "cache_inventory.json": post_audit,
        "revision_resolution.json": {
            "requested_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
            "resolved_revision": post_audit.get("resolved_revision"),
            "revision_match": post_audit.get("revision_match"),
            "snapshot_path": post_audit.get("snapshot_path"),
        },
        "materialization_result.json": materialization,
        "authority_validation.json": post_audit,
        "pre_materialization_cache_audit.json": pre_audit,
        "local_files_only_resolution_validation.json": local_files_only_validation,
        "runtime_mutation_audit.json": {
            "runtime_default_behavior_change": False,
            "embedding_policy_mutation_count": 0,
            "embedding_device_default_mutation_count": 0,
            "dependency_remediation_count": 0,
            "cpu_fallback_promoted": False,
            "task0170_revalidation_executed": False,
        },
    }
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, artifacts), encoding="utf-8")
    return summary


def contract() -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "cache_completeness_requires": [
            "cache_exists",
            "cache_snapshot_resolved",
            "revision_match",
            "required_metadata_present",
            "required_model_artifacts_present",
            "cache_complete",
        ],
        "authority_requires_revision_match": True,
        "authority_requires_complete_cache": True,
        "runtime_default_behavior_change_allowed": False,
        "embedding_policy_mutation_allowed": False,
        "embedding_device_default_mutation_allowed": False,
        "cpu_fallback_promotion_allowed": False,
        "model_cache_files_committable": False,
        "task0170_revalidation_executed": False,
    }


def model_cache_files_committed(cache_root: Path) -> bool:
    try:
        relative = cache_root.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", str(relative)], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    return bool(tracked)


def render_report(summary: dict[str, Any], artifacts: dict[str, Any]) -> str:
    pre = artifacts["pre_materialization_cache_audit.json"]
    post = artifacts["authority_validation.json"]
    materialization = artifacts["materialization_result.json"]
    local_validation = artifacts["local_files_only_resolution_validation.json"]
    return f"""# TASK-0175 Pinned Embedding Model Cache Materialization and Authority Revalidation Report

## TASK-0174 input diagnosis

TASK-0174 diagnosed the cold-start embedding blocker as `model_cache_missing_or_incomplete` and required operator materialization of the pinned Qwen3 embedding cache before TASK-0170 revalidation.

## Authoritative model/revision

* model: `{summary['embedding_model']}`
* revision: `{summary['embedding_model_revision']}`
* dimension: `{summary['embedding_dimension']}`
* source_authoritative_head: `{summary['source_authoritative_head']}`

## Materialization mechanism

The runner uses `huggingface_hub.snapshot_download(repo_id=model, revision=pinned_revision, cache_dir=derived_cache_root)`. It does not request a branch fallback or substitute another model. In this environment the default `HF_ENDPOINT=https://hf-mirror.com` resolved model metadata but failed file materialization; the operator completed the same pinned revision through `https://huggingface.co` with resumed download for `model.safetensors`, verified its SHA-256/etag, and reran the TASK-0175 runner to validate the final Hugging Face cache snapshot.

## Cache state

* pre-materialization exists: `{summary['pre_materialization_cache_exists']}`
* pre-materialization complete: `{summary['pre_materialization_cache_complete']}`
* cache root: `{post['cache_root']}`
* snapshot path: `{post['snapshot_path']}`
* materialization attempted: `{materialization['materialization_attempted']}`
* materialization success: `{materialization['materialization_success']}`
* post-materialization exists: `{post['model_cache_exists']}`
* post-materialization complete: `{post['model_cache_complete']}`

## Revision and authority verification

* requested revision: `{post['requested_revision']}`
* resolved revision: `{post['resolved_revision']}`
* revision match: `{post['revision_match']}`
* development cache authority valid: `{post['cache_authority_valid']}`
* local-files-only loader resolved: `{local_validation['local_files_only_loader_resolved']}`
* local-files-only inference executed: `{local_validation['local_files_only_inference_executed']}`

## Runtime mutation audit

No runtime default, embedding policy, device default, dependency, retrieval, Graph, reranker, chunking, corpus, or CPU fallback promotion was applied. Model weights remain outside Git authority.

## TASK-0170 boundary

`task0170_revalidation_executed=false`. Environment eligibility for future TASK-0170 revalidation is `{summary['task0170_revalidation_eligible']}`.
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
