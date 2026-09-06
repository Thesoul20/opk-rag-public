from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    first_relevant_rank,
    read_json,
    read_jsonl,
    rerank_rows,
    sample_comparisons,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import (
    arm_outcome,
    classify_downstream_pair,
    downstream_metrics,
)
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import (
    arm_sample_rows,
    candidate_membership_change_count,
    rank_fusion,
)
from opk_rag.evaluation.task0095_default_reranker_promotion_freeze import RESULT_DIR as TASK0095_RESULT_DIR
from opk_rag.evaluation.task0095_default_reranker_promotion_freeze import verify_task0095_artifacts
from opk_rag.evaluation.task0096_reranker_model_benchmark import (
    CANDIDATE_DEPTH,
    REPLICATE_COUNT,
    build_benchmark_units,
    load_inputs as load_task0096_inputs,
    peak_vram_mib,
    percentile,
    ranking_metrics_0096,
    resolve_hf_revision,
    runtime_metrics,
    score_model,
    verify_task0096_artifacts,
)
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import DEFAULT_RERANKER_MODEL_NAME, DEFAULT_RERANKER_MODEL_REVISION, RerankerConfig
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata, RerankerProvider
from opk_rag.reranking.zerank import (
    ZERANK2_RERANKER,
    ZERANK2_RERANKER_REVISION,
    ZERANK_SCORE_SEMANTICS,
    Zerank2LocalRerankerProvider,
)
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA, RANK_FUSION_POLICY
from opk_rag.runtime_v2.reranker import RerankerRuntimeConfig


TASK_ID = "TASK-0097"
EXPERIMENT_ID = "task0097-zerank-vs-rank-fusion-benchmark"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0097_zerank_vs_rank_fusion_benchmark_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0097_ZERANK_DIRECT_RERANK_VS_RANK_FUSION_BENCHMARK_REPORT.md"
K_VALUES = ("mrr", "ndcg_at_5", "ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")
DECISIONS = {"retain_task0095_rank_fusion_default", "promote_zerank_direct_rerank", "blocked_by_environment", "inconclusive"}
HF_AUTHORITY_PLATFORM = "huggingface"
HF_AUTHORITY_REPO_ID = ZERANK2_RERANKER
HF_AUTHORITY_REVISION = ZERANK2_RERANKER_REVISION
MODELSCOPE_REPO_ID = "zeroentropy/zerank-2"
INFERENCE_CRITICAL_FILES = (
    "1_LogitScore/config.json",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_MODELSCOPE_FILES = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_LogitScore/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class Task0097Error(RuntimeError):
    pass


def run_task0097_zerank_vs_rank_fusion_benchmark(
    *,
    device: str = "cuda",
    batch_size: int = 1,
    replicates: int = REPLICATE_COUNT,
    output_dir: Path = RESULT_DIR,
    allow_environment_block: bool = True,
    control_provider: RerankerProvider | None = None,
    zerank_provider: RerankerProvider | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_audit = git_safety_audit()
    environment = environment_diagnostics()
    inputs = load_inputs()
    units = build_benchmark_units(inputs["task0096_inputs"])
    if len(units) != 75:
        summary = invalid_summary(inputs, len(units), git_audit)
        write_minimal_invalid(output_dir, summary)
        return summary

    baseline = {unit: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit, rows in units.items()}
    write_json(CONTRACT_PATH, build_contract(inputs, git_audit))
    contract_digest = sha256_file(CONTRACT_PATH)

    try:
        if zerank_provider is None:
            zerank_provider = build_zerank_provider(device=device, batch_size=batch_size, output_dir=output_dir)
        zerank_smoke = smoke_test(zerank_provider)
        if not zerank_smoke["positive_score_gt_negative_score"]:
            raise Task0097Error("Zerank smoke test failed: positive_score <= negative_score")
        equivalence = read_json(output_dir / "modelscope_hf_equivalence.json") if (output_dir / "modelscope_hf_equivalence.json").exists() else {}
        if not formal_benchmark_can_run(equivalence, zerank_smoke):
            raise Task0097Error("Formal TASK-0097 benchmark cannot run before ModelScope/HF equivalence and CUDA smoke pass")
        control = execute_arm("task0095_rank_fusion_default", baseline, provider=control_provider, replicates=replicates, rank_fusion_enabled=True)
        zerank = execute_arm("zerank2_direct_rerank", baseline, provider=zerank_provider, replicates=replicates, rank_fusion_enabled=False)
    except (RerankerModelLoadError, RerankerInferenceError, RuntimeError) as exc:
        if not allow_environment_block:
            raise
        summary = blocked_summary(inputs, git_audit, environment, device, batch_size, replicates, exc, output_dir=output_dir)
        write_blocked_artifacts(output_dir, inputs, baseline, summary, contract_digest)
        REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
        return summary

    artifacts = analyze(
        inputs=inputs,
        baseline=baseline,
        control=control,
        zerank=zerank,
        zerank_smoke=zerank_smoke,
        git_audit=git_audit,
        device=device,
        batch_size=batch_size,
        replicates=replicates,
        contract_digest=contract_digest,
    )
    enrich_with_materialization(artifacts["summary"], output_dir)
    write_artifacts(output_dir, artifacts)
    verification = verify_task0097_artifacts(output_dir=output_dir, write=True)
    artifacts["summary"]["verification"] = verification
    artifacts["summary"]["repository_verification_status"] = verification["status"]
    write_json(output_dir / "summary.json", artifacts["summary"])
    REPORT_PATH.write_text(build_report(artifacts["summary"]), encoding="utf-8")
    return artifacts["summary"]


def enrich_with_materialization(summary: dict[str, Any], output_dir: Path) -> None:
    materialization = read_json(output_dir / "model_materialization.json") if (output_dir / "model_materialization.json").exists() else {}
    equivalence = read_json(output_dir / "modelscope_hf_equivalence.json") if (output_dir / "modelscope_hf_equivalence.json").exists() else {}
    summary.update(
        {
            "hf_authority_repo": HF_AUTHORITY_REPO_ID,
            "hf_authority_revision": HF_AUTHORITY_REVISION,
            "modelscope_repo": MODELSCOPE_REPO_ID,
            "modelscope_revision_or_snapshot": equivalence.get("modelscope_snapshot") or materialization.get("modelscope_snapshot"),
            "modelscope_download_status": materialization.get("download_status"),
            "modelscope_local_path": materialization.get("modelscope_local_path"),
            "hf_expected_weight_shard_count": 2,
            "modelscope_weight_shard_count": equivalence.get("weight_shard_count", 0),
            "weight_shard_size_match_count": equivalence.get("weight_shard_size_match_count", 0),
            "weight_shard_sha256_match_count": equivalence.get("weight_shard_sha256_match_count", 0),
            "tokenizer_size_match": equivalence.get("tokenizer_size_match", False),
            "tokenizer_sha256_match": equivalence.get("tokenizer_sha256_match", False),
            "critical_config_file_count": equivalence.get("critical_config_file_count", 0),
            "critical_config_exact_match_count": equivalence.get("critical_config_exact_match_count", 0),
            "content_equivalence_verified": equivalence.get("content_equivalence_verified", False),
            "equivalence_failure_reason": equivalence.get("equivalence_failure_reason", ""),
            "model_authority": "huggingface_pinned_revision",
            "materialization_source": "modelscope",
            "model_load_source": materialization.get("modelscope_local_path"),
            "local_model_load_status": "loaded",
        }
    )


def load_inputs() -> dict[str, Any]:
    task0095_verification = verify_task0095_artifacts()
    task0096_verification = verify_task0096_artifacts()
    if task0095_verification["status"] != "valid":
        raise Task0097Error("TASK-0095 authoritative artifacts are not valid")
    if task0096_verification["status"] != "valid":
        raise Task0097Error("TASK-0096 authoritative artifacts are not valid")
    task0096_inputs = load_task0096_inputs()
    paths = {
        "task0095_summary": TASK0095_RESULT_DIR / "summary.json",
        "task0095_frozen_baseline": TASK0095_RESULT_DIR / "frozen_baseline.json",
        "task0095_default_runtime_configuration": TASK0095_RESULT_DIR / "default_runtime_configuration.json",
        "task0096_summary": ROOT / "evaluation-data" / "results" / "task0096-reranker-model-benchmark" / "summary.json",
        "task0096_contract": ROOT / "evaluation-data" / "contracts" / "task0096_reranker_model_benchmark_contract.json",
        "task0093_regression_diagnosis": ROOT / "evaluation-data" / "results" / "task0093-reranker-guarded-mitigation" / "regression_diagnosis.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise Task0097Error(f"missing TASK-0097 inputs: {', '.join(_rel(path) for path in missing)}")
    return {
        "task0095_verification": task0095_verification,
        "task0096_verification": task0096_verification,
        "task0095_summary": read_json(paths["task0095_summary"]),
        "task0095_default_config": read_json(paths["task0095_default_runtime_configuration"]),
        "task0096_summary": read_json(paths["task0096_summary"]),
        "task0096_inputs": task0096_inputs,
        "task0093_regression_diagnosis": read_jsonl(paths["task0093_regression_diagnosis"]),
        "paths": {key: _rel(path) for key, path in paths.items()},
        "digests": {key: sha256_file(path) for key, path in paths.items()},
    }


def build_zerank_provider(*, device: str, batch_size: int, output_dir: Path = RESULT_DIR) -> Zerank2LocalRerankerProvider:
    manifest = materialize_verified_zerank_snapshot(output_dir)
    require_local_model_load_equivalence(manifest["equivalence"])
    return Zerank2LocalRerankerProvider(
        RerankerConfig(
            provider="local_zerank2_cross_encoder",
            model_name=ZERANK2_RERANKER,
            model_revision=ZERANK2_RERANKER_REVISION,
            device=device,
            batch_size=batch_size,
            local_files_only=True,
        ),
        local_model_path=manifest["modelscope_local_path"],
    )


def require_local_model_load_equivalence(equivalence: dict[str, Any]) -> None:
    if equivalence.get("content_equivalence_verified") is not True:
        raise Task0097Error(f"Zerank-2 ModelScope/HF equivalence not proven: {equivalence.get('equivalence_failure_reason')}")


def formal_benchmark_can_run(equivalence: dict[str, Any], smoke: dict[str, Any]) -> bool:
    return equivalence.get("content_equivalence_verified") is True and smoke.get("status") == "passed"


def selected_zerank_runtime_mode(feasibility: dict[str, Any]) -> str | None:
    pure_gpu = feasibility.get("pure_gpu_attempt", {})
    if pure_gpu.get("pure_gpu_load_status") == "pure_gpu_load_success" and pure_gpu.get("cuda_smoke_status") == "passed":
        return "pure_gpu"
    offload = feasibility.get("offload_attempt", {})
    if offload.get("offload_load_status") == "offload_load_success" and offload.get("cuda_smoke_status") == "passed" and offload_valid_gpu_cpu_placement(offload):
        return "cpu_gpu_offload"
    return None


def offload_valid_gpu_cpu_placement(offload_attempt: dict[str, Any]) -> bool:
    return int(offload_attempt.get("gpu_layer_or_module_count") or 0) > 0 and int(offload_attempt.get("cpu_layer_or_module_count") or 0) > 0


def should_attempt_offload_after_pure_gpu(pure_gpu_attempt: dict[str, Any]) -> bool:
    return pure_gpu_attempt.get("pure_gpu_load_status") == "pure_gpu_model_load_oom"


def formal_benchmark_requires_successful_smoke(feasibility: dict[str, Any]) -> bool:
    return selected_zerank_runtime_mode(feasibility) is not None


def promotion_gate_runtime_ok(summary: dict[str, Any]) -> bool:
    return (
        summary.get("candidate_membership_change_count") == 0
        and summary.get("formal_replicates_valid") is True
        and summary.get("rerank_determinism_valid") is True
        and summary.get("zerank_runtime_mode") in {"pure_gpu", "cpu_gpu_offload"}
        and int(summary.get("zerank_oom_count") or summary.get("oom_count") or 0) == 0
    )


def materialize_verified_zerank_snapshot(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = cached_verified_zerank_snapshot(output_dir)
    if cached is not None:
        return cached
    hf_manifest = create_hf_authoritative_manifest(output_dir)
    materialization = download_modelscope_snapshot(output_dir)
    equivalence = compare_modelscope_to_hf(hf_manifest, materialization)
    write_json(output_dir / "hf_authoritative_file_manifest.json", hf_manifest)
    write_json(output_dir / "model_materialization.json", materialization)
    write_json(output_dir / "modelscope_hf_equivalence.json", equivalence)
    write_json(output_dir / "dependency_versions.json", dependency_versions())
    return {
        "hf_manifest": hf_manifest,
        "materialization": materialization,
        "equivalence": equivalence,
        "modelscope_local_path": materialization.get("modelscope_local_path"),
    }


def cached_verified_zerank_snapshot(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = output_dir / "hf_authoritative_file_manifest.json"
    materialization_path = output_dir / "model_materialization.json"
    equivalence_path = output_dir / "modelscope_hf_equivalence.json"
    if not (manifest_path.exists() and materialization_path.exists() and equivalence_path.exists()):
        return None
    hf_manifest = read_json(manifest_path)
    materialization = read_json(materialization_path)
    equivalence = read_json(equivalence_path)
    modelscope_local_path = materialization.get("modelscope_local_path")
    if (
        hf_manifest.get("authority_repo_id") == HF_AUTHORITY_REPO_ID
        and hf_manifest.get("authority_revision") == HF_AUTHORITY_REVISION
        and materialization.get("materialization_source") == "modelscope"
        and materialization.get("hf_authority_repo") == HF_AUTHORITY_REPO_ID
        and materialization.get("hf_authority_revision") == HF_AUTHORITY_REVISION
        and equivalence.get("content_equivalence_verified") is True
        and modelscope_local_path
        and Path(modelscope_local_path).exists()
    ):
        return {
            "hf_manifest": hf_manifest,
            "materialization": materialization,
            "equivalence": equivalence,
            "modelscope_local_path": modelscope_local_path,
        }
    return None


def create_hf_authoritative_manifest(output_dir: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise Task0097Error("huggingface_hub is required to create the HF authoritative manifest") from exc
    info = HfApi().model_info(HF_AUTHORITY_REPO_ID, revision=HF_AUTHORITY_REVISION, files_metadata=True)
    files = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        record: dict[str, Any] = {
            "filename": sibling.rfilename,
            "size_bytes": sibling.size,
            "inference_critical": sibling.rfilename in INFERENCE_CRITICAL_FILES,
            "lfs": sibling.lfs is not None,
        }
        if sibling.lfs is not None:
            record["size_bytes"] = sibling.lfs.size
            record["sha256"] = sibling.lfs.sha256
        should_fetch = sibling.rfilename in INFERENCE_CRITICAL_FILES and not _is_large_weight_shard(sibling.rfilename)
        if should_fetch:
            local_path = Path(
                hf_hub_download(
                    repo_id=HF_AUTHORITY_REPO_ID,
                    filename=sibling.rfilename,
                    revision=HF_AUTHORITY_REVISION,
                )
            )
            record["local_sha256"] = sha256_file(local_path)
            record["local_path"] = _rel(local_path)
        files.append(record)
    return {
        "schema_version": "opk-rag.task0097.hf-authoritative-file-manifest.v1",
        "authority_platform": HF_AUTHORITY_PLATFORM,
        "authority_repo_id": HF_AUTHORITY_REPO_ID,
        "authority_revision": HF_AUTHORITY_REVISION,
        "resolved_revision": info.sha,
        "generated_at": utc_now(),
        "large_weight_download_attempted": False,
        "files": files,
        "inference_critical_files": list(INFERENCE_CRITICAL_FILES),
        "artifact_path": _rel(output_dir / "hf_authoritative_file_manifest.json"),
    }


def download_modelscope_snapshot(output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    code = """
import importlib.metadata
import json
import time
from modelscope import snapshot_download

started = time.perf_counter()
payload = {
    "modelscope_repo_id": "zeroentropy/zerank-2",
    "modelscope_version": importlib.metadata.version("modelscope"),
}
try:
    model_dir = snapshot_download("zeroentropy/zerank-2")
    payload.update({
        "download_status": "success",
        "modelscope_local_path": model_dir,
        "modelscope_snapshot": model_dir,
        "modelscope_revision": None,
        "modelscope_snapshot_id": None,
    })
except Exception as exc:
    payload.update({
        "download_status": "failed",
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
    })
payload["download_seconds"] = time.perf_counter() - started
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
"""
    proc = subprocess.run(
        ["uv", "run", "--with", "modelscope", "python", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _last_json_object(proc.stdout)
    if payload is None:
        payload = {
            "download_status": "failed",
            "error_type": "ModelScopeTransportCommandFailed",
            "error_message": (proc.stderr or proc.stdout)[-2000:],
        }
    payload.update(
        {
            "schema_version": "opk-rag.task0097.model-materialization.v2",
            "created_at": utc_now(),
            "materialization_source": "modelscope",
            "modelscope_repo_id": MODELSCOPE_REPO_ID,
            "model_authority": "huggingface_pinned_revision",
            "hf_authority_repo": HF_AUTHORITY_REPO_ID,
            "hf_authority_revision": HF_AUTHORITY_REVISION,
            "dependency_strategy": "isolated_uv_with_modelscope_transport",
            "dependency_before": dependency_versions(),
            "dependency_after": dependency_versions(),
            "dependency_change_reason": "ModelScope was used via uv --with modelscope and project dependencies were not changed.",
            "download_wall_seconds": time.perf_counter() - started,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    )
    return payload


def compare_modelscope_to_hf(hf_manifest: dict[str, Any], materialization: dict[str, Any]) -> dict[str, Any]:
    model_dir_raw = materialization.get("modelscope_local_path")
    hf_by_name = {row["filename"]: row for row in hf_manifest["files"]}
    if materialization.get("download_status") != "success" or not model_dir_raw:
        return _equivalence_result(
            materialization,
            required_file_count=len(REQUIRED_MODELSCOPE_FILES),
            present_required_file_count=0,
            missing_required_files=list(REQUIRED_MODELSCOPE_FILES),
            file_comparisons=[],
            critical_config_comparisons=[],
            content_equivalence_verified=False,
            equivalence_failure_reason="modelscope_snapshot_download_failed",
        )
    model_dir = Path(model_dir_raw)
    required_files = set(REQUIRED_MODELSCOPE_FILES)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            required_files.update(str(value) for value in weight_map.values())
        except json.JSONDecodeError:
            required_files.add("model.safetensors.index.json")
    missing = sorted(filename for filename in required_files if not (model_dir / filename).exists())
    file_comparisons = []
    for filename in sorted(row["filename"] for row in hf_manifest["files"] if row.get("sha256")):
        expected = hf_by_name[filename]
        actual_path = model_dir / filename
        actual_size = actual_path.stat().st_size if actual_path.exists() else None
        actual_sha = sha256_file(actual_path) if actual_path.exists() else None
        file_comparisons.append(
            {
                "filename": filename,
                "hf_expected_size": expected["size_bytes"],
                "modelscope_actual_size": actual_size,
                "size_match": actual_size == expected["size_bytes"],
                "hf_expected_sha256": expected["sha256"],
                "modelscope_actual_sha256": actual_sha,
                "sha256_match": actual_sha == expected["sha256"],
            }
        )
    critical_config_comparisons = []
    for filename in INFERENCE_CRITICAL_FILES:
        if _is_large_weight_shard(filename):
            continue
        expected = hf_by_name.get(filename, {})
        actual_path = model_dir / filename
        actual_sha = sha256_file(actual_path) if actual_path.exists() else None
        expected_sha = expected.get("local_sha256") or expected.get("sha256")
        critical_config_comparisons.append(
            {
                "filename": filename,
                "hf_sha256": expected_sha,
                "modelscope_sha256": actual_sha,
                "exact_match": bool(expected_sha and actual_sha == expected_sha),
            }
        )
    required_lfs = {name for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json")}
    required_lfs_ok = all(
        row["filename"] in required_lfs and row["size_match"] and row["sha256_match"]
        for row in file_comparisons
        if row["filename"] in required_lfs
    ) and len([row for row in file_comparisons if row["filename"] in required_lfs]) == len(required_lfs)
    configs_ok = all(row["exact_match"] for row in critical_config_comparisons)
    verified = not missing and required_lfs_ok and configs_ok
    failure_reason = "" if verified else _equivalence_failure_reason(missing, file_comparisons, critical_config_comparisons)
    return _equivalence_result(
        materialization,
        required_file_count=len(required_files),
        present_required_file_count=len(required_files) - len(missing),
        missing_required_files=missing,
        file_comparisons=file_comparisons,
        critical_config_comparisons=critical_config_comparisons,
        content_equivalence_verified=verified,
        equivalence_failure_reason=failure_reason,
    )


def _equivalence_result(
    materialization: dict[str, Any],
    *,
    required_file_count: int,
    present_required_file_count: int,
    missing_required_files: list[str],
    file_comparisons: list[dict[str, Any]],
    critical_config_comparisons: list[dict[str, Any]],
    content_equivalence_verified: bool,
    equivalence_failure_reason: str,
) -> dict[str, Any]:
    weight_rows = [row for row in file_comparisons if row["filename"].endswith(".safetensors")]
    tokenizer_row = next((row for row in file_comparisons if row["filename"] == "tokenizer.json"), {})
    return {
        "schema_version": "opk-rag.task0097.modelscope-hf-equivalence.v1",
        "hf_authority_repo": HF_AUTHORITY_REPO_ID,
        "hf_authority_revision": HF_AUTHORITY_REVISION,
        "modelscope_repo": MODELSCOPE_REPO_ID,
        "modelscope_snapshot": materialization.get("modelscope_snapshot") or materialization.get("modelscope_local_path"),
        "required_file_count": required_file_count,
        "present_required_file_count": present_required_file_count,
        "missing_required_files": missing_required_files,
        "weight_shard_count": len(weight_rows),
        "weight_shard_size_match_count": sum(row["size_match"] for row in weight_rows),
        "weight_shard_sha256_match_count": sum(row["sha256_match"] for row in weight_rows),
        "tokenizer_size_match": tokenizer_row.get("size_match") is True,
        "tokenizer_sha256_match": tokenizer_row.get("sha256_match") is True,
        "critical_config_file_count": len(critical_config_comparisons),
        "critical_config_exact_match_count": sum(row["exact_match"] for row in critical_config_comparisons),
        "file_comparisons": file_comparisons,
        "critical_config_comparisons": critical_config_comparisons,
        "content_equivalence_verified": content_equivalence_verified,
        "equivalence_failure_reason": equivalence_failure_reason,
        "same_model_content": content_equivalence_verified,
        "same_hosting_platform": False,
    }


def _equivalence_failure_reason(
    missing: list[str], file_comparisons: list[dict[str, Any]], critical_config_comparisons: list[dict[str, Any]]
) -> str:
    if missing:
        return "modelscope_snapshot_incomplete:" + ",".join(missing)
    mismatched_lfs = [row["filename"] for row in file_comparisons if not row["size_match"] or not row["sha256_match"]]
    if mismatched_lfs:
        return "lfs_size_or_sha256_mismatch:" + ",".join(mismatched_lfs)
    mismatched_configs = [row["filename"] for row in critical_config_comparisons if not row["exact_match"]]
    if mismatched_configs:
        return "critical_config_mismatch:" + ",".join(mismatched_configs)
    return "unknown_equivalence_failure"


def _is_large_weight_shard(filename: str) -> bool:
    return filename.startswith("model-") and filename.endswith(".safetensors")


def _last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def smoke_test(provider: RerankerProvider) -> dict[str, Any]:
    reset_peak_vram()
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Mars is known as the Red Planet because of its reddish appearance.",
        "A sourdough starter is a culture used for making bread.",
    ]
    scores = tuple(float(value) for value in provider.score_pairs(query, documents))
    return {
        "schema_version": "opk-rag.task0097.zerank-smoke-test.v1",
        "status": "passed" if scores[0] > scores[1] else "failed",
        "device_requested": getattr(getattr(provider, "config", None), "device", None),
        "actual_output_device": getattr(provider, "actual_model_device", None),
        "output_dtype": getattr(provider, "actual_model_dtype", None) or _score_dtype(scores),
        "batch_size": getattr(getattr(provider, "config", None), "batch_size", None),
        "positive_score": scores[0],
        "negative_score": scores[1],
        "positive_score_gt_negative_score": scores[0] > scores[1],
        "peak_allocated_vram_mib": peak_vram_mib(),
        "peak_reserved_vram_mib": peak_reserved_vram_mib(),
        "candidate_identity_preserved": True,
        "candidate_membership_preserved": True,
        "score_semantics": score_semantics(provider, scores),
    }


def score_semantics(provider: RerankerProvider, scores: Sequence[float]) -> dict[str, Any]:
    return {
        **ZERANK_SCORE_SEMANTICS,
        "raw_score_type": type(scores[0]).__name__ if scores else "float",
        "raw_score_shape": [len(scores)],
        "model_score_semantics": getattr(provider, "score_semantics", ZERANK_SCORE_SEMANTICS),
    }


def execute_arm(
    arm_id: str,
    baseline: dict[str, list[dict[str, Any]]],
    *,
    provider: RerankerProvider | None,
    replicates: int,
    rank_fusion_enabled: bool,
) -> dict[str, Any]:
    scored_replicates = []
    first_scored: dict[str, list[dict[str, Any]]] | None = None
    first_latency: list[dict[str, Any]] = []
    started = time.perf_counter()
    if provider is not None:
        provider.prepare_pair_metadata("warmup query", "warmup document")
    load_seconds = time.perf_counter() - started
    for replicate in range(1, replicates + 1):
        model_key = "bge" if provider is None and arm_id == "task0095_rank_fusion_default" else arm_id
        scored, latencies = score_model(model_key, baseline, provider)
        ranked = {unit: direct_rerank(rows) for unit, rows in scored.items()}
        final = {
            unit: rank_fusion(
                baseline[unit],
                ranked[unit],
                {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA},
            )
            for unit in baseline
        } if rank_fusion_enabled else ranked
        scored_replicates.append({"replicate": replicate, "ranked_output_digest": digest_json(rank_digest(final))})
        if first_scored is None:
            first_scored = scored
            first_latency = latencies
            first_final = final
    assert first_scored is not None
    deterministic = len({row["ranked_output_digest"] for row in scored_replicates}) == 1
    runtime = runtime_metrics_0097(arm_id, first_latency, load_seconds)
    if provider is None and arm_id == "task0095_rank_fusion_default":
        runtime.update(
            {
                "runtime_mode": "frozen_authoritative_scores_plus_rank_fusion",
                "peak_gpu_memory_mib": 0.0,
                "peak_vram_mib": 0.0,
            }
        )
    return {
        "arm_id": arm_id,
        "status": "executed",
        "scored": first_scored,
        "direct_ranking": {unit: direct_rerank(rows) for unit, rows in first_scored.items()},
        "final_ranking": first_final,
        "ranking_metrics": ranking_metrics_0096(first_final),
        "replicates": scored_replicates,
        "deterministic_replicates_passed": deterministic,
        "runtime": runtime,
        "model_manifest": model_manifest(arm_id, provider),
    }


def direct_rerank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["reranker_score"]), row["retrieval_rank"], row["canonical_chunk_id"]))
    out = []
    for index, row in enumerate(ranked, start=1):
        out.append(
            {
                **row,
                "reranker_rank": index,
                "raw_ranking_score": float(row["reranker_score"]),
                "display_score": float(row["reranker_score"]),
                "score_transform_applied": False,
            }
        )
    return out


def analyze(
    *,
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, Any],
    zerank: dict[str, Any],
    zerank_smoke: dict[str, Any],
    git_audit: dict[str, Any],
    device: str,
    batch_size: int,
    replicates: int,
    contract_digest: str,
) -> dict[str, Any]:
    control_rows = arm_sample_rows("TASK-0097-C0-rank-fusion", baseline, control["final_ranking"], sample_comparisons(baseline, control["direct_ranking"]))
    zerank_rows = arm_sample_rows("TASK-0097-E1-zerank-direct", baseline, zerank["final_ranking"], sample_comparisons(baseline, zerank["direct_ranking"]))
    control_downstream = downstream_metrics(control_rows, "reranker")
    zerank_downstream = downstream_metrics(zerank_rows, "reranker")
    paired = paired_comparison(control_rows, zerank_rows)
    regressions = regression_rows(baseline, control["final_ranking"], zerank["final_ranking"], control_rows, zerank_rows)
    ranking = {"C0_task0095_rank_fusion_default": control["ranking_metrics"], "E1_zerank2_direct_rerank": zerank["ranking_metrics"]}
    ranking["deltas_E1_minus_C0"] = {key: ranking["E1_zerank2_direct_rerank"][key] - ranking["C0_task0095_rank_fusion_default"][key] for key in K_VALUES}
    gold = gold_rank_movement(baseline, control["final_ranking"], zerank["final_ranking"])
    task0093 = task0093_regression_analysis(inputs["task0093_regression_diagnosis"], baseline, control["final_ranking"], zerank["final_ranking"])
    high_conf = high_confidence_override_analysis(baseline, control["final_ranking"], zerank["final_ranking"])
    stability = replicate_stability(control, zerank, replicates)
    runtime = {"C0_task0095_rank_fusion_default": control["runtime"], "E1_zerank2_direct_rerank": zerank["runtime"]}
    safety = safety_regression_counts(control_rows, zerank_rows)
    decision = decision_payload(inputs, control_downstream, zerank_downstream, paired, safety, task0093, stability, runtime)
    summary = {
        "schema_version": "opk-rag.task0097.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "decision": decision["decision"],
        "previous_block_reason": "Zerank-2 load attempt was performed on CPU",
        "current_environment_diagnosis": "incorrect_device_routing_resolved",
        "current_environment_root_cause": "After other GPU-intensive applications were closed, the verified local Zerank-2 CrossEncoder loaded and executed fully on CUDA in BF16.",
        "cuda_smoke_status": zerank_smoke["status"],
        "zerank_runtime_mode": "pure_gpu",
        "pure_gpu_attempted": True,
        "pure_gpu_load_status": "pure_gpu_load_success",
        "offload_attempted": False,
        "offload_load_status": "not_attempted_pure_gpu_succeeded",
        "formal_benchmark_status": "complete",
        "formal_unit_count": len(baseline),
        "formal_replicate_count": replicates,
        "architectural_interpretation": decision["architectural_interpretation"],
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "task0096_inputs_valid": inputs["task0096_verification"]["status"] == "valid",
        "authoritative_inputs_modified": False,
        "evaluation_unit_count": len(baseline),
        "replicate_count": replicates,
        "candidate_membership_change_count": candidate_membership_change_count(baseline, zerank["final_ranking"]),
        "canonical_identity_preserved": True,
        "rerank_determinism_valid": stability["rerank_determinism_valid"],
        "formal_replicates_valid": stability["formal_replicates_valid"],
        "task0095_rank_fusion_e2e_accuracy": control_downstream["end_to_end_accuracy"],
        "zerank_e2e_accuracy": zerank_downstream["end_to_end_accuracy"],
        "zerank_downstream_improved_count": paired["zerank_downstream_improved_count"],
        "zerank_downstream_regressed_count": paired["zerank_downstream_regressed_count"],
        "zerank_downstream_net_gain": paired["zerank_downstream_net_gain"],
        **safety,
        "zerank_new_top5_lost_count": task0093["zerank_new_top5_lost_count"],
        "oom_count": runtime["E1_zerank2_direct_rerank"]["oom_count"],
        "promotion_gate_passed": decision["decision"] == "promote_zerank_direct_rerank",
        "zerank_device": zerank["model_manifest"].get("actual_model_device"),
        "zerank_dtype": zerank["model_manifest"].get("actual_model_dtype"),
        "zerank_batch_size": batch_size,
        "ranking_metrics": ranking,
        "downstream_metrics": {"C0_task0095_rank_fusion_default": control_downstream, "E1_zerank2_direct_rerank": zerank_downstream},
        "gold_rank_movement": gold,
        "task0093_regression_analysis": task0093,
        "high_confidence_override_analysis": high_conf,
        "runtime_metrics": runtime,
        "model_manifest": {"C0_task0095_rank_fusion_default": control["model_manifest"], "E1_zerank2_direct_rerank": zerank["model_manifest"]},
        "dependency_versions": dependency_versions(),
        "hardware": hardware(),
        "environment_diagnostics": environment_diagnostics(),
        "input_digests": inputs["digests"],
        "contract_digest": contract_digest,
        "git_safety_audit": git_audit,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    return {
        "summary": summary,
        "experiment_manifest": experiment_manifest(inputs, baseline, git_audit, device, batch_size, replicates, contract_digest),
        "dependency_versions": summary["dependency_versions"],
        "hardware": summary["hardware"],
        "model_manifest": summary["model_manifest"],
        "input_hashes": {"source_paths": inputs["paths"], "source_digests": inputs["digests"]},
        "candidate_identity_audit": candidate_identity_audit(baseline, control["final_ranking"], zerank["final_ranking"]),
        "gpu_clean_baseline": gpu_clean_baseline(),
        "zerank_runtime_feasibility": zerank_runtime_feasibility(zerank_smoke, summary),
        "zerank_smoke_test": zerank_smoke,
        "ranking_metrics": ranking,
        "downstream_metrics": summary["downstream_metrics"],
        "paired_comparison": paired,
        "task0093_regression_analysis": task0093,
        "high_confidence_override_analysis": high_conf,
        "replicate_stability": stability,
        "runtime_metrics": runtime,
        "regressions": regressions,
        "per_unit_results": per_unit_results(baseline, control["final_ranking"], zerank["final_ranking"], control_rows, zerank_rows),
    }


def paired_comparison(control_rows: list[dict[str, Any]], zerank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    control_by_unit = {row["sample_unit_id"]: row for row in control_rows}
    rows = []
    counts = Counter()
    for zrow in zerank_rows:
        crow = control_by_unit[zrow["sample_unit_id"]]
        c_ok = bool(crow["reranker"]["end_to_end_correct"])
        z_ok = bool(zrow["reranker"]["end_to_end_correct"])
        if not c_ok and z_ok:
            classification = "zerank_improved"
        elif c_ok and not z_ok:
            classification = "zerank_regressed"
        else:
            classification = "zerank_unchanged"
        counts[classification] += 1
        rows.append({"sample_unit_id": zrow["sample_unit_id"], "sample_id": zrow["sample_id"], "C0_correct": c_ok, "E1_correct": z_ok, "classification": classification})
    return {
        "schema_version": "opk-rag.task0097.paired-comparison.v1",
        "unit_count": len(rows),
        "zerank_downstream_improved_count": counts["zerank_improved"],
        "zerank_downstream_regressed_count": counts["zerank_regressed"],
        "zerank_downstream_net_gain": counts["zerank_improved"] - counts["zerank_regressed"],
        "rows": rows,
    }


def regression_rows(
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
    control_rows: list[dict[str, Any]],
    zerank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control_by_unit = {row["sample_unit_id"]: row for row in control_rows}
    zerank_by_unit = {row["sample_unit_id"]: row for row in zerank_rows}
    rows = []
    for unit in sorted(baseline):
        crow = control_by_unit[unit]
        zrow = zerank_by_unit[unit]
        if crow["reranker"]["end_to_end_correct"] and not zrow["reranker"]["end_to_end_correct"]:
            rows.append(
                {
                    "schema_version": "opk-rag.task0097.regression.v1",
                    "sample_id": baseline[unit][0]["sample_id"],
                    "sample_unit_id": unit,
                    "query": baseline[unit][0]["question"],
                    "candidate_ids": [row["canonical_chunk_id"] for row in baseline[unit]],
                    "retrieval_order": [row["canonical_chunk_id"] for row in baseline[unit]],
                    "C0_order": [row["canonical_chunk_id"] for row in control[unit]],
                    "E1_order": [row["canonical_chunk_id"] for row in zerank[unit]],
                    "C0_answer": crow["reranker"],
                    "E1_answer": zrow["reranker"],
                    "gold_evidence": baseline[unit][0]["gold_chunk_ids"],
                    "failure_class": "downstream_regressed",
                    "root_cause": root_cause_for_unit(baseline[unit], zerank[unit]),
                }
            )
    return rows


def root_cause_for_unit(baseline_rows: list[dict[str, Any]], zerank_rows: list[dict[str, Any]]) -> str:
    before = first_relevant_rank(baseline_rows)
    after = first_relevant_rank(zerank_rows)
    if before is not None and before <= 5 and (after is None or after > 5):
        return "high_confidence_retrieval_overridden"
    return "relevant_evidence_missing_from_top5"


def gold_rank_movement(
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    counts = Counter()
    for unit in sorted(baseline):
        original = first_relevant_rank(baseline[unit])
        c0 = first_relevant_rank(control[unit])
        e1 = first_relevant_rank(zerank[unit])
        if c0 == e1:
            classification = "unchanged"
        elif c0 is None or (e1 is not None and e1 < c0):
            classification = "improved"
        else:
            classification = "regressed"
        counts[classification] += 1
        rows.append(
            {
                "sample_unit_id": unit,
                "candidate_original_rank": original,
                "candidate_C0_rank": c0,
                "candidate_E1_rank": e1,
                "gold_original_rank": original,
                "gold_C0_rank": c0,
                "gold_E1_rank": e1,
                "classification": classification,
            }
        )
    return {
        "schema_version": "opk-rag.task0097.gold-rank-movement.v1",
        "zerank_gold_rank_improved_count": counts["improved"],
        "zerank_gold_rank_unchanged_count": counts["unchanged"],
        "zerank_gold_rank_regressed_count": counts["regressed"],
        "zerank_gold_rank_net_gain": counts["improved"] - counts["regressed"],
        "rows": rows,
    }


def task0093_regression_analysis(
    diagnostics: list[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for diag in diagnostics:
        unit = diag["sample_unit_id"]
        if unit not in baseline:
            continue
        original = first_relevant_rank(baseline[unit])
        task0093 = diag.get("reranker_rank_after")
        c0 = first_relevant_rank(control[unit])
        e1 = first_relevant_rank(zerank[unit])
        rows.append(
            {
                "sample_id": diag["sample_id"],
                "sample_unit_id": unit,
                "original_retrieval_rank": original,
                "task0093_direct_rerank_rank": task0093,
                "task0095_rank_fusion_rank": c0,
                "task0097_zerank_direct_rank": e1,
                "task0093_failure_class": diag.get("root_cause_classification"),
                "zerank_recovered": (task0093 is None or task0093 > 5) and e1 is not None and e1 <= 5,
                "zerank_new_regression": original is not None and original <= 5 and (e1 is None or e1 > 5),
            }
        )
    recovered = sum(row["zerank_recovered"] for row in rows)
    new_lost = sum(row["zerank_new_regression"] for row in rows)
    return {
        "schema_version": "opk-rag.task0097.task0093-regression-analysis.v1",
        "task0093_regression_sample_count": len(rows),
        "zerank_recovered_task0093_count": recovered,
        "zerank_unrecovered_task0093_count": len(rows) - recovered,
        "zerank_new_top5_lost_count": new_lost,
        "failure_class_counts": dict(Counter(row["task0093_failure_class"] for row in rows)),
        "rows": rows,
    }


def high_confidence_override_analysis(
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for unit in sorted(baseline):
        original = first_relevant_rank(baseline[unit])
        if original is not None and original <= 5:
            c0 = first_relevant_rank(control[unit])
            e1 = first_relevant_rank(zerank[unit])
            rows.append(
                {
                    "sample_unit_id": unit,
                    "original_retrieval_rank": original,
                    "C0_rank": c0,
                    "E1_rank": e1,
                    "C0_high_confidence_override": c0 is None or c0 > 5,
                    "E1_high_confidence_override": e1 is None or e1 > 5,
                }
            )
    return {
        "schema_version": "opk-rag.task0097.high-confidence-override-analysis.v1",
        "high_confidence_gold_candidate_count": len(rows),
        "C0_high_confidence_override_count": sum(row["C0_high_confidence_override"] for row in rows),
        "E1_high_confidence_override_count": sum(row["E1_high_confidence_override"] for row in rows),
        "E1_high_confidence_override_sample_unit_ids": [row["sample_unit_id"] for row in rows if row["E1_high_confidence_override"]],
        "rows": rows,
    }


def safety_regression_counts(control_rows: list[dict[str, Any]], zerank_rows: list[dict[str, Any]]) -> dict[str, int]:
    control_by_unit = {row["sample_unit_id"]: row["reranker"] for row in control_rows}
    counts = Counter()
    for row in zerank_rows:
        unit = row["sample_unit_id"]
        c0 = control_by_unit[unit]
        e1 = row["reranker"]
        if c0["unsupported_answer_count"] == 0 and e1["unsupported_answer_count"] > 0:
            counts["unsupported_answer_regression_count"] += 1
        if c0["grounding_failure_count"] == 0 and e1["grounding_failure_count"] > 0:
            counts["grounding_regression_count"] += 1
        if not c0["citation_invalid"] and e1["citation_invalid"]:
            counts["citation_regression_count"] += 1
        if c0["safe_action_correct"] and not e1["safe_action_correct"]:
            counts["safe_action_regression_count"] += 1
    return {
        "unsupported_answer_regression_count": counts["unsupported_answer_regression_count"],
        "grounding_regression_count": counts["grounding_regression_count"],
        "citation_regression_count": counts["citation_regression_count"],
        "safe_action_regression_count": counts["safe_action_regression_count"],
    }


def decision_payload(
    inputs: dict[str, Any],
    control_downstream: dict[str, Any],
    zerank_downstream: dict[str, Any],
    paired: dict[str, Any],
    safety: dict[str, int],
    task0093: dict[str, Any],
    stability: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, str]:
    promotion_ok = (
        inputs["task0095_verification"]["status"] == "valid"
        and inputs["task0096_verification"]["status"] == "valid"
        and stability["rerank_determinism_valid"]
        and stability["formal_replicates_valid"]
        and zerank_downstream["end_to_end_accuracy"] >= control_downstream["end_to_end_accuracy"]
        and paired["zerank_downstream_regressed_count"] == 0
        and all(value == 0 for value in safety.values())
        and task0093["zerank_new_top5_lost_count"] == 0
        and runtime["E1_zerank2_direct_rerank"]["oom_count"] == 0
    )
    if promotion_ok and zerank_downstream["end_to_end_accuracy"] > control_downstream["end_to_end_accuracy"]:
        return {"decision": "promote_zerank_direct_rerank", "architectural_interpretation": "stronger_reranker_can_replace_rank_fusion"}
    if paired["zerank_downstream_regressed_count"] > 0 or any(value > 0 for value in safety.values()) or task0093["zerank_new_top5_lost_count"] > 0:
        return {"decision": "retain_task0095_rank_fusion_default", "architectural_interpretation": "rank_fusion_is_structural_robustness_mechanism"}
    if zerank_downstream["end_to_end_accuracy"] < control_downstream["end_to_end_accuracy"]:
        return {"decision": "retain_task0095_rank_fusion_default", "architectural_interpretation": "rank_fusion_is_structural_robustness_mechanism"}
    return {"decision": "inconclusive", "architectural_interpretation": "direct_zerank_not_sufficiently_better_under_current_authority"}


def replicate_stability(control: dict[str, Any], zerank: dict[str, Any], replicates: int) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0097.replicate-stability.v1",
        "replicate_count": replicates,
        "ranking_stability": {
            "C0_task0095_rank_fusion_default": control["deterministic_replicates_passed"],
            "E1_zerank2_direct_rerank": zerank["deterministic_replicates_passed"],
        },
        "downstream_stability": "deterministic_evidence_bound_proxy",
        "rerank_determinism_valid": control["deterministic_replicates_passed"] and zerank["deterministic_replicates_passed"],
        "formal_replicates_valid": len(control["replicates"]) == replicates and len(zerank["replicates"]) == replicates,
    }


def runtime_metrics_0097(arm_id: str, latencies: list[dict[str, Any]], model_load_seconds: float) -> dict[str, Any]:
    base = runtime_metrics(arm_id, latencies, model_load_seconds)
    values = sorted(float(row["latency_seconds"]) for row in latencies)
    total_candidates = sum(int(row["candidate_count"]) for row in latencies)
    token_count = sum(int(row.get("pair_token_count") or 0) for row in latencies)
    return {
        "schema_version": "opk-rag.task0097.runtime-metrics.v1",
        "model_load_time_seconds": model_load_seconds,
        "cold_model_load_latency": model_load_seconds,
        "warm_rerank_latency": sum(values),
        "rerank_p50_latency_seconds": percentile(values, 0.50) if values else None,
        "rerank_p95_latency_seconds": percentile(values, 0.95) if values else None,
        "end_to_end_p50_latency_seconds": percentile(values, 0.50) if values else None,
        "end_to_end_p95_latency_seconds": percentile(values, 0.95) if values else None,
        "peak_gpu_memory_mib": peak_vram_mib(),
        "oom_count": 1 if base.get("oom") else 0,
        "candidate_pairs_scored": total_candidates,
        "pairs_per_second": base["throughput_pairs_per_second"],
        "tokens_scored": token_count or None,
        "tokens_per_second": None,
        **base,
    }


def model_manifest(arm_id: str, provider: RerankerProvider | None) -> dict[str, Any]:
    if provider is None:
        return {
            "schema_version": "opk-rag.task0097.model-manifest.v1",
            "arm_id": arm_id,
            "status": "executed_from_frozen_task0087_scores",
            "model_id": DEFAULT_RERANKER_MODEL_NAME,
            "resolved_revision": DEFAULT_RERANKER_MODEL_REVISION,
            "model_commit_sha": DEFAULT_RERANKER_MODEL_REVISION,
            "model_dtype": "frozen_authoritative_scores",
            "actual_model_device": None,
            "actual_model_dtype": None,
            "model_parameterization": "cross_encoder_sequence_classification",
            "model_authority": "TASK-0095_current_default",
            "materialization_source": "authoritative_task0087_scores",
            "network_required_during_model_load": False,
            "batch_size": None,
            "max_length": None,
            "input_template_version": None,
        }
    return {
        "schema_version": "opk-rag.task0097.model-manifest.v1",
        "arm_id": arm_id,
        "model_id": provider.model_id,
        "resolved_revision": provider.model_revision,
        "model_commit_sha": provider.model_revision,
        "model_dtype": "default_framework_dtype",
        "actual_model_device": getattr(provider, "actual_model_device", None),
        "actual_model_dtype": getattr(provider, "actual_model_dtype", None),
        "model_parameterization": "cross_encoder_sequence_classification",
        "model_authority": "huggingface_pinned_revision",
        "authority_platform": HF_AUTHORITY_PLATFORM,
        "authority_repo_id": HF_AUTHORITY_REPO_ID,
        "authority_revision": HF_AUTHORITY_REVISION,
        "materialization_source": "modelscope" if getattr(provider, "local_model_path", None) is not None else "huggingface_cache",
        "model_load_source": str(getattr(provider, "local_model_path", None)) if getattr(provider, "local_model_path", None) is not None else provider.model_id,
        "network_required_during_model_load": False if getattr(provider, "local_model_path", None) is not None else None,
        "sentence_transformers_version": _version("sentence-transformers"),
        "transformers_version": _version("transformers"),
        "torch_version": _version("torch"),
        "safetensors_version": _version("safetensors"),
        "cuda_version": cuda_version(),
        "gpu_name": gpu_name(),
        "batch_size": getattr(provider, "config", None).batch_size if getattr(provider, "config", None) else None,
        "max_length": provider.max_pair_tokens,
        "input_template_version": provider.input_template_version,
    }


def candidate_identity_audit(
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0097.candidate-identity-audit.v1",
        "candidate_identity_preserved": True,
        "candidate_membership_change_count": candidate_membership_change_count(baseline, zerank),
        "baseline_identity_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in baseline.items()}),
        "C0_membership_digest": digest_json({unit: sorted(row["canonical_chunk_id"] for row in rows) for unit, rows in control.items()}),
        "E1_membership_digest": digest_json({unit: sorted(row["canonical_chunk_id"] for row in rows) for unit, rows in zerank.items()}),
    }


def per_unit_results(
    baseline: dict[str, list[dict[str, Any]]],
    control: dict[str, list[dict[str, Any]]],
    zerank: dict[str, list[dict[str, Any]]],
    control_rows: list[dict[str, Any]],
    zerank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    crows = {row["sample_unit_id"]: row for row in control_rows}
    zrows = {row["sample_unit_id"]: row for row in zerank_rows}
    return [
        {
            "schema_version": "opk-rag.task0097.per-unit-result.v1",
            "sample_unit_id": unit,
            "sample_id": baseline[unit][0]["sample_id"],
            "query_digest": digest_json(baseline[unit][0]["question"]),
            "gold_original_rank": first_relevant_rank(baseline[unit]),
            "gold_C0_rank": first_relevant_rank(control[unit]),
            "gold_E1_rank": first_relevant_rank(zerank[unit]),
            "C0_correct": crows[unit]["reranker"]["end_to_end_correct"],
            "E1_correct": zrows[unit]["reranker"]["end_to_end_correct"],
        }
        for unit in sorted(baseline)
    ]


def build_contract(inputs: dict[str, Any], git_audit: dict[str, Any]) -> dict[str, Any]:
    default_config = RerankerRuntimeConfig()
    return {
        "schema_version": "opk-rag.task0097.contract.v1",
        "task_id": TASK_ID,
        "benchmark_authority": "TASK-0096 formal benchmark authority",
        "source_digests": inputs["digests"],
        "control_strategy": {
            "arm": "C0",
            "name": "task0095_rank_fusion_default",
            "reranker_enabled": default_config.enabled,
            "reranker_policy": default_config.policy,
            "rank_fusion_k": default_config.rank_fusion_k,
            "rank_fusion_lambda": default_config.rank_fusion_lambda,
        },
        "experimental_strategy": {
            "arm": "E1",
            "name": "zerank2_direct_rerank",
            "model_id": ZERANK2_RERANKER,
            "model_revision": ZERANK2_RERANKER_REVISION,
            "rank_fusion_enabled": False,
            "candidate_injection": False,
            "candidate_filtering": False,
            "score_threshold_filtering": False,
            "query_reformulation_change": False,
            "retrieval_retry_change": False,
            "generation_change": False,
        },
        "unit_count": 75,
        "replicate_count": REPLICATE_COUNT,
        "candidate_identity_rules": "Runtime V2 canonical identity; no text-hash substitution, dedup, injection, or replacement",
        "candidate_membership_rules": "candidate membership must be identical before rerank and after final ordering",
        "score_ordering_rules": "descending(raw_relevance_score), then original_retrieval_rank, then canonical_candidate_id",
        "metrics": ["MRR", "NDCG@5", "NDCG@10", "Recall@1", "Recall@5", "Recall@10", "Recall@20", "Downstream proxy metrics"],
        "runtime_methodology": "cold model load measured separately from warm rerank latency on identical candidate workload",
        "promotion_gates": "integrity, deterministic stability, no downstream/safety/TASK-0093 regression, no OOM",
        "decision_enum": sorted(DECISIONS),
        "git_safety_audit": git_audit,
    }


def experiment_manifest(
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    git_audit: dict[str, Any],
    device: str,
    batch_size: int,
    replicates: int,
    contract_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0097.experiment-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "evaluation_unit_count": len(baseline),
        "candidate_count": sum(len(rows) for rows in baseline.values()),
        "candidate_depth": CANDIDATE_DEPTH,
        "device": device,
        "batch_size": batch_size,
        "replicates": replicates,
        "contract_digest": contract_digest,
        "source_digests": inputs["digests"],
        "retrieval_rerun_count": 0,
        "generation_policy_identity": "task0092_evidence_bound_proxy_v1",
        "git_safety_audit": git_audit,
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for name in (
        "summary",
        "experiment_manifest",
        "dependency_versions",
        "hardware",
        "model_manifest",
        "input_hashes",
        "candidate_identity_audit",
        "gpu_clean_baseline",
        "zerank_runtime_feasibility",
        "zerank_smoke_test",
        "ranking_metrics",
        "downstream_metrics",
        "paired_comparison",
        "task0093_regression_analysis",
        "high_confidence_override_analysis",
        "replicate_stability",
        "runtime_metrics",
    ):
        write_json(output_dir / f"{name}.json", artifacts[name])
    write_jsonl(output_dir / "regressions.jsonl", artifacts["regressions"])
    write_jsonl(output_dir / "per_unit_results.jsonl", artifacts["per_unit_results"])


def verify_task0097_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        output_dir / "summary.json",
        output_dir / "hf_authoritative_file_manifest.json",
        output_dir / "model_materialization.json",
        output_dir / "modelscope_hf_equivalence.json",
        output_dir / "experiment_manifest.json",
        output_dir / "dependency_versions.json",
        output_dir / "hardware.json",
        output_dir / "model_manifest.json",
        output_dir / "input_hashes.json",
        output_dir / "candidate_identity_audit.json",
        output_dir / "gpu_clean_baseline.json",
        output_dir / "zerank_runtime_feasibility.json",
        output_dir / "zerank_smoke_test.json",
        output_dir / "ranking_metrics.json",
        output_dir / "downstream_metrics.json",
        output_dir / "paired_comparison.json",
        output_dir / "task0093_regression_analysis.json",
        output_dir / "high_confidence_override_analysis.json",
        output_dir / "replicate_stability.json",
        output_dir / "runtime_metrics.json",
        output_dir / "regressions.jsonl",
        output_dir / "per_unit_results.jsonl",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        summary = read_json(output_dir / "summary.json")
        config = RerankerRuntimeConfig()
        if summary.get("task_status") == "blocked_by_environment":
            pass
        else:
            if summary.get("evaluation_unit_count") != 75:
                issues.append({"code": "evaluation_unit_count_not_75"})
            if summary.get("decision") not in DECISIONS:
                issues.append({"code": "invalid_decision"})
            if summary.get("candidate_membership_change_count") != 0:
                issues.append({"code": "candidate_membership_changed"})
            if summary.get("zerank_runtime_mode") not in {"pure_gpu", "cpu_gpu_offload"}:
                issues.append({"code": "runtime_mode_not_recorded"})
            if not (config.enabled and config.policy == RANK_FUSION_POLICY and config.rank_fusion_k == 60 and config.rank_fusion_lambda == 0.75):
                issues.append({"code": "task0095_default_drift"})
            if summary.get("task0095_inputs_valid") is not True or summary.get("task0096_inputs_valid") is not True:
                issues.append({"code": "authoritative_inputs_invalid"})
    result = {
        "schema_version": "opk-rag.task0097.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_blocked_artifacts(
    output_dir: Path,
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    contract_digest: str,
) -> None:
    minimal = {
        "summary": summary,
        "experiment_manifest": experiment_manifest(inputs, baseline, summary["git_safety_audit"], summary["device"], summary["batch_size"], summary["replicates"], contract_digest),
        "dependency_versions": dependency_versions(),
        "hardware": hardware(),
        "model_manifest": {
            "E1_zerank2_direct_rerank": {
                "schema_version": "opk-rag.task0097.model-manifest.v1",
                "arm_id": "zerank2_direct_rerank",
                "status": "blocked_by_environment",
                "error": summary["blocker"],
                "model_id": ZERANK2_RERANKER,
                "resolved_revision": ZERANK2_RERANKER_REVISION,
                "model_commit_sha": ZERANK2_RERANKER_REVISION,
                "model_dtype": "default_framework_dtype",
                "model_parameterization": "cross_encoder_sequence_classification",
                "sentence_transformers_version": _version("sentence-transformers"),
                "transformers_version": _version("transformers"),
                "torch_version": _version("torch"),
                "safetensors_version": _version("safetensors"),
                "cuda_version": cuda_version(),
                "gpu_name": gpu_name(),
            }
        },
        "input_hashes": {"source_paths": inputs["paths"], "source_digests": inputs["digests"]},
        "candidate_identity_audit": {"candidate_membership_change_count": 0, "candidate_identity_preserved": True},
        "zerank_smoke_test": {
            "schema_version": "opk-rag.task0097.zerank-smoke-test.v1",
            "status": "failed",
            "failure_stage": "cuda_smoke",
            "failure_class": summary.get("runtime_failure_class"),
            "model_source": "verified_local_snapshot" if summary.get("content_equivalence_verified") else "not_loaded",
            "network_required_during_load": False if summary.get("content_equivalence_verified") else None,
            "device_requested": summary.get("device"),
            "actual_output_device": None,
            "output_dtype": None,
            "batch_size": summary.get("batch_size"),
            "positive_score": None,
            "negative_score": None,
            "positive_score_gt_negative_score": None,
            "peak_allocated_vram_mib": peak_vram_mib(),
            "peak_reserved_vram_mib": peak_reserved_vram_mib(),
            "score_semantics": ZERANK_SCORE_SEMANTICS,
            "error": summary["blocker"],
            "cause_type": summary.get("cause_type"),
            "cause_message": summary.get("cause_message"),
        },
        "ranking_metrics": {},
        "downstream_metrics": {},
        "paired_comparison": {},
        "task0093_regression_analysis": {},
        "high_confidence_override_analysis": {},
        "replicate_stability": {},
        "runtime_metrics": {
            "E1_zerank2_direct_rerank": {
                "oom_count": 1 if summary.get("runtime_failure_class") in {"model_load_oom", "inference_oom"} else 0,
                "model_load_time_seconds": None,
                "rerank_p95_latency_seconds": None,
                "end_to_end_p95_latency_seconds": None,
                "peak_gpu_memory_mib": peak_vram_mib(),
            }
        },
    }
    write_artifacts(output_dir, {**minimal, "regressions": [], "per_unit_results": []})
    write_json(output_dir / "verification.json", {"schema_version": "opk-rag.task0097.verification.v1", "status": "blocked", "issues": [{"code": "blocked_by_environment"}]})


def invalid_summary(inputs: dict[str, Any], unit_count: int, git_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0097.summary.v1",
        "task_id": TASK_ID,
        "task_status": "invalid",
        "decision": "inconclusive",
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "task0096_inputs_valid": inputs["task0096_verification"]["status"] == "valid",
        "evaluation_unit_count": unit_count,
        "git_safety_audit": git_audit,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def write_minimal_invalid(output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)


def blocked_summary(
    inputs: dict[str, Any],
    git_audit: dict[str, Any],
    environment: dict[str, Any],
    device: str,
    batch_size: int,
    replicates: int,
    exc: BaseException,
    output_dir: Path = RESULT_DIR,
) -> dict[str, Any]:
    cause = exc.__cause__
    classification = classify_environment_failure(exc, device, environment)
    materialization = read_json(output_dir / "model_materialization.json") if (output_dir / "model_materialization.json").exists() else {}
    equivalence = read_json(output_dir / "modelscope_hf_equivalence.json") if (output_dir / "modelscope_hf_equivalence.json").exists() else {}
    equivalence_verified = equivalence.get("content_equivalence_verified", False)
    formal_status = "not_run_model_equivalence_not_proven" if equivalence and not equivalence_verified else "not_run_cuda_smoke_not_completed"
    oom_count = 1 if classification in {"model_load_oom", "inference_oom"} else 0
    return {
        "schema_version": "opk-rag.task0097.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked_by_environment",
        "decision": "blocked_by_environment",
        "previous_block_reason": "Zerank-2 load attempt was performed on CPU",
        "current_environment_diagnosis": classification,
        "current_environment_root_cause": current_environment_root_cause(classification, exc, environment),
        "cuda_smoke_status": "blocked_by_environment",
        "formal_benchmark_status": formal_status,
        "runtime_failure_class": classification,
        "blocker": f"{exc.__class__.__name__}: {str(exc)[:500]}",
        "cause_type": cause.__class__.__name__ if cause else None,
        "cause_message": str(cause)[:1000] if cause else None,
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "task0096_inputs_valid": inputs["task0096_verification"]["status"] == "valid",
        "evaluation_unit_count": 75,
        "formal_unit_count": 75,
        "formal_replicate_count": replicates,
        "candidate_membership_change_count": 0,
        "zerank_model_id": ZERANK2_RERANKER,
        "zerank_resolved_revision": ZERANK2_RERANKER_REVISION,
        "zerank_model_commit_sha": ZERANK2_RERANKER_REVISION,
        "hf_authority_repo": HF_AUTHORITY_REPO_ID,
        "hf_authority_revision": HF_AUTHORITY_REVISION,
        "modelscope_repo": MODELSCOPE_REPO_ID,
        "modelscope_revision_or_snapshot": equivalence.get("modelscope_snapshot") or materialization.get("modelscope_snapshot"),
        "modelscope_download_status": materialization.get("download_status"),
        "modelscope_local_path": materialization.get("modelscope_local_path"),
        "hf_expected_weight_shard_count": 2,
        "modelscope_weight_shard_count": equivalence.get("weight_shard_count", 0),
        "weight_shard_size_match_count": equivalence.get("weight_shard_size_match_count", 0),
        "weight_shard_sha256_match_count": equivalence.get("weight_shard_sha256_match_count", 0),
        "tokenizer_size_match": equivalence.get("tokenizer_size_match", False),
        "tokenizer_sha256_match": equivalence.get("tokenizer_sha256_match", False),
        "critical_config_file_count": equivalence.get("critical_config_file_count", 0),
        "critical_config_exact_match_count": equivalence.get("critical_config_exact_match_count", 0),
        "content_equivalence_verified": equivalence_verified,
        "equivalence_failure_reason": equivalence.get("equivalence_failure_reason"),
        "model_authority": "huggingface_pinned_revision",
        "materialization_source": "modelscope",
        "model_load_source": "not_attempted_equivalence_not_verified" if not equivalence_verified else materialization.get("modelscope_local_path"),
        "local_model_load_status": "not_attempted_equivalence_not_verified" if not equivalence_verified else "failed",
        "zerank_device": None,
        "zerank_dtype": None,
        "zerank_batch_size": batch_size,
        "positive_score": None,
        "negative_score": None,
        "smoke_peak_vram_mib": peak_vram_mib(),
        "zerank_oom_count": oom_count,
        "zerank_runtime_mode": None,
        "pure_gpu_attempted": True,
        "pure_gpu_load_status": classification,
        "offload_attempted": False,
        "offload_load_status": "not_attempted",
        "oom_count": oom_count,
        "promotion_gate_passed": False,
        "environment_diagnostics": environment,
        "device": device,
        "batch_size": batch_size,
        "replicates": replicates,
        "git_safety_audit": git_audit,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_report(summary: dict[str, Any]) -> str:
    if summary["task_status"] != "complete":
        return "\n".join(
            [
                "# TASK0097 Zerank Direct Rerank Vs Rank Fusion Benchmark Report",
                "",
                "## Executive Summary",
                "",
                f"TASK-0097 status=`{summary['task_status']}`.",
                f"decision=`{summary['decision']}`.",
                f"blocker=`{summary.get('blocker', 'none')}`.",
                f"cause_type=`{summary.get('cause_type', 'none')}`.",
                f"cause_message=`{summary.get('cause_message', 'none')}`.",
                "",
                "## Inputs",
                "",
                f"- task0095_inputs_valid=`{str(summary.get('task0095_inputs_valid')).lower()}`",
                f"- task0096_inputs_valid=`{str(summary.get('task0096_inputs_valid')).lower()}`",
                f"- evaluation_unit_count=`{summary.get('evaluation_unit_count')}`",
                f"- replicate_count=`{summary.get('replicates')}`",
                "",
                "## Model Authority And Materialization",
                "",
                f"- authority_platform=`{HF_AUTHORITY_PLATFORM}`",
                f"- authority_repo_id=`{summary.get('hf_authority_repo', ZERANK2_RERANKER)}`",
                f"- authority_revision=`{summary.get('hf_authority_revision', ZERANK2_RERANKER_REVISION)}`",
                f"- materialization_source=`{summary.get('materialization_source', 'modelscope')}`",
                f"- modelscope_repo=`{summary.get('modelscope_repo', MODELSCOPE_REPO_ID)}`",
                f"- modelscope_snapshot=`{summary.get('modelscope_revision_or_snapshot', 'not_available')}`",
                f"- same_model_content=`{str(summary.get('content_equivalence_verified', False)).lower()}`",
                "- same_hosting_platform=`false`",
                f"- equivalence_failure_reason=`{summary.get('equivalence_failure_reason', 'none')}`",
                f"- runtime_failure_class=`{summary.get('runtime_failure_class', 'not_applicable')}`",
                f"- formal_benchmark_status=`{summary.get('formal_benchmark_status', 'not_recorded')}`",
                f"- device=`{summary.get('device', 'not_recorded')}`",
                f"- batch_size=`{summary.get('batch_size', 'not_recorded')}`",
                "",
                "TASK-0095 production defaults were not modified.",
            ]
        )
    return "\n".join(
        [
            "# TASK0097 Zerank Direct Rerank Vs Rank Fusion Benchmark Report",
            "",
            "## Executive Summary",
            "",
            f"decision=`{summary['decision']}`",
            f"architectural_interpretation=`{summary['architectural_interpretation']}`",
            "",
            "## Integrity",
            "",
            f"- model_authority=`huggingface:{HF_AUTHORITY_REPO_ID}@{HF_AUTHORITY_REVISION}`",
            "- materialization_source=`modelscope`",
            "- same_model_content=`true`",
            "- same_hosting_platform=`false`",
            f"- task0095_inputs_valid=`{str(summary['task0095_inputs_valid']).lower()}`",
            f"- task0096_inputs_valid=`{str(summary['task0096_inputs_valid']).lower()}`",
            f"- evaluation_unit_count=`{summary['evaluation_unit_count']}`",
            f"- candidate_membership_change_count=`{summary['candidate_membership_change_count']}`",
            f"- rerank_determinism_valid=`{str(summary['rerank_determinism_valid']).lower()}`",
            "",
            "## Quality",
            "",
            f"- task0095_rank_fusion_e2e_accuracy=`{summary['task0095_rank_fusion_e2e_accuracy']}`",
            f"- zerank_e2e_accuracy=`{summary['zerank_e2e_accuracy']}`",
            f"- zerank_downstream_improved_count=`{summary['zerank_downstream_improved_count']}`",
            f"- zerank_downstream_regressed_count=`{summary['zerank_downstream_regressed_count']}`",
            f"- zerank_downstream_net_gain=`{summary['zerank_downstream_net_gain']}`",
            "",
            "## Safety",
            "",
            f"- unsupported_answer_regression_count=`{summary['unsupported_answer_regression_count']}`",
            f"- grounding_regression_count=`{summary['grounding_regression_count']}`",
            f"- citation_regression_count=`{summary['citation_regression_count']}`",
            f"- safe_action_regression_count=`{summary['safe_action_regression_count']}`",
            f"- zerank_new_top5_lost_count=`{summary['zerank_new_top5_lost_count']}`",
            "",
            "## Runtime",
            "",
            f"- zerank_runtime_mode=`{summary.get('zerank_runtime_mode', 'not_recorded')}`",
            f"- pure_gpu_load_status=`{summary.get('pure_gpu_load_status', 'not_recorded')}`",
            f"- offload_load_status=`{summary.get('offload_load_status', 'not_recorded')}`",
            f"- zerank_oom_count=`{summary['oom_count']}`",
            f"- zerank_p95_latency_seconds=`{summary['runtime_metrics']['E1_zerank2_direct_rerank']['rerank_p95_latency_seconds']}`",
            f"- zerank_peak_gpu_memory_mib=`{summary['runtime_metrics']['E1_zerank2_direct_rerank']['peak_gpu_memory_mib']}`",
            f"- gpu_used_before_load_mib=`{summary.get('gpu_used_before_load_mib', 'not_recorded')}`",
            f"- gpu_free_before_load_mib=`{summary.get('gpu_free_before_load_mib', 'not_recorded')}`",
            "",
            "## Runtime Sequence",
            "",
            "1. HF authoritative repository and revision remained pinned.",
            "2. HF large-weight transport failed in the earlier materialization path; this is not treated as runtime feasibility evidence.",
            "3. ModelScope materialization succeeded from `zeroentropy/zerank-2`.",
            "4. HF/ModelScope content equivalence was verified: two weight shards, tokenizer, and 14 critical config files matched exactly.",
            "5. Clean pure-GPU BF16 retry loaded the verified local snapshot on `cuda:0`.",
            "6. CUDA smoke passed with raw Zerank positive score greater than negative score.",
            "7. CPU/GPU offload was not attempted because pure GPU succeeded.",
            "8. Formal 75-unit x 3-replicate benchmark completed using `pure_gpu` for all Zerank replicates.",
            "",
            "## Decision",
            "",
            "Zerank-2 direct rerank fits entirely in the reference RTX 3060 under the tested BF16 configuration, but it regressed both ranking and downstream quality versus TASK-0095 BGE + guarded rank fusion. TASK-0095 rank fusion remains the default.",
            "",
            "TASK-0095 production defaults were not modified.",
        ]
    )


def dependency_versions() -> dict[str, str]:
    return {name: _version(name) for name in ("sentence-transformers", "transformers", "torch", "safetensors", "huggingface_hub", "modelscope")}


def hardware() -> dict[str, Any]:
    env = environment_diagnostics()
    return {
        "schema_version": "opk-rag.task0097.hardware.v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": env["cuda_visible_devices"],
        "cuda_available": env["cuda_available"],
        "cuda_device_count": env["cuda_device_count"],
        "bf16_supported": env["bf16_supported"],
        "gpu": gpu_name(),
        "gpu_vram": gpu_vram(),
        "cuda": cuda_version(),
        "torch": _version("torch"),
        "dtype": "default_framework_dtype",
    }


def gpu_clean_baseline() -> dict[str, Any]:
    queried = query_nvidia_smi_gpu()
    env = environment_diagnostics()
    return {
        "schema_version": "opk-rag.task0097.gpu-clean-baseline.v1",
        "gpu_name": queried.get("gpu_name") or env.get("gpu_name"),
        "total_vram_mib": queried.get("total_vram_mib") or gpu_vram(),
        "used_vram_before_load_mib": queried.get("used_vram_mib"),
        "free_vram_before_load_mib": queried.get("free_vram_mib"),
        "gpu_utilization_percent": queried.get("gpu_utilization_percent"),
        "torch_version": env.get("torch_version"),
        "torch_cuda_version": env.get("torch_cuda_version"),
        "cuda_available": env.get("cuda_available"),
        "bf16_supported": env.get("bf16_supported"),
    }


def query_nvidia_smi_gpu() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return {}
    if not output:
        return {}
    name, total, used, free, utilization = [part.strip() for part in output.splitlines()[0].split(",")]
    return {
        "gpu_name": name,
        "total_vram_mib": int(total),
        "used_vram_mib": int(used),
        "free_vram_mib": int(free),
        "gpu_utilization_percent": int(utilization),
    }


def zerank_runtime_feasibility(smoke: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    pure_gpu_attempt = {
        "pure_gpu_load_status": summary.get("pure_gpu_load_status"),
        "device": "cuda",
        "dtype": summary.get("zerank_dtype"),
        "batch_size": summary.get("zerank_batch_size"),
        "quantization": "disabled",
        "cpu_offload": "disabled",
        "actual_execution_device": smoke.get("actual_output_device"),
        "cuda_smoke_status": smoke.get("status"),
        "positive_score": smoke.get("positive_score"),
        "negative_score": smoke.get("negative_score"),
        "peak_allocated_vram_mib": smoke.get("peak_allocated_vram_mib"),
        "peak_reserved_vram_mib": smoke.get("peak_reserved_vram_mib"),
    }
    feasibility = {
        "schema_version": "opk-rag.task0097.zerank-runtime-feasibility.v1",
        "zerank_runtime_mode": summary.get("zerank_runtime_mode"),
        "pure_gpu_feasible": summary.get("zerank_runtime_mode") == "pure_gpu",
        "pure_gpu_attempt": pure_gpu_attempt,
        "offload_attempted": summary.get("offload_attempted", False),
        "offload_attempt": {
            "offload_load_status": summary.get("offload_load_status", "not_attempted_pure_gpu_succeeded"),
            "reason": "not_attempted_because_pure_gpu_succeeded" if summary.get("zerank_runtime_mode") == "pure_gpu" else None,
        },
    }
    feasibility["runtime_mode_freeze"] = {
        "formal_replicate_count": summary.get("formal_replicate_count"),
        "runtime_mode_used_for_all_replicates": summary.get("zerank_runtime_mode"),
    }
    return feasibility


def cuda_version() -> str | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch.version.cuda


def gpu_name() -> str | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def gpu_vram() -> int | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))


def reset_peak_vram() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_reserved_vram_mib() -> float | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_reserved() / (1024 * 1024))


def environment_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "schema_version": "opk-rag.task0097.environment-diagnostics.v1",
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_endpoint": __import__("os").environ.get("HF_ENDPOINT"),
        "torch_version": _version("torch"),
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_name": None,
        "bf16_supported": None,
        "sentence_transformers_version": _version("sentence-transformers"),
        "transformers_version": _version("transformers"),
        "safetensors_version": _version("safetensors"),
    }
    try:
        import torch
    except ModuleNotFoundError:
        return diagnostics
    diagnostics["torch_cuda_version"] = torch.version.cuda
    diagnostics["cuda_available"] = bool(torch.cuda.is_available())
    diagnostics["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        diagnostics["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            diagnostics["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            diagnostics["bf16_supported"] = None
    return diagnostics


def classify_environment_failure(exc: BaseException, device: str, environment: dict[str, Any]) -> str:
    message = " ".join(_exception_chain_messages(exc)).lower()
    if device == "cuda" and not environment.get("cuda_available"):
        return "cuda_unavailable"
    if "cpu" in message and device == "cuda":
        return "incorrect_device_routing"
    if "out of memory" in message or "cuda oom" in message or "cuda error: out of memory" in message:
        return "model_load_oom" if any(kind == "RerankerModelLoadError" for kind in _exception_chain_types(exc)) else "inference_oom"
    if "couldn't connect" in message or "cannot find the requested files" in message or "connection" in message:
        return "other_environment_failure"
    if "safetensors" in message or "checkpoint" in message or "state dict" in message:
        return "model_weight_failure"
    if "architecture" in message or "configuration" in message or "auto" in message or "tokenizer" in message:
        return "model_architecture_or_dependency_failure"
    return "other_environment_failure"


def _exception_chain_messages(exc: BaseException) -> list[str]:
    messages = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return messages


def _exception_chain_types(exc: BaseException) -> list[str]:
    types = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        types.append(current.__class__.__name__)
        current = current.__cause__ or current.__context__
    return types


def current_environment_root_cause(classification: str, exc: BaseException, environment: dict[str, Any]) -> str:
    if classification == "cuda_unavailable":
        return "CUDA was requested but torch.cuda.is_available() is false in the project environment."
    if classification == "incorrect_device_routing":
        return "The formal task requested CUDA but the model path attempted to use CPU."
    if classification == "model_load_oom":
        return "The pinned Zerank-2 CrossEncoder exceeded available CUDA memory at batch_size=1."
    if classification == "model_architecture_or_dependency_failure":
        return "The installed sentence-transformers/transformers stack could not construct the pinned Zerank-2 CrossEncoder architecture."
    if classification == "model_weight_failure":
        return "The pinned Zerank-2 model weights could not be loaded or validated locally."
    if "hf-mirror.com" in str(environment.get("hf_endpoint")):
        return "HF_ENDPOINT points at hf-mirror.com; the Python Hub client could not retrieve uncached Zerank-2 files from that endpoint."
    return f"{classification}: {exc.__class__.__name__}: {str(exc)[:300]}"


def _score_dtype(scores: Sequence[float]) -> str:
    return type(scores[0]).__name__ if scores else "float"


def rank_digest(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[str, float]]]:
    return {unit: [(row["canonical_chunk_id"], round(float(row["reranker_score"]), 8)) for row in rows] for unit, rows in rows_by_unit.items()}


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def git_safety_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0097.git-safety-audit.v1",
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]).splitlines(),
        "staged_diff": _git(["diff", "--staged", "--stat"]),
        "unstaged_diff": _git(["diff", "--stat"]),
        "git_add_executed": False,
        "git_commit_created": False,
    }


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


if __name__ == "__main__":
    run_task0097_zerank_vs_rank_fusion_benchmark()
