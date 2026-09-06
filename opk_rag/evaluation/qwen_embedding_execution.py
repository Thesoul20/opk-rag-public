from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import resource
import tempfile
import time
from typing import Any, Callable, Sequence

from opk_rag.embedding.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_REVISION
from opk_rag.embedding.provider import EmbeddingInferenceError
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.evaluation.evidence_identity import digest_json


EXECUTION_PLAN_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_qwen_representation_execution_plan_v1.json"
EXECUTION_RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_qwen_representation_execution_v1.json"
PRIVATE_TASK_PATH = ROOT / ".private" / "evaluation" / "task0057"
BATCH_SIZE_LADDER = (16, 8, 4, 2, 1)
STORAGE_DTYPE = "float32"
VECTOR_BYTES = DEFAULT_EMBEDDING_DIMENSION * 4


@dataclass(frozen=True)
class MaterializedEmbedding:
    variant_id: str
    vectors: dict[str, list[float]]
    token_counts: list[int]
    manifest: dict[str, Any]


def build_execution_plan(
    *,
    contract: dict[str, Any],
    candidate_universe: dict[str, Any],
    model_execution: dict[str, Any],
    selected_batch_size: int | None = None,
    max_batch_token_budget: int | None = None,
) -> dict[str, Any]:
    batch_size = int(selected_batch_size or os.environ.get("OPK_RAG_TASK0057_BATCH_SIZE", "1"))
    token_budget = int(max_batch_token_budget or os.environ.get("OPK_RAG_TASK0057_MAX_BATCH_TOKENS", "8192"))
    plan = {
        "schema_version": "opk-rag.qwen-representation-execution-plan.v1",
        "plan_id": "phase2-qwen-representation-execution-plan-v1",
        "experiment_id": "phase2-chunk-representation-experiment-v1",
        "candidate_universe_chunk_count": candidate_universe.get("candidate_universe_chunk_count"),
        "candidate_universe_digest": candidate_universe.get("candidate_universe_digest"),
        "experiment_contract_digest": digest_json(contract) if contract else None,
        "model": DEFAULT_EMBEDDING_MODEL_NAME,
        "model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "dimension": DEFAULT_EMBEDDING_DIMENSION,
        "device_policy": {
            "requested_device": model_execution.get("device"),
            "priority": ["cuda_float32_streaming", "cuda_bfloat16_compute_float32_storage", "cuda_float16_compute_float32_storage", "cpu_float32_streaming"],
            "no_remote_embedding_api": True,
        },
        "precision_policy": {
            "compute_dtype": model_execution.get("dtype") or "float32",
            "storage_dtype": STORAGE_DTYPE,
            "low_precision_requires_parity_gate": True,
            "quantization_allowed": False,
        },
        "batch_policy": {
            "batch_size_ladder": list(BATCH_SIZE_LADDER),
            "selected_batch_size": batch_size,
            "max_batch_token_budget": token_budget,
            "maximum_sequence_length": 8192,
            "padding_policy": "provider_default_dynamic_padding_per_batch",
            "truncation_policy": "reject_without_silent_truncation",
        },
        "cache_policy": {
            "private_cache_path": ".private/evaluation/task0057/",
            "storage_dtype": STORAGE_DTYPE,
            "atomic_manifest_write": True,
            "old_task0056_35_vector_cache_reuse": "rejected",
        },
        "resume_policy": {
            "resume_completed_vectors": True,
            "completed_manifest_status_required": "completed",
            "partial_manifest_status": "partial",
            "missing_vectors_fail_formal_run": True,
        },
        "parity_policy": {
            "reference": "cuda_float32_batch_1_when_low_precision_is_used",
            "mean_cosine_min": 0.9990,
            "minimum_cosine_min": 0.9950,
            "top20_candidate_identity_agreement": 1.0,
            "scope_match_metric_agreement": 1.0,
            "status": "not_required_for_float32" if (model_execution.get("dtype") in {None, "float32", "unknown"}) else "required",
        },
        "failure_policy": {
            "no_surrogate_fallback": True,
            "no_zero_vector_fill": True,
            "incomplete_variant_invalidates_experiment": True,
            "single_batch_oom_may_decrease_batch_once": True,
        },
        "bindings": {
            "query_contract_digest": (contract.get("bindings") or {}).get("query_contract_digest"),
            "representation_template_digests": (contract.get("bindings") or {}).get("variant_digests"),
            "query_template_digests": (contract.get("bindings") or {}).get("query_variant_digests"),
            "tokenizer_digest": model_execution.get("tokenizer_digest"),
            "canonical_identity_schema": (contract.get("bindings") or {}).get("canonical_evidence_identity_schema"),
            "output_normalization_contract": "l2_normalized_cosine",
            "similarity_metric": "cosine",
        },
    }
    write_json(EXECUTION_PLAN_PATH, plan)
    private_path = PRIVATE_TASK_PATH / "execution-plan.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(private_path, plan)
    return plan


def profile_qwen_execution(
    *,
    provider: QwenLocalEmbeddingProvider,
    short_input: str,
    long_input: str,
    previous_batch_inputs: Sequence[str],
) -> dict[str, Any]:
    import torch

    profile: dict[str, Any] = {
        "schema_version": "opk-rag.qwen-execution-profile.v1",
        "profile_id": "phase2-qwen-representation-execution-profile-v1",
        "gpu_name": None,
        "gpu_total_memory": None,
        "gpu_free_memory_before_model": None,
        "model_parameter_count": None,
        "model_loaded_memory": None,
        "tokenizer_memory": None,
        "batch_input_memory": None,
        "forward_peak_allocated": None,
        "forward_peak_reserved": None,
        "pooling_peak_memory": None,
        "normalization_peak_memory": None,
        "cache_write_memory": None,
        "host_memory_peak": _host_memory_bytes(),
        "diagnostic_cases": [],
        "oom_stage": None,
        "oom_classification": "not_observed",
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
    }
    if provider.device == "cuda" and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        profile["gpu_name"] = torch.cuda.get_device_name(device_index)
        free, total = torch.cuda.mem_get_info(device_index)
        profile["gpu_total_memory"] = int(total)
        profile["gpu_free_memory_before_model"] = int(free)
        torch.cuda.reset_peak_memory_stats()
    load_started = _cuda_allocated(provider.device)
    model = provider._model
    profile["model_parameter_count"] = _parameter_count(model)
    profile["model_loaded_memory"] = _cuda_allocated(provider.device) - load_started if load_started is not None else None
    profile["tokenizer_memory"] = None
    cases = [
        ("model_load", []),
        ("single_short_input", [short_input]),
        ("single_long_input", [long_input]),
        ("small_batch", list(previous_batch_inputs[: min(4, len(previous_batch_inputs))])),
        ("full_previous_batch_path", list(previous_batch_inputs[: max(1, min(len(previous_batch_inputs), provider.config.batch_size))])),
    ]
    for case_id, texts in cases:
        before = _cuda_allocated(provider.device)
        started = time.perf_counter()
        try:
            if texts:
                provider.embed_documents(texts)
            status = "pass"
            error = None
        except EmbeddingInferenceError as exc:
            status = "oom" if _is_oom_message(str(exc)) else "failed"
            error = str(exc)
            if status == "oom" and profile["oom_stage"] is None:
                profile["oom_stage"] = _classify_oom_stage(case_id)
                profile["oom_classification"] = _classify_oom_stage(case_id)
        case = {
            "case_id": case_id,
            "input_count": len(texts),
            "token_count": sum(_safe_count_tokens(provider, text) for text in texts),
            "status": status,
            "error": error,
            "wall_time": time.perf_counter() - started,
            "cuda_allocated_delta": (_cuda_allocated(provider.device) - before) if before is not None else None,
            "host_memory_peak": _host_memory_bytes(),
        }
        profile["diagnostic_cases"].append(case)
        if provider.device == "cuda" and torch.cuda.is_available():
            profile["forward_peak_allocated"] = int(torch.cuda.max_memory_allocated())
            profile["forward_peak_reserved"] = int(torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
    profile["pooling_peak_memory"] = profile["forward_peak_allocated"]
    profile["normalization_peak_memory"] = profile["forward_peak_allocated"]
    profile["batch_input_memory"] = max((row.get("token_count") or 0 for row in profile["diagnostic_cases"]), default=0)
    profile["cache_write_memory"] = VECTOR_BYTES
    profile["host_memory_peak"] = _host_memory_bytes()
    return profile


def select_safe_batch_plan(provider: QwenLocalEmbeddingProvider, probe_inputs: Sequence[str]) -> dict[str, Any]:
    selected = None
    attempts = []
    for batch_size in BATCH_SIZE_LADDER:
        batch = list(probe_inputs[: min(batch_size, len(probe_inputs))])
        if not batch:
            continue
        try:
            provider.embed_documents(batch)
            attempts.append({"batch_size": batch_size, "status": "pass", "token_count": sum(_safe_count_tokens(provider, text) for text in batch)})
            selected = batch_size
            break
        except EmbeddingInferenceError as exc:
            attempts.append({"batch_size": batch_size, "status": "oom" if _is_oom_message(str(exc)) else "failed", "error": str(exc)})
            if not _is_oom_message(str(exc)):
                raise
    if selected is None:
        selected = 1
    return {
        "batch_size_ladder": list(BATCH_SIZE_LADDER),
        "selected_batch_size": selected,
        "attempts": attempts,
        "max_batch_token_budget": max((sum(_safe_count_tokens(provider, text) for text in probe_inputs[:selected]), 1)),
    }


def materialize_embedding_variant(
    *,
    variant_id: str,
    kind: str,
    identities: Sequence[str],
    texts: Sequence[str],
    provider: QwenLocalEmbeddingProvider,
    execution_plan: dict[str, Any],
    cache_key_rows: Sequence[dict[str, Any]],
    expected_count: int,
    count_tokens: Callable[[str], int] | None = None,
) -> MaterializedEmbedding:
    if len(identities) != len(texts) or len(texts) != len(cache_key_rows):
        raise ValueError("identities, texts, and cache_key_rows must have the same length")
    variant_dir = PRIVATE_TASK_PATH / "embeddings" / variant_id
    variant_dir.mkdir(parents=True, exist_ok=True)
    vector_path = variant_dir / "vectors.float32"
    manifest_path = variant_dir / "manifest.json"
    token_counts = [(count_tokens or (lambda text: _safe_count_tokens(provider, text)))(text) for text in texts]
    existing = _load_manifest(manifest_path)
    cache_contract_digest = digest_json(
        {
            "execution_plan_digest": digest_json(execution_plan),
            "variant_id": variant_id,
            "kind": kind,
            "identities": list(identities),
            "rendered_input_digests": [digest_json(text) for text in texts],
            "cache_key_rows": list(cache_key_rows),
        }
    )
    if _manifest_completed(existing, cache_contract_digest, expected_count, vector_path):
        vectors = _read_memmap(vector_path, expected_count)
        return MaterializedEmbedding(variant_id=variant_id, vectors=dict(zip(identities, vectors, strict=True)), token_counts=token_counts, manifest=existing)
    completed = set(existing.get("completed_identities") or []) if existing.get("cache_contract_digest") == cache_contract_digest else set()
    memmap = _open_memmap(vector_path, expected_count)
    batch_size = int((execution_plan.get("batch_policy") or {}).get("selected_batch_size") or 1)
    started = time.perf_counter()
    failed_batches = 0
    forward_batches = 0
    for start in range(0, len(texts), batch_size):
        batch_identities = list(identities[start : start + batch_size])
        pending_offsets = [offset for offset, identity in enumerate(batch_identities, start=start) if identity not in completed]
        if not pending_offsets:
            continue
        batch_texts = [texts[offset] for offset in pending_offsets]
        try:
            vectors = provider.embed_documents(batch_texts)
        except EmbeddingInferenceError as exc:
            if batch_size > 1 and _is_oom_message(str(exc)):
                failed_batches += 1
                batch_size = max(1, batch_size // 2)
                execution_plan.setdefault("batch_policy", {})["selected_batch_size"] = batch_size
                return materialize_embedding_variant(
                    variant_id=variant_id,
                    kind=kind,
                    identities=identities,
                    texts=texts,
                    provider=provider,
                    execution_plan=execution_plan,
                    cache_key_rows=cache_key_rows,
                    expected_count=expected_count,
                    count_tokens=count_tokens,
                )
            raise
        forward_batches += 1
        for offset, identity, vector in zip(pending_offsets, (identities[offset] for offset in pending_offsets), vectors, strict=True):
            validated = _validate_vector(vector)
            memmap[offset, :] = validated
            completed.add(identity)
        memmap.flush()
        _atomic_write_json(
            manifest_path,
            _build_manifest(
                variant_id=variant_id,
                kind=kind,
                expected_count=expected_count,
                completed=completed,
                token_counts=token_counts,
                cache_contract_digest=cache_contract_digest,
                execution_plan=execution_plan,
                vector_path=vector_path,
                status="partial" if len(completed) < expected_count else "completed",
                started=started,
                forward_batches=forward_batches,
                failed_batches=failed_batches,
            ),
        )
    manifest = _build_manifest(
        variant_id=variant_id,
        kind=kind,
        expected_count=expected_count,
        completed=completed,
        token_counts=token_counts,
        cache_contract_digest=cache_contract_digest,
        execution_plan=execution_plan,
        vector_path=vector_path,
        status="completed" if len(completed) == expected_count else "invalid",
        started=started,
        forward_batches=forward_batches,
        failed_batches=failed_batches,
    )
    _atomic_write_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"Embedding materialization incomplete for {variant_id}: {len(completed)} != {expected_count}")
    vectors = _read_memmap(vector_path, expected_count)
    return MaterializedEmbedding(variant_id=variant_id, vectors=dict(zip(identities, vectors, strict=True)), token_counts=token_counts, manifest=manifest)


def build_cache_key_row(*, execution_plan: dict[str, Any], identity: str | None, rendered_text: str, variant_id: str | None, query_variant_id: str | None, template_digest: str | None) -> dict[str, Any]:
    payload = {
        "execution_plan_digest": digest_json(execution_plan),
        "experiment_contract_digest": execution_plan.get("experiment_contract_digest"),
        "candidate_universe_digest": execution_plan.get("candidate_universe_digest"),
        "embedding_model_id": execution_plan.get("model"),
        "model_revision": execution_plan.get("model_revision"),
        "tokenizer_digest": (execution_plan.get("bindings") or {}).get("tokenizer_digest"),
        "compute_dtype": (execution_plan.get("precision_policy") or {}).get("compute_dtype"),
        "storage_dtype": (execution_plan.get("precision_policy") or {}).get("storage_dtype"),
        "representation_variant_id": variant_id,
        "query_variant_id": query_variant_id,
        "representation_template_digest": template_digest if variant_id else None,
        "query_template_digest": template_digest if query_variant_id else None,
        "canonical_identity_digest": identity,
        "rendered_input_digest": digest_json(rendered_text),
        "max_sequence_length": (execution_plan.get("batch_policy") or {}).get("maximum_sequence_length"),
        "truncation_policy": (execution_plan.get("batch_policy") or {}).get("truncation_policy"),
        "normalization_contract": (execution_plan.get("bindings") or {}).get("output_normalization_contract"),
    }
    return {"cache_key": digest_json(payload), **payload}


def write_execution_result(payload: dict[str, Any]) -> None:
    write_json(EXECUTION_RESULT_PATH, payload)


def vector_digest(vectors: dict[str, list[float]]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(vectors):
        digest.update(identity.encode("utf-8"))
        for value in vectors[identity]:
            digest.update(f"{float(value):.9g},".encode("ascii"))
    return digest.hexdigest()


def _open_memmap(path: Path, expected_count: int):
    import numpy as np

    return np.memmap(path, dtype=np.float32, mode="w+" if not path.exists() else "r+", shape=(expected_count, DEFAULT_EMBEDDING_DIMENSION))


def _read_memmap(path: Path, expected_count: int) -> list[list[float]]:
    import numpy as np

    array = np.memmap(path, dtype=np.float32, mode="r", shape=(expected_count, DEFAULT_EMBEDDING_DIMENSION))
    return array.astype(np.float32).tolist()


def _validate_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != DEFAULT_EMBEDDING_DIMENSION:
        raise RuntimeError(f"Qwen embedding dimension mismatch: {len(vector)} != {DEFAULT_EMBEDDING_DIMENSION}")
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Qwen embedding contains non-finite values")
    norm = sum(value * value for value in values) ** 0.5
    if abs(norm - 1.0) > 1e-3:
        raise RuntimeError(f"Qwen embedding is not L2 normalized: norm={norm:.6f}")
    return values


def _build_manifest(
    *,
    variant_id: str,
    kind: str,
    expected_count: int,
    completed: set[str],
    token_counts: list[int],
    cache_contract_digest: str,
    execution_plan: dict[str, Any],
    vector_path: Path,
    status: str,
    started: float,
    forward_batches: int,
    failed_batches: int,
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.qwen-embedding-manifest.v1",
        "variant_id": variant_id,
        "kind": kind,
        "expected_count": expected_count,
        "completed_count": len(completed),
        "batch_count": forward_batches,
        "failed_batch_count": failed_batches,
        "vector_digest": _file_sha256(vector_path) if vector_path.exists() else None,
        "identity_digest": digest_json(sorted(completed)),
        "cache_contract_digest": cache_contract_digest,
        "execution_plan_digest": digest_json(execution_plan),
        "status": status,
        "completed_identities": sorted(completed),
        "storage_dtype": STORAGE_DTYPE,
        "dimension": DEFAULT_EMBEDDING_DIMENSION,
        "l2_normalized": True,
        "expected_input_count": expected_count,
        "rendered_input_count": expected_count,
        "embedded_vector_count": len(completed),
        "stored_vector_count": len(completed),
        "unique_identity_count": len(completed),
        "missing_vector_count": max(0, expected_count - len(completed)),
        "invalid_dimension_count": 0,
        "non_finite_vector_count": 0,
        "non_normalized_vector_count": 0,
        "embedding_input_token_mean": sum(token_counts) / len(token_counts) if token_counts else 0.0,
        "embedding_input_token_p95": _p95(token_counts),
        "embedding_wall_time": time.perf_counter() - started,
        "embedding_throughput_inputs_per_second": (len(completed) / (time.perf_counter() - started)) if time.perf_counter() > started else 0.0,
        "peak_gpu_allocated_bytes": _cuda_peak("allocated"),
        "peak_gpu_reserved_bytes": _cuda_peak("reserved"),
        "peak_host_memory_bytes": _host_memory_bytes(),
        "raw_vector_payload_bytes": expected_count * VECTOR_BYTES,
        "vector_path_private": vector_path.relative_to(ROOT).as_posix(),
    }


def _manifest_completed(manifest: dict[str, Any], cache_contract_digest: str, expected_count: int, vector_path: Path) -> bool:
    return (
        manifest.get("status") == "completed"
        and manifest.get("cache_contract_digest") == cache_contract_digest
        and manifest.get("completed_count") == expected_count
        and manifest.get("stored_vector_count") == expected_count
        and vector_path.exists()
        and vector_path.stat().st_size == expected_count * VECTOR_BYTES
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_count_tokens(provider: QwenLocalEmbeddingProvider, text: str) -> int:
    try:
        return int(provider.count_tokens(text))
    except Exception:
        return max(1, len(text) // 4)


def _p95(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def _cuda_allocated(device: str) -> int | None:
    if device != "cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


def _cuda_peak(kind: str) -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if kind == "reserved":
            return int(torch.cuda.max_memory_reserved())
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _parameter_count(model: Any) -> int | None:
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def _host_memory_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)


def _is_oom_message(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "oom" in lowered or "ran out of memory" in lowered


def _classify_oom_stage(case_id: str) -> str:
    if case_id == "model_load":
        return "model_load"
    if case_id in {"single_short_input", "single_long_input", "small_batch", "full_previous_batch_path"}:
        return "forward_or_batch_padding"
    return "unknown"
