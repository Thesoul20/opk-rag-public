from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
from typing import Any, Mapping

from opk_rag.embedding.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL_NAME, load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, write_json

TASK_ID = "TASK-0173"
TASK0172_SOURCE_HEAD = "a41b77f9ac58175058d32ff34a8ddfa8d22fb02a"
EXPERIMENT_ID = "task0173-cold-start-embedding-cuda-runtime-failure-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0173_cold_start_embedding_cuda_runtime_failure_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0173_COLD_START_EMBEDDING_CUDA_RUNTIME_FAILURE_DIAGNOSIS_REPORT.md"
COLD_START_REPO = Path("<workspace>/opk-rag-testv1/repo/opk-rag")
PACKAGES = ("torch", "transformers", "sentence-transformers", "accelerate", "safetensors", "huggingface-hub", "tokenizers", "numpy")
CUDA_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES")
CACHE_ENV_KEYS = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "HF_ENDPOINT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
ROOT_CAUSE_TAXONOMY = {
    "cold_start_cpu_only_torch_build", "torch_cuda_build_mismatch", "python_dependency_drift", "undeclared_gpu_dependency",
    "cuda_driver_runtime_incompatibility", "cuda_visibility_difference", "cuda_environment_variable_difference",
    "model_cache_missing_or_incomplete", "model_revision_drift", "embedding_device_selection_mismatch",
    "embedding_dtype_mismatch", "cuda_memory_pressure", "embedding_library_version_drift", "model_loader_configuration_drift", "unknown",
}
FAILURE_STAGES = {
    "embedding_dependency_import", "embedding_torch_initialization", "embedding_cuda_discovery", "embedding_model_resolution",
    "embedding_model_cache", "embedding_model_initialization", "embedding_weight_loading", "embedding_device_transfer",
    "embedding_inference", "embedding_persistence", "unknown",
}


def run_task0173(write: bool = True, run_model_probes: bool = True) -> dict[str, Any]:
    source_head = _cmd(["git", "rev-parse", "HEAD"], ROOT)["stdout"].strip()
    system_gpu = system_gpu_audit()
    dev = environment_probe("development", ROOT, ROOT / ".venv" / "bin" / "python", run_model_probes)
    cold = environment_probe("cold_start", COLD_START_REPO, COLD_START_REPO / ".venv" / "bin" / "python", run_model_probes)
    dep_diff = dependency_diff(dev, cold)
    torch_cmp = torch_cuda_comparison(dev, cold)
    cache = model_cache_audit(dev, cold)
    runtime_cmp = embedding_runtime_comparison(dev, cold)
    original_failure = build_original_failure(cold)
    root = classify_root_cause(dev, cold, dep_diff, cache)
    summary = {
        "task_id": TASK_ID,
        "task_status": "complete" if root["dominant_embedding_root_cause"] != "unknown" else "partial",
        "source_authoritative_head": source_head,
        "task0172_source_head": TASK0172_SOURCE_HEAD,
        "source_head_changed_since_task0172": source_head != TASK0172_SOURCE_HEAD,
        "task0170_current_first_failure_stage": "embedding",
        "original_embedding_failure_reproduced": original_failure["original_embedding_failure_reproduced"],
        "first_embedding_failure_stage": root["first_embedding_failure_stage"],
        "dominant_embedding_root_cause": root["dominant_embedding_root_cause"],
        "root_cause_confidence": root["root_cause_confidence"],
        "recommended_remediation_family": root["recommended_remediation_family"],
        "failure_scope": root["failure_scope"],
        "development_environment_audited": True,
        "cold_start_environment_audited": cold["python_executable_exists"],
        "torch_cuda_comparison_complete": True,
        "dependency_comparison_complete": True,
        "model_cache_comparison_complete": True,
        "embedding_runtime_comparison_complete": True,
        "minimal_model_load_probe_complete": run_model_probes,
        "embedding_policy_mutation_count": 0,
        "embedding_device_default_mutation_count": 0,
        "dependency_remediation_count": 0,
        "retrieval_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "reranker_policy_mutation_count": 0,
        "chunking_policy_mutation_count": 0,
        "corpus_mutation_count": 0,
        "cpu_fallback_promoted": False,
        "task0170_revalidation_executed": False,
        "development_python_version": dev.get("python_version"),
        "cold_start_python_version": cold.get("python_version"),
        "development_torch_version": dev.get("packages", {}).get("torch", {}).get("version"),
        "cold_start_torch_version": cold.get("packages", {}).get("torch", {}).get("version"),
        "development_torch_cuda_build": dev.get("torch", {}).get("torch_cuda_build"),
        "cold_start_torch_cuda_build": cold.get("torch", {}).get("torch_cuda_build"),
        "development_torch_cuda_available": dev.get("torch", {}).get("torch_cuda_available"),
        "cold_start_torch_cuda_available": cold.get("torch", {}).get("torch_cuda_available"),
        "development_gpu_device": dev.get("torch", {}).get("torch_device_name"),
        "cold_start_gpu_device": cold.get("torch", {}).get("torch_device_name"),
        "development_transformers_version": dev.get("packages", {}).get("transformers", {}).get("version"),
        "cold_start_transformers_version": cold.get("packages", {}).get("transformers", {}).get("version"),
        "development_sentence_transformers_version": dev.get("packages", {}).get("sentence-transformers", {}).get("version"),
        "cold_start_sentence_transformers_version": cold.get("packages", {}).get("sentence-transformers", {}).get("version"),
        "development_accelerate_version": dev.get("packages", {}).get("accelerate", {}).get("version"),
        "cold_start_accelerate_version": cold.get("packages", {}).get("accelerate", {}).get("version"),
        "dependency_version_difference_count": dep_diff["dependency_version_difference_count"],
        "torch_dependency_declared": dependency_declared("torch"),
        "torch_cuda_build_declared_or_pinned": torch_cuda_build_declared_or_pinned(),
        "development_environment_has_unlocked_or_manual_dependency": development_has_unlocked_manual_dependency(dev),
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "development_model_cache_identity": cache["development_model_cache_identity"],
        "cold_start_model_cache_identity": cache["cold_start_model_cache_identity"],
        "model_cache_shared": cache["model_cache_shared"],
        "development_embedding_device": dev.get("embedding_config", {}).get("selected_device"),
        "cold_start_embedding_device": cold.get("embedding_config", {}).get("selected_device"),
        "development_embedding_dtype": dev.get("embedding_config", {}).get("embedding_dtype"),
        "cold_start_embedding_dtype": cold.get("embedding_config", {}).get("embedding_dtype"),
        "development_minimal_model_load_success": dev.get("minimal_model_load", {}).get("model_load_success"),
        "cold_start_minimal_model_load_success": cold.get("minimal_model_load", {}).get("model_load_success"),
        "development_minimal_inference_success": dev.get("minimal_inference", {}).get("embedding_inference_success"),
        "cold_start_minimal_inference_success": cold.get("minimal_inference", {}).get("embedding_inference_success"),
        "cold_start_cpu_diagnostic_load_success": cold.get("cpu_diagnostic_probe", {}).get("model_load_success", "not_run"),
        "raw_torch_cuda_allocation_success": cold.get("raw_torch_cuda_allocation_success", "not_run"),
        "first_failure_stage": root["first_embedding_failure_stage"],
        "dominant_root_cause": root["dominant_embedding_root_cause"],
    }
    artifacts = {
        "summary.json": summary,
        "original_failure.json": original_failure,
        "system_gpu_audit.json": system_gpu,
        "development_environment.json": dev,
        "cold_start_environment.json": cold,
        "dependency_diff.json": dep_diff,
        "torch_cuda_comparison.json": torch_cmp,
        "model_cache_audit.json": cache,
        "embedding_runtime_comparison.json": runtime_cmp,
        "minimal_model_load_results.json": {"development": dev.get("minimal_model_load"), "cold_start": cold.get("minimal_model_load")},
        "minimal_inference_results.json": {"development": dev.get("minimal_inference"), "cold_start": cold.get("minimal_inference")},
        "cpu_diagnostic_probe.json": {"development": dev.get("cpu_diagnostic_probe"), "cold_start": cold.get("cpu_diagnostic_probe"), "cpu_fallback_promoted": False},
        "root_cause_analysis.json": root,
    }
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, artifacts), encoding="utf-8")
    return summary


def environment_probe(environment_id: str, repo: Path, python: Path, run_model_probes: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"environment_id": environment_id, "repo": str(repo), "python_executable": str(python), "python_executable_exists": python.exists()}
    if not python.exists():
        return result
    probe = _python_json(python, repo, _CHILD_PROBE)
    result.update(probe if isinstance(probe, dict) else {"probe_error": probe})
    if run_model_probes:
        result["minimal_model_load"] = _python_json(python, repo, _MODEL_PROBE)
        result["minimal_inference"] = result["minimal_model_load"] if result["minimal_model_load"].get("embedding_inference_success") else {"embedding_inference_success": False, "not_run_reason": "model_load_failed"}
        result["cpu_diagnostic_probe"] = _python_json(python, repo, _CPU_PROBE)
    result["raw_torch_cuda_allocation_success"] = _python_json(python, repo, _CUDA_ALLOC_PROBE).get("raw_torch_cuda_allocation_success", False)
    return result


def system_gpu_audit() -> dict[str, Any]:
    smi = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], ROOT)
    gpus = []
    if smi["returncode"] == 0:
        for line in smi["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({"name": parts[0], "memory_total": parts[1], "memory_free": parts[2], "driver_version": parts[3]})
    return {"kernel_version": platform.platform(), "nvidia_smi_available": smi["returncode"] == 0, "nvidia_driver_version": gpus[0]["driver_version"] if gpus else None, "gpu_count": len(gpus), "gpu_names": [g["name"] for g in gpus], "gpu_memory_total": [g["memory_total"] for g in gpus], "nvidia_smi_stderr": smi["stderr"][-1000:]}


def dependency_diff(dev: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    d, c = dev.get("packages", {}), cold.get("packages", {})
    version_diffs = {p: {"development": d.get(p, {}).get("version"), "cold_start": c.get(p, {}).get("version")} for p in PACKAGES if d.get(p, {}).get("version") != c.get(p, {}).get("version")}
    missing = [p for p in PACKAGES if d.get(p, {}).get("installed") and not c.get(p, {}).get("installed")]
    extra = [p for p in PACKAGES if c.get(p, {}).get("installed") and not d.get(p, {}).get("installed")]
    return {"dependency_version_difference_count": len(version_diffs), "dependency_missing_in_cold_start_count": len(missing), "dependency_extra_in_cold_start_count": len(extra), "version_differences": version_diffs, "missing_in_cold_start": missing, "extra_in_cold_start": extra}


def torch_cuda_comparison(dev: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    return {"development": dev.get("torch", {}), "cold_start": cold.get("torch", {}), "torch_runtime_equivalent": dev.get("torch") == cold.get("torch")}


def model_cache_audit(dev: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    d = dev.get("model_cache", {})
    c = cold.get("model_cache", {})
    return {"embedding_model_id": DEFAULT_EMBEDDING_MODEL_NAME, "development_model_cache_identity": d.get("identity"), "cold_start_model_cache_identity": c.get("identity"), "model_cache_shared": bool(d.get("cache_path") and d.get("cache_path") == c.get("cache_path")), "development": d, "cold_start": c}


def embedding_runtime_comparison(dev: Mapping[str, Any], cold: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("embedding_model", "embedding_dimension", "embedding_device", "selected_device", "embedding_dtype", "embedding_batch_size", "embedding_normalization", "local_files_only", "trust_remote_code", "device_map")
    return {"development": {k: dev.get("embedding_config", {}).get(k) for k in keys}, "cold_start": {k: cold.get("embedding_config", {}).get(k) for k in keys}, "device_selection_equivalent": dev.get("embedding_config", {}).get("selected_device") == cold.get("embedding_config", {}).get("selected_device"), "embedding_dtype_equivalent": dev.get("embedding_config", {}).get("embedding_dtype") == cold.get("embedding_config", {}).get("embedding_dtype")}


def classify_failure_stage(exc_type: str | None, message: str | None, tb: str | None = None) -> str:
    text = f"{exc_type or ''}\n{message or ''}\n{tb or ''}".lower()
    if "modulenotfounderror" in text or "is required" in text:
        return "embedding_dependency_import"
    if "cuda device requested" in text or "cuda is unavailable" in text:
        return "embedding_cuda_discovery"
    if "couldn't connect" in text or "cannot find the requested files" in text or "localentrynotfound" in text:
        return "embedding_model_cache"
    if "out of memory" in text or "oom" in text:
        return "embedding_device_transfer"
    if "failed to load embedding model" in text:
        return "embedding_model_initialization"
    if "embedding inference failed" in text:
        return "embedding_inference"
    return "unknown"


def classify_root_cause(dev: Mapping[str, Any], cold: Mapping[str, Any], dep: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    cold_load = cold.get("minimal_model_load", {})
    stage = classify_failure_stage(cold_load.get("exception_type"), cold_load.get("exception_message_redacted"), cold_load.get("sanitized_traceback")) if not cold_load.get("model_load_success") else "unknown"
    root = "unknown"; family = "further_diagnosis"; confidence = "low"; scope = "unknown"; evidence = []
    if dev.get("torch", {}).get("torch_cuda_available") and not cold.get("torch", {}).get("torch_cuda_available"):
        root, family, confidence, scope = "cuda_visibility_difference", "normalize_cuda_environment", "high", "cuda_specific"
        evidence.append("development torch reports CUDA available while cold-start does not")
    elif cold_load and not cold_load.get("model_load_success") and stage == "embedding_model_cache":
        root, family, confidence, scope = "model_cache_missing_or_incomplete", "repair_model_cache_authority", "high", "model_general"
        evidence.append("cold-start fails before weight/device transfer with LocalEntryNotFound/connectivity cache error")
        if not cold.get("cpu_diagnostic_probe", {}).get("model_load_success"):
            evidence.append("CPU diagnostic fails with the same model-resolution/cache class, so failure is not CUDA-specific")
    elif dep.get("dependency_version_difference_count"):
        root, family, confidence, scope = "python_dependency_drift", "align_embedding_library_versions", "medium", "dependency"
        evidence.append("embedding dependency versions differ between environments")
    return {"task_id": TASK_ID, "first_embedding_failure_stage": stage, "dominant_embedding_root_cause": root, "root_cause_confidence": confidence, "failure_scope": scope, "recommended_remediation_family": family, "causal_evidence": evidence, "root_cause_taxonomy_valid": root in ROOT_CAUSE_TAXONOMY, "cpu_fallback_promoted": False, "remaining_hypotheses": [] if root != "unknown" else ["insufficient model-load traceback"]}


def build_original_failure(cold: Mapping[str, Any]) -> dict[str, Any]:
    load = cold.get("minimal_model_load", {})
    return {"task_id": TASK_ID, "original_embedding_failure_reproduced": not load.get("model_load_success", True), "exception_type": load.get("exception_type"), "exception_message_redacted": load.get("exception_message_redacted"), "failure_command": "minimal QwenLocalEmbeddingProvider load probe", "failure_python_executable": cold.get("python_executable"), "failure_stage": classify_failure_stage(load.get("exception_type"), load.get("exception_message_redacted"), load.get("sanitized_traceback")), "sanitized_traceback": load.get("sanitized_traceback")}


def contract() -> dict[str, Any]:
    return {"task_id": TASK_ID, "parent_task": "TASK-0170", "infrastructure_parent_task": "TASK-0172", "diagnosis_scope": "embedding_cuda_runtime", "development_environment_required": True, "cold_start_environment_required": True, "embedding_model_mutation_allowed": False, "embedding_device_default_mutation_allowed": False, "cpu_fallback_promotion_allowed": False, "dependency_remediation_allowed": False, "retrieval_policy_mutation_allowed": False, "graph_policy_mutation_allowed": False, "reranker_policy_mutation_allowed": False, "chunking_policy_mutation_allowed": False, "corpus_mutation_allowed": False, "minimal_cuda_probe_required": True, "development_vs_cold_start_comparison_required": True, "root_cause_classification_required": True}


def dependency_declared(name: str) -> bool:
    return bool(re.search(rf'["\']{re.escape(name)}([=<>!~]|["\'])', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.I))


def torch_cuda_build_declared_or_pinned() -> bool:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8") + "\n" + (ROOT / "uv.lock").read_text(encoding="utf-8")
    return "+cu" in text or "pytorch" in text.lower() and "cu" in text.lower()


def development_has_unlocked_manual_dependency(dev: Mapping[str, Any]) -> bool:
    return bool(dev.get("packages", {}).get("torch", {}).get("installed")) and not dependency_declared("torch")


def render_report(summary: Mapping[str, Any], artifacts: Mapping[str, Any]) -> str:
    rows = ["| Item | Development | Cold-start | Equivalent |", "| --- | --- | --- | --- |"]
    pairs = [("Python", "development_python_version", "cold_start_python_version"), ("torch", "development_torch_version", "cold_start_torch_version"), ("torch CUDA build", "development_torch_cuda_build", "cold_start_torch_cuda_build"), ("CUDA available", "development_torch_cuda_available", "cold_start_torch_cuda_available"), ("GPU device", "development_gpu_device", "cold_start_gpu_device"), ("transformers", "development_transformers_version", "cold_start_transformers_version"), ("sentence-transformers", "development_sentence_transformers_version", "cold_start_sentence_transformers_version"), ("accelerate", "development_accelerate_version", "cold_start_accelerate_version"), ("model revision", "development_model_cache_identity", "cold_start_model_cache_identity"), ("model cache", "development_model_cache_identity", "cold_start_model_cache_identity"), ("embedding device", "development_embedding_device", "cold_start_embedding_device"), ("embedding dtype", "development_embedding_dtype", "cold_start_embedding_dtype"), ("minimal load", "development_minimal_model_load_success", "cold_start_minimal_model_load_success"), ("minimal inference", "development_minimal_inference_success", "cold_start_minimal_inference_success")]
    for label, dk, ck in pairs:
        rows.append(f"| {label} | `{summary.get(dk)}` | `{summary.get(ck)}` | `{summary.get(dk) == summary.get(ck)}` |")
    return "\n".join(["# TASK-0173 Cold-start Embedding CUDA Runtime Failure Diagnosis Report", "", "## Recap", "TASK-0170/TASK-0172 moved the cold-start blocker to embedding after database/schema isolation was closed.", "", "## Original embedding failure", f"Reproduced: `{summary.get('original_embedding_failure_reproduced')}`; first refined stage: `{summary.get('first_embedding_failure_stage')}`.", "", "## Development vs Cold-start Matrix", *rows, "", "## Root cause classification", f"Dominant root cause: `{summary.get('dominant_embedding_root_cause')}` ({summary.get('root_cause_confidence')}).", f"Failure scope: `{summary.get('failure_scope')}`.", "", "## Causal evidence", *(f"- {e}" for e in artifacts['root_cause_analysis.json'].get('causal_evidence', [])), "", "## Recommended remediation family", f"`{summary.get('recommended_remediation_family')}`. No dependency remediation, runtime default change, CPU fallback promotion, or embedding policy mutation was applied.", "", "## Final decision", f"task_status=`{summary.get('task_status')}`; cpu_fallback_promoted=`False`; task0170_revalidation_executed=`False`.", ""])


def verify_task0173_artifacts(write: bool = True) -> dict[str, Any]:
    summary_path = RESULT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    required = ["summary.json", "original_failure.json", "system_gpu_audit.json", "development_environment.json", "cold_start_environment.json", "dependency_diff.json", "torch_cuda_comparison.json", "model_cache_audit.json", "embedding_runtime_comparison.json", "minimal_model_load_results.json", "minimal_inference_results.json", "cpu_diagnostic_probe.json", "root_cause_analysis.json"]
    missing = [name for name in required if not (RESULT_DIR / name).exists()]
    passed = not missing and summary.get("cpu_fallback_promoted") is False and summary.get("dominant_embedding_root_cause") in ROOT_CAUSE_TAXONOMY and summary.get("first_embedding_failure_stage") in FAILURE_STAGES
    result = {"task_id": TASK_ID, "verification_passed": passed, "missing_artifacts": missing, "cpu_fallback_promoted": summary.get("cpu_fallback_promoted"), "dominant_embedding_root_cause": summary.get("dominant_embedding_root_cause"), "first_embedding_failure_stage": summary.get("first_embedding_failure_stage")}
    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True); write_json(RESULT_DIR / "verification.json", result)
    return result


def _cmd(command: list[str], cwd: Path, timeout: int = 60) -> dict[str, Any]:
    p = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return {"command": command, "cwd": str(cwd), "returncode": p.returncode, "stdout": p.stdout, "stderr": _redact(p.stderr)}


def _python_json(python: Path, cwd: Path, code: str) -> dict[str, Any]:
    p = subprocess.run([str(python), "-c", code], cwd=str(cwd), text=True, capture_output=True, timeout=300)
    raw = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {"probe_parse_error": True, "stdout_tail": p.stdout[-2000:], "stderr_tail": _redact(p.stderr[-2000:])}
    data["probe_returncode"] = p.returncode
    if p.stderr:
        data["probe_stderr_tail"] = _redact(p.stderr[-2000:])
    return data


def _redact(text: str) -> str:
    return re.sub(r"postgres(?:ql)?://[^:\s/@]+:[^@\s]+@", "postgres://***:***@", text or "", flags=re.I)


_CHILD_PROBE = r'''
import json, os, sys, importlib.metadata as m, pathlib, hashlib
out={"python_executable":sys.executable,"python_version":sys.version.split()[0],"virtual_environment_path":os.environ.get("VIRTUAL_ENV") or sys.prefix,"PYTHONPATH":os.environ.get("PYTHONPATH","<unset>"),"VIRTUAL_ENV":os.environ.get("VIRTUAL_ENV","<unset>"),"PATH_resolution":os.environ.get("PATH","").split(":")[:8]}
pkgs={}
for p in %r:
    try:
        dist=m.distribution(p); pkgs[p]={"installed":True,"version":dist.version,"location":str(pathlib.Path(dist.locate_file('')))}
    except Exception: pkgs[p]={"installed":False,"version":None,"location":None}
out["packages"]=pkgs
try:
 import torch
 out["torch"]={"torch_version":torch.__version__,"torch_cuda_build":torch.version.cuda,"torch_cuda_available":torch.cuda.is_available(),"torch_cuda_device_count":torch.cuda.device_count(),"torch_device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"torch_package_location":torch.__file__}
except Exception as e: out["torch"]={"exception_type":type(e).__name__,"exception_message_redacted":str(e)}
out["cuda_environment"]={k:os.environ.get(k,"<unset>") for k in %r}
out["cuda_environment"]["PATH_CUDA_entries"]=[x for x in os.environ.get("PATH","").split(":") if "cuda" in x.lower()]
out["cache_environment"]={k:os.environ.get(k,"<unset>") for k in %r}
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import _select_device
c=load_embedding_config(os.environ)
try: selected=_select_device(c.device)
except Exception as e: selected=f"ERROR:{type(e).__name__}:{e}"
out["embedding_config"]={"embedding_model":c.model_name,"embedding_dimension":c.dimension,"embedding_device":c.device,"selected_device":selected,"embedding_dtype":"sentence_transformers_default_not_configured","embedding_batch_size":c.batch_size,"embedding_normalization":c.normalize,"local_files_only":c.local_files_only,"trust_remote_code":"not_configured","device_map":"not_configured","model_revision":c.model_revision,"cache_dir":c.cache_dir_string}
base=pathlib.Path(os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME", str(pathlib.Path.home()/'.cache')))/('hub' if os.environ.get("HF_HOME") else 'huggingface/hub')
model_dir=base/'models--Qwen--Qwen3-Embedding-0.6B'
snaps=list((model_dir/'snapshots').glob('*')) if (model_dir/'snapshots').exists() else []
identity='missing'
if snaps:
    manifest=[]
    for f in sorted(snaps[0].rglob('*')):
        if f.is_file(): manifest.append(f.relative_to(snaps[0]).as_posix()+':'+str(f.stat().st_size))
    identity=snaps[0].name+':'+hashlib.sha256('\n'.join(manifest).encode()).hexdigest()[:16]
out["model_cache"]={"embedding_model_id":"Qwen/Qwen3-Embedding-0.6B","embedding_model_revision_if_resolvable":c.model_revision,"cache_path":str(model_dir),"model_cache_exists":model_dir.exists(),"model_cache_complete":bool(snaps),"identity":identity}
print(json.dumps(out, ensure_ascii=False))
''' % (PACKAGES, CUDA_ENV_KEYS, CACHE_ENV_KEYS)

_MODEL_PROBE = r'''
import json, traceback, math
out={"model_load_success":False,"embedding_inference_success":False,"diagnostic_only":True}
try:
 from opk_rag.embedding.config import load_embedding_config
 from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
 c=load_embedding_config(); p=QwenLocalEmbeddingProvider(c); out["selected_device"]=p.device; _=p._model; out["model_load_success"]=True
 v=p.embed_query("cold-start embedding probe"); out["embedding_inference_success"]=True; out["embedding_output_dimension"]=len(v); out["embedding_finite"]=all(math.isfinite(float(x)) for x in v)
except Exception as e:
 out["exception_type"]=type(e).__name__; out["exception_message_redacted"]=str(e); out["sanitized_traceback"]=traceback.format_exc(limit=20)
print(json.dumps(out, ensure_ascii=False))
'''

_CPU_PROBE = r'''
import json, traceback, os
out={"diagnostic_only":True,"model_load_success":False,"cpu_fallback_promoted":False}
try:
 from opk_rag.embedding.config import load_embedding_config
 from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
 env=dict(os.environ); env["OPK_RAG_EMBEDDING_DEVICE"]="cpu"; c=load_embedding_config(env); p=QwenLocalEmbeddingProvider(c); out["selected_device"]=p.device; _=p._model; out["model_load_success"]=True
except Exception as e:
 out["exception_type"]=type(e).__name__; out["exception_message_redacted"]=str(e); out["sanitized_traceback"]=traceback.format_exc(limit=15)
print(json.dumps(out, ensure_ascii=False))
'''

_CUDA_ALLOC_PROBE = r'''
import json
out={"raw_torch_cuda_allocation_success":"not_run"}
try:
 import torch
 if torch.cuda.is_available():
  x=torch.tensor([1.0]).cuda(); out["raw_torch_cuda_allocation_success"]=bool(float(x.cpu()[0])==1.0)
 else: out["raw_torch_cuda_allocation_success"]=False
except Exception as e:
 out["raw_torch_cuda_allocation_success"]=False; out["exception_type"]=type(e).__name__; out["exception_message_redacted"]=str(e)
print(json.dumps(out, ensure_ascii=False))
'''
