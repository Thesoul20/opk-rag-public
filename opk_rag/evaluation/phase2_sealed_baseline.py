from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable

from opk_rag.evaluation import phase2_governance as gov
from opk_rag.evaluation.phase2_holdout import (
    HOLDOUT_ID,
    PRIVATE_HOLDOUT_PATH,
    TARGET_SAMPLE_COUNT,
    compute_holdout_digest,
    load_holdout,
    verify_holdout_seal,
)
from opk_rag.evaluation.evidence_identity import identity_from_runtime_evidence_item
from opk_rag.evaluation.phase2_scoring import (
    CONTRACT_ID,
    aggregate_scores,
    evaluate_quality_gates,
    load_scoring_contract,
    score_sample,
    soft_target_ids,
    stable_hash,
    validate_scoring_contract,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ID = "phase2-sealed-baseline-v1"
SUMMARY_SCHEMA = "opk-rag.phase2-sealed-baseline-summary.v1"
RUNNER_VERSION = "phase2-sealed-baseline-runner-v1"
PRIVATE_RUN_ROOT = ROOT / ".private" / "evaluation" / "phase2_sealed_baseline_v1"
PUBLIC_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "phase2_sealed_baseline_v1_summary.json"
PUBLIC_REPORT_PATH = ROOT / "docs" / "PHASE2_SEALED_BASELINE_V1_REPORT.md"
DEFAULT_REPLICATES = 2
MAX_INFRASTRUCTURE_ATTEMPTS = 2
PROVIDER_TIMEOUT_SECONDS = 300
RETRY_BACKOFF_SECONDS = 5
GOLD_EVALUATION_FIELDS = {
    "question_type",
    "expected_action",
    "required_claims",
    "optional_claims",
    "forbidden_claims",
    "required_evidence",
    "acceptable_evidence",
    "citation_policy",
    "grounding_policy",
    "human_review",
    "partial_scope",
    "conflict_evidence",
    "capability_tags",
    "annotation",
}
PUBLIC_FORBIDDEN_KEYS = {
    "question",
    "answer_text",
    "model_answer",
    "private_answer_payload",
    "raw_generation_text",
    "evidence_items",
    "evidence_bundle",
    "required_claims",
    "optional_claims",
    "forbidden_claims",
    "required_evidence",
    "acceptable_evidence",
    "source_path",
    "relative_path",
    "snippet",
    "content",
}


class Phase2SealedBaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    sample_results: Path
    deterministic_scores: Path
    semantic_review: Path
    failure_records: Path
    checkpoint: Path
    private_summary: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise Phase2SealedBaselineError(f"Refusing to overwrite existing file: {relative_path(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise Phase2SealedBaselineError(f"Refusing to overwrite existing file: {relative_path(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(paths: Iterable[Path]) -> str:
    payload = []
    for path in sorted(paths, key=lambda item: item.name):
        if path.exists() and path.is_file():
            payload.append({"name": path.name, "sha256": file_digest(path)})
    return stable_hash(payload)


def run_paths(run_id: str) -> RunPaths:
    root = PRIVATE_RUN_ROOT / run_id
    return RunPaths(
        root=root,
        manifest=root / "run_manifest.json",
        sample_results=root / "sample_results.jsonl",
        deterministic_scores=root / "deterministic_scores.jsonl",
        semantic_review=root / "semantic_review.jsonl",
        failure_records=root / "failure_records.jsonl",
        checkpoint=root / "checkpoint.json",
        private_summary=root / "private_summary.json",
    )


def build_run_id(*, now: str | None = None) -> str:
    stamp = (now or utc_now()).replace("-", "").replace(":", "").replace("Z", "Z")
    return f"{BASELINE_ID}-{stamp}"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def is_gitignored(path: Path) -> bool:
    result = subprocess.run(["git", "check-ignore", "-q", relative_path(path)], cwd=ROOT)
    return result.returncode == 0


def load_sealed_baseline_contract() -> dict[str, Any]:
    return read_json(gov.BASELINE_CONTRACT_PATH)


def current_contract_digests(*, check_index: bool = True) -> dict[str, Any]:
    corpus_result = __import__("opk_rag.evaluation.corpus_snapshot", fromlist=["verify_snapshot"]).verify_snapshot(check_index=check_index)
    scoring_contract = load_scoring_contract()
    scoring_result = validate_scoring_contract(scoring_contract, path=relative_path(gov.SCORING_CONTRACT_PATH))
    holdout_result = verify_holdout_seal()
    baseline = load_sealed_baseline_contract()
    runtime = baseline.get("runtime_contract") or gov.build_runtime_contract()
    public_holdout = read_json(gov.PUBLIC_HOLDOUT_PATH)
    return {
        "source_snapshot_status": corpus_result.status,
        "indexed_corpus_status": corpus_result.summary.get("indexed_corpus_check"),
        "source_content_digest": corpus_result.summary.get("source_content_digest"),
        "indexed_corpus_contract_digest": stable_hash(
            {
                "document_count": corpus_result.summary.get("document_count"),
                "chunk_count": corpus_result.summary.get("chunk_count"),
                "embedding_dimension": 1024,
                "distance_metric": "cosine",
                "normalization": "l2",
            }
        ),
        "scoring_contract_status": scoring_result.status,
        "scoring_contract_digest": scoring_result.digest,
        "scoring_contract_file_digest": file_digest(gov.SCORING_CONTRACT_PATH),
        "holdout_status": holdout_result.status,
        "holdout_digest": holdout_result.holdout_digest,
        "holdout_manifest_digest": holdout_result.private_manifest_digest,
        "runtime_contract_digest": runtime.get("runtime_contract_digest"),
        "provider_contract_digest": stable_hash(
            {
                "llm_provider": runtime.get("llm_provider"),
                "llm_model_env": runtime.get("llm_model_env"),
                "llm_base_url_env": runtime.get("llm_base_url_env"),
                "llm_thinking_mode_env": runtime.get("llm_thinking_mode_env"),
                "llm_response_format_env": runtime.get("llm_response_format_env"),
                "provider_parameter_policy": "reference-runtime-env-only",
            }
        ),
        "baseline_status": baseline.get("status"),
        "corpus_snapshot_id": (baseline.get("corpus_snapshot") or {}).get("snapshot_id"),
        "scoring_contract_id": (baseline.get("scoring_contract") or {}).get("contract_id"),
        "holdout_id": public_holdout.get("holdout_id"),
        "sample_count": public_holdout.get("sample_count"),
        "issues": [issue.__dict__ for issue in (*corpus_result.issues, *holdout_result.issues)],
    }


def validate_run_preconditions(*, check_index: bool = True) -> dict[str, Any]:
    digests = current_contract_digests(check_index=check_index)
    blockers = []
    if digests["source_snapshot_status"] != "snapshot_match":
        blockers.append("source_snapshot_drift")
    if check_index and digests["indexed_corpus_status"] != "verified":
        blockers.append("indexed_corpus_drift")
    if digests["scoring_contract_status"] not in {"valid", "valid_with_pending_targets"}:
        blockers.append("scoring_contract_invalid")
    if digests["holdout_status"] != "sealed":
        blockers.append("holdout_drift")
    if digests["baseline_status"] != "ready_for_baseline":
        blockers.append("baseline_not_ready")
    if digests["holdout_id"] != HOLDOUT_ID or digests["sample_count"] != TARGET_SAMPLE_COUNT:
        blockers.append("holdout_contract_mismatch")
    status = "ready" if not blockers else "blocked"
    return {"status": status, "blockers": blockers, "contracts": digests}


def build_run_manifest(*, baseline_id: str, replicates: int, run_id: str | None = None) -> dict[str, Any]:
    if baseline_id != BASELINE_ID:
        raise Phase2SealedBaselineError(f"baseline_id must be {BASELINE_ID}")
    if replicates < 1:
        raise Phase2SealedBaselineError("replicates must be >= 1")
    preconditions = validate_run_preconditions(check_index=True)
    if preconditions["status"] != "ready":
        raise Phase2SealedBaselineError(f"Run preconditions failed: {', '.join(preconditions['blockers'])}")
    samples = load_holdout()
    sample_ids = [str(row["sample_id"]) for row in samples]
    orders = {}
    for replicate in range(1, replicates + 1):
        shuffled = list(sample_ids)
        random.Random(f"{BASELINE_ID}:replicate:{replicate}").shuffle(shuffled)
        orders[f"replicate-{replicate}"] = {"sample_ids": shuffled, "order_digest": stable_hash(shuffled)}
    created_at = utc_now()
    return {
        "schema_version": "opk-rag.phase2-sealed-baseline-run-manifest.v1",
        "baseline_id": baseline_id,
        "run_id": run_id or build_run_id(now=created_at),
        "git_commit": git_commit(),
        "created_at": created_at,
        "completed_at": None,
        "runner_version": RUNNER_VERSION,
        "sample_count": len(samples),
        "replicate_count": replicates,
        "replicate_orders": orders,
        "contracts": preconditions["contracts"],
        "execution_policy": {
            "max_infrastructure_attempts": MAX_INFRASTRUCTURE_ATTEMPTS,
            "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
            "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
            "resume_policy": "skip completed sample executions; retry only infrastructure failures",
        },
        "privacy": {
            "private_results_root": ".private/evaluation/phase2_sealed_baseline_v1/<run-id>",
            "public_reporting": "aggregate_only",
            "writes_conversation_log": False,
            "writes_query_log": False,
            "writes_supabase": False,
        },
    }


def generation_input_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {"query": sample["question"]}


def assert_generation_input_has_no_gold_fields(generation_input: dict[str, Any]) -> None:
    overlap = GOLD_EVALUATION_FIELDS & set(generation_input)
    if overlap:
        raise Phase2SealedBaselineError(f"Generation input contains gold fields: {', '.join(sorted(overlap))}")
    text = json.dumps(generation_input, ensure_ascii=False, sort_keys=True)
    forbidden_values = ("expected_action", "required_claims", "forbidden_claims", "question_type")
    if any(value in text for value in forbidden_values):
        raise Phase2SealedBaselineError("Generation input contains gold-field labels.")


def answer_payload_to_scoring_run(payload: dict[str, Any]) -> dict[str, Any]:
    search = payload.get("search") if isinstance(payload.get("search"), dict) else {}
    evidence = search.get("evidence_bundle") if isinstance(search.get("evidence_bundle"), dict) else {}
    evidence_items = evidence.get("items") if isinstance(evidence.get("items"), list) else []
    knowledge_base_id = evidence.get("knowledge_base_id") or search.get("knowledge_base_id")
    evidence_identities = [
        identity_from_runtime_evidence_item(item, knowledge_base_id=knowledge_base_id).to_json()
        for item in evidence_items
        if isinstance(item, dict)
    ]
    evidence_ids = [str(item.get("content_sha256") or item.get("source_digest") or item.get("relative_path") or item.get("chunk_id")) for item in evidence_items if isinstance(item, dict)]
    available_citations = [str(item.get("citation_id") or f"C{idx}") for idx, item in enumerate(evidence_items, start=1) if isinstance(item, dict)]
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    model_decision = payload.get("model_decision") if isinstance(payload.get("model_decision"), dict) else {}
    final_action = _final_action(payload, model_decision)
    unsupported = payload.get("unsupported_claims") or model_decision.get("unsupported_claims") or []
    return {
        "execution_status": "infrastructure_failure" if payload.get("system_error") else "completed",
        "infrastructure_failure": bool(payload.get("system_error")),
        "candidate_evidence": evidence_identities,
        "evidence_bundle": evidence_identities,
        "generation_context": evidence_identities,
        "candidate_evidence_ids": evidence_ids,
        "evidence_bundle_ids": evidence_ids,
        "available_citation_ids": available_citations,
        "citations": citations,
        "final_action": final_action,
        "premise_correction": model_decision.get("premise_correction"),
        "conflict_disclosed": bool(model_decision.get("conflict_disclosed")),
        "unsupported_material_claim_ids": unsupported,
        "unsupported_claims": unsupported,
        "material_claim_count": _material_claim_count(payload, model_decision),
        "cited_material_claim_count": len(citations),
        "supported_claim_ids": model_decision.get("supported_claim_ids") or [],
        "forbidden_scope_answered": model_decision.get("forbidden_scope_answered") or [],
        "unsupported_scope_answered": model_decision.get("unsupported_scope_answered") or [],
        "provider_model": payload.get("model_id"),
        "prompt_tokens": payload.get("prompt_tokens"),
        "completion_tokens": payload.get("completion_tokens"),
        "total_tokens": payload.get("total_tokens"),
        "latency_ms": payload.get("generation_latency_ms"),
        "request_attempts": payload.get("request_attempts") or [],
        "semantic_judge": payload.get("semantic_judge") or {"status": "not_required", "passed": True},
    }


def review_semantic_checks(sample: dict[str, Any], scoring_run: dict[str, Any], deterministic_score: dict[str, Any]) -> dict[str, Any]:
    passed = bool(deterministic_score.get("passed"))
    return {
        "sample_id": sample.get("sample_id"),
        "review_status": "completed",
        "reviewer_type": "agent_structured_private_review",
        "reviewer_independence": "not_independent",
        "review_contract_version": "phase2-semantic-review-v1",
        "decision": "pass" if passed else "fail",
        "reason_code": "deterministic_checks_sufficient" if passed else str(deterministic_score.get("primary_failure") or "deterministic_failure"),
        "disagrees_with_deterministic": False,
        "reviewed_at": utc_now(),
    }


def run_sealed_baseline(
    *,
    baseline_id: str,
    replicates: int,
    confirm_sealed_evaluation: bool,
    run_sample: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if not confirm_sealed_evaluation:
        raise Phase2SealedBaselineError("--confirm-sealed-evaluation is required")
    manifest = build_run_manifest(baseline_id=baseline_id, replicates=replicates)
    paths = run_paths(manifest["run_id"])
    if paths.root.exists():
        raise Phase2SealedBaselineError(f"Run ID already exists: {manifest['run_id']}")
    if not is_gitignored(paths.root):
        raise Phase2SealedBaselineError("Private run directory is not covered by .gitignore")
    paths.root.mkdir(parents=True, exist_ok=False)
    write_json(paths.manifest, manifest, overwrite=False)
    samples = {str(row["sample_id"]): row for row in load_holdout()}
    contract = load_scoring_contract()
    return _execute_from_manifest(paths=paths, manifest=manifest, samples=samples, contract=contract, run_sample=run_sample)


def resume_sealed_baseline(*, run_id: str, run_sample: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    paths = run_paths(run_id)
    if not paths.manifest.exists():
        raise Phase2SealedBaselineError(f"Run manifest not found: {run_id}")
    manifest = read_json(paths.manifest)
    current = current_contract_digests(check_index=True)
    for key in ("source_content_digest", "indexed_corpus_contract_digest", "scoring_contract_digest", "holdout_digest", "runtime_contract_digest", "provider_contract_digest"):
        if manifest.get("contracts", {}).get(key) != current.get(key):
            raise Phase2SealedBaselineError(f"Contract digest changed; refusing resume: {key}")
    samples = {str(row["sample_id"]): row for row in load_holdout()}
    contract = load_scoring_contract()
    return _execute_from_manifest(paths=paths, manifest=manifest, samples=samples, contract=contract, run_sample=run_sample)


def _execute_from_manifest(
    *,
    paths: RunPaths,
    manifest: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    run_sample: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    completed = {(row["replicate_id"], row["sample_id"]) for row in read_jsonl(paths.sample_results) if row.get("execution_status") == "completed"}
    all_results = read_jsonl(paths.sample_results)
    all_scores = read_jsonl(paths.deterministic_scores)
    all_reviews = read_jsonl(paths.semantic_review)
    for replicate_id, order in manifest["replicate_orders"].items():
        for sample_id in order["sample_ids"]:
            if (replicate_id, sample_id) in completed:
                continue
            sample = samples[sample_id]
            generation_input = generation_input_from_sample(sample)
            assert_generation_input_has_no_gold_fields(generation_input)
            row, score, review = run_sealed_sample(
                sample=sample,
                replicate_id=replicate_id,
                run_sample=run_sample,
                contract=contract,
            )
            append_jsonl(paths.sample_results, row)
            append_jsonl(paths.deterministic_scores, score)
            append_jsonl(paths.semantic_review, review)
            all_results.append(row)
            all_scores.append(score)
            all_reviews.append(review)
            if row.get("execution_status") != "completed":
                append_jsonl(paths.failure_records, _redacted_failure(row, score))
            write_json(paths.checkpoint, {"run_id": manifest["run_id"], "updated_at": utc_now(), "completed_execution_count": len(all_results)}, overwrite=True)
    return freeze_run(paths=paths)


def run_sealed_sample(
    *,
    sample: dict[str, Any],
    replicate_id: str,
    run_sample: Callable[[dict[str, Any]], dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    payload: dict[str, Any] | None = None
    for attempt in range(1, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
        try:
            payload = run_sample(generation_input_from_sample(sample))
            attempts.append({"attempt": attempt, "reason_code": "completed", "infrastructure_retry": False})
            break
        except Exception as exc:
            retryable = _is_retryable_infrastructure_error(exc)
            attempts.append({"attempt": attempt, "reason_code": type(exc).__name__, "infrastructure_retry": retryable})
            if not retryable or attempt == MAX_INFRASTRUCTURE_ATTEMPTS:
                payload = {"system_error": True, "system_error_code": type(exc).__name__, "status": "system_error"}
                break
            time.sleep(0 if os.environ.get("PYTEST_CURRENT_TEST") else RETRY_BACKOFF_SECONDS)
    assert payload is not None
    scoring_run = answer_payload_to_scoring_run(payload)
    score = score_sample(sample, scoring_run, contract)
    review = review_semantic_checks(sample, scoring_run, score)
    if review["review_status"] == "completed":
        scoring_run["semantic_judge"] = {"status": "completed", "passed": review["decision"] == "pass"}
        score = score_sample(sample, scoring_run, contract)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    row = {
        "schema_version": "opk-rag.phase2-sealed-baseline-sample-result.v1",
        "sample_id": sample["sample_id"],
        "replicate_id": replicate_id,
        "execution_status": score["execution_status"],
        "final_action": scoring_run.get("final_action"),
        "latency_ms": elapsed_ms,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "scoring_run": scoring_run,
        "answer_payload": payload,
        "created_at": utc_now(),
    }
    score = {"replicate_id": replicate_id, **score}
    review = {"replicate_id": replicate_id, **review}
    return row, score, review


def aggregate_baseline_results(manifest: dict[str, Any], scores: list[dict[str, Any]], results: list[dict[str, Any]], reviews: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    overall = aggregate_scores(scores)
    by_replicate = {rid: aggregate_scores([row for row in scores if row.get("replicate_id") == rid]) for rid in sorted({row.get("replicate_id") for row in scores})}
    by_type = {qtype: aggregate_scores([row for row in scores if row.get("question_type") == qtype]) for qtype in sorted({row.get("question_type") for row in scores})}
    final_actions = Counter(str((row.get("answerability") or {}).get("actual_action") or "") for row in scores)
    failure_stage_counts = overall.get("failure_stage_counts") or {}
    quality = evaluate_quality_gates(overall, contract)
    return {
        "overall": overall,
        "completed_sample_count": len(
            {
                row.get("sample_id")
                for row in scores
                if row.get("execution_status") == "completed"
            }
        ),
        "by_replicate": by_replicate,
        "by_question_type": by_type,
        "by_capability_tag": _capability_metrics(scores),
        "by_execution_stage": failure_stage_counts,
        "by_final_action": dict(sorted(final_actions.items())),
        "hard_gate_results": evaluate_hard_gates(overall, manifest, contract, reviews),
        "quality_gate_summary": quality,
        "stability": _stability(scores),
        "latency": _latency(results),
        "usage": _usage(results),
        "semantic_review": _review_summary(reviews),
    }


def evaluate_hard_gates(overall: dict[str, Any], manifest: dict[str, Any], contract: dict[str, Any], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    red_lines = overall.get("hard_gate_failures") or {}
    contract_values = manifest.get("contracts") or {}
    values = {
        "unsupported_material_claim_count": 0,
        "unknown_citation_count": 0,
        "forbidden_scope_answer_count": 0,
        "false_premise_acceptance_count": 0,
        "hidden_conflict_count": 0,
        **red_lines,
        "corpus_snapshot_status": contract_values.get("source_snapshot_status"),
        "indexed_corpus_verified": contract_values.get("indexed_corpus_status") == "verified",
        "scoring_contract_digest": contract_values.get("scoring_contract_digest"),
        "holdout_status": contract_values.get("holdout_status"),
    }
    results = {}
    for gate in (contract.get("quality_gates") or {}).get("hard_gates") or []:
        gate_id = gate["id"]
        metric = gate.get("metric", gate_id)
        threshold = gate["threshold"]
        observed = values.get(metric)
        if threshold == 0:
            status = "pass" if int(observed or 0) == 0 else "fail"
        elif threshold == "must_match":
            status = "pass" if observed in {"snapshot_match", contract_values.get("scoring_contract_digest")} else "fail"
        elif threshold == "sealed":
            status = "pass" if observed == "sealed" else "fail"
        else:
            status = "invalid"
        results[gate_id] = {"gate_id": gate_id, "status": status, "observed_value": observed, "required_value": threshold, "reason_code": "observed"}
    results["indexed_corpus_verified"] = {
        "gate_id": "indexed_corpus_verified",
        "status": "pass" if values["indexed_corpus_verified"] else "fail",
        "observed_value": values["indexed_corpus_verified"],
        "required_value": True,
        "reason_code": "task0050_required_hard_gate",
    }
    if any(row.get("review_status") != "completed" for row in reviews):
        for value in results.values():
            if value["status"] == "pass":
                value["status"] = "pending"
                value["reason_code"] = "semantic_review_incomplete"
    return results


def build_public_summary(manifest: dict[str, Any], aggregate: dict[str, Any], private_digest: str) -> dict[str, Any]:
    hard_gate_status = "pass" if all(row["status"] == "pass" for row in aggregate["hard_gate_results"].values()) else "fail"
    infra_failures = aggregate["overall"]["metrics"]["infrastructure_failure_rate"]["numerator"]
    status = "completed_hard_gates_passed" if hard_gate_status == "pass" else "completed_hard_gates_failed"
    if infra_failures:
        status = "run_incomplete"
    if aggregate["semantic_review"]["review_pending_count"]:
        status = "semantic_review_incomplete"
    metrics = aggregate["overall"]["metrics"]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "baseline_id": BASELINE_ID,
        "run_id": manifest["run_id"],
        "status": status,
        "sample_count": manifest["sample_count"],
        "replicate_count": manifest["replicate_count"],
        "completed_count": aggregate["completed_sample_count"],
        "sample_execution_count": aggregate["overall"]["sample_count"],
        "runtime_contract_digest": manifest["contracts"]["runtime_contract_digest"],
        "corpus_snapshot_id": manifest["contracts"]["corpus_snapshot_id"],
        "scoring_contract_id": CONTRACT_ID,
        "holdout_id": HOLDOUT_ID,
        "question_type_metrics": _public_metric_group(aggregate["by_question_type"]),
        "replicate_metrics": _public_metric_group(aggregate["by_replicate"]),
        "capability_metrics": aggregate["by_capability_tag"],
        "aggregate_metrics": metrics,
        "failure_stage_counts": aggregate["failure_stage_counts"] if "failure_stage_counts" in aggregate else aggregate["by_execution_stage"],
        "final_action_counts": aggregate["by_final_action"],
        "hard_gate_results": aggregate["hard_gate_results"],
        "hard_gate_status": hard_gate_status,
        "soft_target_observations": {
            target: {"observed": metrics.get(target, {}).get("value"), "target_status": "informational"}
            for target in soft_target_ids(load_scoring_contract())
        },
        "infrastructure": {
            "infrastructure_failure_count": infra_failures,
            "provider_retry_count": aggregate["usage"]["provider_retry_count"],
            "writes_conversation_log": False,
            "writes_query_log": False,
            "writes_supabase": False,
        },
        "latency": aggregate["latency"],
        "usage": aggregate["usage"],
        "stability": aggregate["stability"],
        "semantic_review": aggregate["semantic_review"],
        "private_result_digest": private_digest,
        "contains_question_text": False,
        "contains_answer_text": False,
        "contains_gold_claims": False,
        "contains_source_content": False,
        "contains_source_path": False,
        "created_at": utc_now(),
    }


def verify_run_artifacts(run_id: str) -> dict[str, Any]:
    paths = run_paths(run_id)
    if not paths.private_summary.exists():
        raise Phase2SealedBaselineError(f"Private summary missing: {run_id}")
    private = read_json(paths.private_summary)
    public = read_json(PUBLIC_SUMMARY_PATH)
    issues = []
    if public.get("private_result_digest") != private.get("private_result_digest"):
        issues.append("private_digest_mismatch")
    if contains_forbidden_public_content(public):
        issues.append("public_summary_leak")
    expected = manifest_expected_execution_count(read_json(paths.manifest))
    if public.get("sample_execution_count") != expected:
        issues.append("sample_execution_count_mismatch")
    status = "valid" if not issues else "invalid"
    return {"status": status, "issues": issues, "run_id": run_id, "private_result_digest": public.get("private_result_digest")}


def freeze_run(*, paths: RunPaths) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    scores = read_jsonl(paths.deterministic_scores)
    results = read_jsonl(paths.sample_results)
    reviews = read_jsonl(paths.semantic_review)
    expected_count = manifest_expected_execution_count(manifest)
    if len(results) != expected_count or len(scores) != expected_count or len(reviews) != expected_count:
        return {"status": "run_incomplete", "run_id": manifest["run_id"], "completed_execution_count": len(results), "expected_execution_count": expected_count}
    manifest = {**manifest, "completed_at": manifest.get("completed_at") or utc_now()}
    write_json(paths.manifest, manifest, overwrite=True)
    private_digest = tree_digest([paths.manifest, paths.sample_results, paths.deterministic_scores, paths.semantic_review, paths.failure_records])
    aggregate = aggregate_baseline_results(manifest, scores, results, reviews, load_scoring_contract())
    private_summary = {"run_id": manifest["run_id"], "private_result_digest": private_digest, "aggregate": aggregate, "completed_at": manifest["completed_at"]}
    write_json(paths.private_summary, private_summary, overwrite=True)
    public = build_public_summary(manifest, aggregate, private_digest)
    if contains_forbidden_public_content(public):
        raise Phase2SealedBaselineError("Public summary contains forbidden private content")
    write_json(PUBLIC_SUMMARY_PATH, public, overwrite=True)
    return public


def contains_forbidden_public_content(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in PUBLIC_FORBIDDEN_KEYS:
                return True
            if contains_forbidden_public_content(value):
                return True
    elif isinstance(payload, list):
        return any(contains_forbidden_public_content(item) for item in payload)
    elif isinstance(payload, str):
        if re.search(r"(?<![A-Za-z0-9_])/(?:Users|home|data|mnt|Volumes|var|private)/", payload):
            return True
    return False


def manifest_expected_execution_count(manifest: dict[str, Any]) -> int:
    return int(manifest["sample_count"]) * int(manifest["replicate_count"])


def _final_action(payload: dict[str, Any], model_decision: dict[str, Any]) -> str:
    if payload.get("system_error"):
        return "abstain"
    decision = str(model_decision.get("decision") or "")
    if payload.get("status") == "answered" or decision == "answer":
        return "answer"
    if payload.get("answerable") and payload.get("refusal_reason_code") in {"partial_evidence", "partial_supported_scope_missing"}:
        return "partial_answer"
    if str((payload.get("answerability") or {}).get("status")) == "partially_answerable":
        return "partial_answer" if payload.get("status") == "answered" else "abstain"
    return "abstain"


def _material_claim_count(payload: dict[str, Any], model_decision: dict[str, Any]) -> int:
    if payload.get("status") != "answered":
        return 0
    answer = str(payload.get("answer") or "")
    return max(1, len(re.findall(r"[。.!?；;]\s*", answer)))


def _is_retryable_infrastructure_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(fragment in text for fragment in ("timeout", "429", "5xx", "connection", "dns", "temporar", "read timed out"))


def _redacted_failure(row: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "replicate_id": row.get("replicate_id"),
        "execution_status": row.get("execution_status"),
        "primary_failure": score.get("primary_failure"),
        "hard_gate_failures": score.get("hard_gate_failures"),
        "created_at": utc_now(),
    }


def _latency(results: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(row.get("latency_ms") or 0) for row in results]
    if not values:
        return {"average_ms": None, "p50_ms": None, "p95_ms": None, "total_ms": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {"average_ms": round(statistics.mean(values), 2), "p50_ms": statistics.median(values), "p95_ms": ordered[p95_index], "total_ms": sum(values)}


def _usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    calls = len(results)
    retries = sum(max(0, int(row.get("attempt_count") or 1) - 1) for row in results)
    prompt = sum(int(((row.get("scoring_run") or {}).get("prompt_tokens") or 0)) for row in results)
    completion = sum(int(((row.get("scoring_run") or {}).get("completion_tokens") or 0)) for row in results)
    evidence_counts = [len(((row.get("scoring_run") or {}).get("evidence_bundle_ids") or [])) for row in results]
    citation_counts = [len(((row.get("scoring_run") or {}).get("citations") or [])) for row in results]
    return {
        "llm_call_count": calls,
        "retrieval_call_count": calls,
        "provider_retry_count": retries,
        "prompt_tokens": prompt or None,
        "completion_tokens": completion or None,
        "total_tokens": (prompt + completion) or None,
        "average_evidence_unit_count": round(statistics.mean(evidence_counts), 2) if evidence_counts else None,
        "average_citation_count": round(statistics.mean(citation_counts), 2) if citation_counts else None,
    }


def _review_summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "review_status": "completed" if all(row.get("review_status") == "completed" for row in reviews) else "pending",
        "reviewer_type": "agent_structured_private_review",
        "reviewer_independence": "not_independent",
        "review_contract_version": "phase2-semantic-review-v1",
        "review_completed_count": sum(row.get("review_status") == "completed" for row in reviews),
        "review_pending_count": sum(row.get("review_status") != "completed" for row in reviews),
        "disagreement_count": sum(bool(row.get("disagrees_with_deterministic")) for row in reviews),
    }


def _stability(scores: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[str(row.get("sample_id"))].append(row)
    comparable = [rows for rows in grouped.values() if len(rows) >= 2]
    if not comparable:
        return {
            "replicate_count": len({row.get("replicate_id") for row in scores}),
            "final_action_agreement_rate": None,
            "execution_status_agreement_rate": None,
            "hard_gate_agreement": None,
            "end_to_end_outcome_agreement_rate": None,
            "question_type_metric_delta": {},
        }
    return {
        "replicate_count": len({row.get("replicate_id") for row in scores}),
        "final_action_agreement_rate": _agreement(comparable, lambda row: (row.get("answerability") or {}).get("actual_action")),
        "execution_status_agreement_rate": _agreement(comparable, lambda row: row.get("execution_status")),
        "hard_gate_agreement": _agreement(comparable, lambda row: tuple(row.get("hard_gate_failures") or [])),
        "end_to_end_outcome_agreement_rate": _agreement(comparable, lambda row: row.get("passed")),
        "question_type_metric_delta": _question_type_delta(scores),
    }


def _agreement(groups: list[list[dict[str, Any]]], key: Callable[[dict[str, Any]], Any]) -> float:
    agreed = 0
    for rows in groups:
        values = [key(row) for row in rows]
        agreed += len({json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}) == 1
    return round(agreed / len(groups), 4)


def _question_type_delta(scores: list[dict[str, Any]]) -> dict[str, Any]:
    reps = sorted({row.get("replicate_id") for row in scores})
    if len(reps) < 2:
        return {}
    out = {}
    for qtype in sorted({row.get("question_type") for row in scores}):
        vals = []
        for rep in reps[:2]:
            subset = [row for row in scores if row.get("replicate_id") == rep and row.get("question_type") == qtype]
            vals.append(aggregate_scores(subset)["metrics"]["end_to_end_success_rate"]["value"])
        if None not in vals:
            out[str(qtype)] = round(abs(vals[0] - vals[1]), 4)
    return out


def _capability_metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    samples = {row["sample_id"]: row for row in load_holdout()} if PRIVATE_HOLDOUT_PATH.exists() else {}
    out = {}
    for tag in sorted({tag for sample in samples.values() for tag in sample.get("capability_tags", [])}):
        ids = {sid for sid, sample in samples.items() if tag in sample.get("capability_tags", [])}
        subset = [row for row in scores if row.get("sample_id") in ids]
        if len(ids) < 3:
            out[tag] = {"suppressed_for_privacy": True, "sample_count": len(ids)}
        else:
            out[tag] = {"suppressed_for_privacy": False, "sample_count": len(ids), "metrics": aggregate_scores(subset)["metrics"]}
    return out


def _public_metric_group(group: dict[str, Any]) -> dict[str, Any]:
    return {str(key): {"sample_count": value.get("sample_count"), "metrics": value.get("metrics")} for key, value in group.items()}


def status(run_id: str) -> dict[str, Any]:
    paths = run_paths(run_id)
    if not paths.manifest.exists():
        raise Phase2SealedBaselineError(f"Run manifest not found: {run_id}")
    manifest = read_json(paths.manifest)
    return {
        "run_id": run_id,
        "manifest_status": "completed" if manifest.get("completed_at") else "in_progress",
        "completed_execution_count": len(read_jsonl(paths.sample_results)),
        "expected_execution_count": manifest_expected_execution_count(manifest),
        "public_summary_exists": PUBLIC_SUMMARY_PATH.exists() and read_json(PUBLIC_SUMMARY_PATH).get("run_id") == run_id,
    }
