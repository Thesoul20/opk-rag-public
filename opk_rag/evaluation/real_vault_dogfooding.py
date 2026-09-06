from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Callable


DATASET_VERSION = "real-vault-dogfooding-v1"
QUESTION_SCHEMA_VERSION = "real-vault-dogfooding-question.v1"
RESULT_SCHEMA_VERSION = "real-vault-dogfooding-result.v1"
REPORT_SCHEMA_VERSION = "real-vault-dogfooding-report.v1"
DIAGNOSIS_SCHEMA_VERSION = "real-vault-failure-diagnosis.v1"
RUNTIME_VERSION = "phase2-reference-runtime"

QUESTION_TYPES = {
    "direct_lookup",
    "single_document_summary",
    "cross_document_synthesis",
    "temporal_or_status",
    "scope_limited",
    "partial_answer",
    "false_premise",
    "no_evidence",
    "ambiguous",
    "source_location",
}
ANSWERABILITY_LABELS = {"fully_answerable", "partially_answerable", "not_answerable"}
FINAL_ACTIONS = {"answer", "partial_answer", "abstain"}
PRIVACY_CLASSES = {"public_safe", "sanitized", "private_local_only"}
ANNOTATION_STATUSES = {"draft", "reviewed", "locked"}
FORMAL_STATUSES = {"reviewed", "locked"}
FAILURE_TYPES = {
    "no_failure",
    "infrastructure_failure",
    "index_freshness_failure",
    "retrieval_no_hit",
    "retrieval_wrong_source",
    "retrieval_ranking_failure",
    "retrieval_insufficient_evidence",
    "answerability_false_abstention",
    "answerability_unsafe_allow",
    "partial_answer_missed",
    "generation_abstention",
    "generation_failure",
    "citation_failure",
    "grounding_failure",
    "unsupported_answer",
    "expected_abstention",
}
SECRET_PATTERNS = (
    re.compile(r"postgres(?:ql)?://[^@\s]+:[^@\s]+@", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk)_[A-Za-z0-9][A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{16,}:[A-Za-z0-9_\-]{16,}\b"),
)
ABSOLUTE_PATH_PATTERN = re.compile(r"(^|[\s\"'])((?:/[^\s\"']+)|(?:[A-Za-z]:\\[^\s\"']+))")


class DogfoodingValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    sample_count: int
    formal_sample_count: int
    warnings: tuple[str, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DogfoodingValidationError(f"Dataset does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DogfoodingValidationError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise DogfoodingValidationError(f"Dataset line {line_number} is not a JSON object.")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_source_aliases(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise DogfoodingValidationError(f"Source alias file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DogfoodingValidationError("Source alias file must be a JSON object.")
    for alias, value in payload.items():
        if not re.fullmatch(r"DOC-[A-Z][0-9]{2}", alias):
            raise DogfoodingValidationError(f"Invalid source alias: {alias}")
        if not isinstance(value, dict):
            raise DogfoodingValidationError(f"Alias mapping must be an object: {alias}")
        if not isinstance(value.get("relative_path"), str) or not value["relative_path"].strip():
            raise DogfoodingValidationError(f"Alias mapping is missing relative_path: {alias}")
        if Path(value["relative_path"]).is_absolute():
            raise DogfoodingValidationError(f"Alias mapping must not contain an absolute path: {alias}")
    return payload


def load_run_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise DogfoodingValidationError(f"Run config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DogfoodingValidationError("Run config must be a JSON object.")
    return payload


def validate_dataset(
    *,
    dataset_path: Path,
    source_aliases_path: Path,
    run_config_path: Path | None = None,
    repo_root: Path | None = None,
) -> ValidationResult:
    rows = read_jsonl(dataset_path)
    aliases = load_source_aliases(source_aliases_path)
    run_config = load_run_config(run_config_path)
    snapshot_ids = _snapshot_ids(run_config)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        prefix = f"line {index}"
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not re.fullmatch(r"RV-[0-9]{3}", sample_id):
            errors.append(f"{prefix}: sample_id must match RV-001 format")
        elif sample_id in seen:
            errors.append(f"{prefix}: duplicate sample_id {sample_id}")
        else:
            seen.add(sample_id)
        if row.get("dataset_version") != DATASET_VERSION:
            errors.append(f"{prefix}: dataset_version must be {DATASET_VERSION}")
        snapshot_id = row.get("vault_snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            errors.append(f"{prefix}: vault_snapshot_id is required")
        elif snapshot_ids and snapshot_id not in snapshot_ids:
            errors.append(f"{prefix}: vault_snapshot_id is not present in run_config")
        _validate_enum(errors, prefix, row, "expected_answerability", ANSWERABILITY_LABELS)
        _validate_enum(errors, prefix, row, "expected_final_action", FINAL_ACTIONS)
        _validate_enum(errors, prefix, row, "privacy_class", PRIVACY_CLASSES)
        _validate_enum(errors, prefix, row, "annotation_status", ANNOTATION_STATUSES)
        _validate_question_type(errors, prefix, row)
        _validate_sources(errors, prefix, row, aliases)
        _validate_scope_contract(errors, prefix, row)
        _validate_privacy_text(errors, prefix, row)
    if repo_root is not None:
        for path in (dataset_path, source_aliases_path):
            if _is_under_private(path) and _is_git_tracked(repo_root, path):
                errors.append(f"private file is tracked by Git: {path}")
    if errors:
        raise DogfoodingValidationError("\n".join(errors))
    formal = [row for row in rows if row.get("annotation_status") in FORMAL_STATUSES]
    if len(formal) < len(rows):
        warnings.append("Only reviewed or locked samples enter formal baseline metrics.")
    return ValidationResult(sample_count=len(rows), formal_sample_count=len(formal), warnings=tuple(warnings))


def run_evaluation(
    *,
    records: list[dict[str, Any]],
    source_aliases: dict[str, dict[str, Any]],
    output_path: Path,
    run_sample: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool = False,
    overwrite: bool = False,
    max_samples: int | None = None,
    sample_ids: set[str] | None = None,
    per_sample_timeout: float = 300.0,
) -> list[dict[str, Any]]:
    selected = [row for row in records if row.get("annotation_status") in FORMAL_STATUSES]
    if sample_ids:
        selected = [row for row in selected if row["sample_id"] in sample_ids]
    if max_samples is not None:
        selected = selected[:max_samples]
    if output_path.exists() and not (resume or overwrite):
        raise DogfoodingValidationError(f"Refusing to overwrite existing output without --resume or --overwrite: {output_path}")
    existing = _existing_by_id(output_path) if resume else {}
    if overwrite and output_path.exists():
        output_path.unlink()
    reverse_aliases = _reverse_aliases(source_aliases)
    results_by_id = dict(existing)
    for record in selected:
        sample_id = record["sample_id"]
        existing_row = results_by_id.get(sample_id)
        if resume and existing_row and existing_row.get("error_type") != "infrastructure_failure":
            continue
        started = time.perf_counter()
        try:
            payload = _with_timeout(lambda: run_sample(record), per_sample_timeout)
            raw_row = score_sample(record=record, answer=payload, reverse_aliases=reverse_aliases)
        except TimeoutError:
            raw_row = score_sample(record=record, answer=_system_error("per_sample_timeout"), reverse_aliases=reverse_aliases)
        except Exception as exc:
            raw_row = score_sample(record=record, answer=_system_error(type(exc).__name__), reverse_aliases=reverse_aliases)
        row = {
            **raw_row,
            "schema_version": RESULT_SCHEMA_VERSION,
            "dataset_version": record["dataset_version"],
            "vault_snapshot_id": record["vault_snapshot_id"],
            "question": record["question"],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "created_at": _utc_now(),
        }
        results_by_id[sample_id] = row
        _persist_ordered(output_path, selected, results_by_id)
    return [results_by_id[row["sample_id"]] for row in selected if row["sample_id"] in results_by_id]


def score_sample(*, record: dict[str, Any], answer: dict[str, Any], reverse_aliases: dict[str, str]) -> dict[str, Any]:
    expected_action = record["expected_final_action"]
    expected_positive = expected_action in {"answer", "partial_answer"}
    actual_action = _actual_action(answer, expected_action)
    retrieved_source_aliases = _retrieved_aliases(answer, reverse_aliases)
    expected_aliases = list(record.get("expected_source_aliases") or [])
    retrieved_expected = not expected_aliases or bool(set(expected_aliases) & set(retrieved_source_aliases))
    retrieval_status = _retrieval_status(expected_aliases, retrieved_source_aliases, retrieved_expected)
    retrieval_sufficient = retrieval_status in {"not_required", "hit"}
    decision_status = _decision_status(answer)
    generation_status = _generation_status(answer, actual_action, decision_status)
    citation_status = _citation_status(answer, actual_action)
    grounding_status = _grounding_status(answer, actual_action)
    error_type = classify_failure(
        expected_action=expected_action,
        expected_answerability=record["expected_answerability"],
        actual_action=actual_action,
        retrieval_status=retrieval_status,
        retrieval_sufficient=retrieval_sufficient,
        decision_status=decision_status,
        generation_status=generation_status,
        citation_status=citation_status,
        grounding_status=grounding_status,
        system_error=bool(answer.get("system_error")),
        unsupported_claims=bool(answer.get("unsupported_claims")),
    )
    return {
        "sample_id": record["sample_id"],
        "question_type": record["question_type"],
        "expected_answerability": record["expected_answerability"],
        "expected_final_action": expected_action,
        "expected_source_aliases": expected_aliases,
        "retrieved_source_aliases": retrieved_source_aliases,
        "retrieval_status": retrieval_status,
        "retrieval_sufficient": retrieval_sufficient,
        "decision_status": decision_status,
        "decision_reason": _decision_reason(answer),
        "generation_status": generation_status,
        "citation_status": citation_status,
        "grounding_status": grounding_status,
        "final_action": actual_action,
        "error_type": error_type,
        "failure_stage": _failure_stage(error_type),
        "supported_scope": _as_list(answer.get("supported_scope") or _model_decision(answer).get("supported_scope")),
        "unsupported_scope": _as_list(answer.get("unsupported_scope") or _model_decision(answer).get("unsupported_scope")),
        "premise_correction": answer.get("premise_correction") or _model_decision(answer).get("premise_correction"),
        "provider_model": answer.get("model_id"),
        "private_answer_payload": answer,
    }


def classify_failure(
    *,
    expected_action: str,
    expected_answerability: str,
    actual_action: str,
    retrieval_status: str,
    retrieval_sufficient: bool,
    decision_status: str,
    generation_status: str,
    citation_status: str,
    grounding_status: str,
    system_error: bool,
    unsupported_claims: bool,
) -> str:
    if system_error:
        return "infrastructure_failure"
    if expected_action == "abstain" and actual_action == "abstain":
        return "expected_abstention"
    if expected_action == "abstain" and actual_action in {"answer", "partial_answer"}:
        return "answerability_unsafe_allow" if not unsupported_claims else "unsupported_answer"
    if expected_action in {"answer", "partial_answer"} and retrieval_status == "empty":
        return "retrieval_no_hit"
    if expected_action in {"answer", "partial_answer"} and retrieval_status == "wrong_source":
        return "retrieval_wrong_source"
    if expected_action in {"answer", "partial_answer"} and not retrieval_sufficient:
        return "retrieval_insufficient_evidence"
    if expected_action == "partial_answer" and actual_action == "abstain":
        return "partial_answer_missed"
    if expected_action == "answer" and actual_action == "abstain" and decision_status == "refused":
        return "answerability_false_abstention"
    if expected_action in {"answer", "partial_answer"} and actual_action == "abstain" and generation_status == "abstained":
        return "generation_abstention"
    if generation_status == "error":
        return "generation_failure"
    if actual_action in {"answer", "partial_answer"} and citation_status != "valid":
        return "citation_failure"
    if actual_action in {"answer", "partial_answer"} and grounding_status not in {"grounded", "disabled"}:
        return "grounding_failure"
    if unsupported_claims:
        return "unsupported_answer"
    return "no_failure"


def build_public_report(*, dataset_path: Path, results: list[dict[str, Any]], records: list[dict[str, Any]], source_aliases: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    formal = [row for row in records if row.get("annotation_status") in FORMAL_STATUSES]
    completed = [row for row in results if row["error_type"] != "infrastructure_failure"]
    expected_positive = [row for row in completed if row["expected_final_action"] in {"answer", "partial_answer"}]
    expected_positive_noninfra = [row for row in results if row["error_type"] != "infrastructure_failure" and row["expected_final_action"] in {"answer", "partial_answer"}]
    expected_partial_completed = [row for row in completed if row["expected_final_action"] == "partial_answer"]
    expected_sources = [row for row in completed if row["expected_source_aliases"]]
    actual_final_actions = dict(sorted(Counter(row["final_action"] for row in results).items()))
    safe_answers = [row for row in expected_positive if row["final_action"] in {"answer", "partial_answer"} and row["error_type"] == "no_failure"]
    false_abstention_rows = [row for row in expected_positive_noninfra if row["final_action"] == "abstain"]
    post_retrieval_population = [row for row in expected_positive_noninfra if not str(row["error_type"]).startswith("retrieval_")]
    post_retrieval_false_abstention_rows = [row for row in post_retrieval_population if row["final_action"] == "abstain"]
    fully_answerable_abstained = sum(row["expected_answerability"] == "fully_answerable" and row["final_action"] == "abstain" for row in false_abstention_rows)
    partially_answerable_abstained = sum(row["expected_answerability"] == "partially_answerable" and row["final_action"] == "abstain" for row in false_abstention_rows)
    retrieval_caused_abstention = sum(row["error_type"] == "retrieval_wrong_source" and row["final_action"] == "abstain" for row in false_abstention_rows)
    generation_abstention = sum(row["error_type"] == "generation_abstention" for row in false_abstention_rows)
    partial_answer_missed = sum(row["error_type"] == "partial_answer_missed" for row in false_abstention_rows)
    false_abstention_numerator = len(false_abstention_rows)
    post_retrieval_false_abstention_numerator = len(post_retrieval_false_abstention_rows)
    excluded_retrieval_primary_failures = len(expected_positive_noninfra) - len(post_retrieval_population)
    if false_abstention_numerator != generation_abstention + partial_answer_missed + retrieval_caused_abstention:
        raise DogfoodingValidationError("False abstention numerator does not satisfy generation + partial_answer_missed + retrieval_caused_abstention conservation.")
    if false_abstention_numerator != fully_answerable_abstained + partially_answerable_abstained:
        raise DogfoodingValidationError("False abstention numerator does not satisfy fully_answerable_abstained + partially_answerable_abstained conservation.")
    if post_retrieval_false_abstention_numerator != generation_abstention + partial_answer_missed:
        raise DogfoodingValidationError("Post-retrieval false abstention numerator does not satisfy generation_abstention + partial_answer_missed conservation.")
    correct_full_answers = [row for row in results if _primary_outcome(row) == "correct_full_answer"]
    correct_partial_answers = [row for row in results if _primary_outcome(row) == "correct_partial_answer"]
    correct_abstentions = [row for row in results if _primary_outcome(row) == "correct_abstention"]
    failure_rows = [row for row in results if _primary_outcome(row) not in {"correct_full_answer", "correct_partial_answer", "correct_abstention"}]
    failure_counter = Counter(_primary_outcome(row) for row in failure_rows)
    failure_attribution_counts = {
        label: failure_counter.get(label, 0)
        for label in sorted(label for label in FAILURE_TYPES if label not in {"no_failure", "expected_abstention"})
    }
    primary_outcome_counts = dict(sorted(Counter(_primary_outcome(row) for row in results).items()))
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed" if results else "not_run",
        "dataset_path": _public_path(dataset_path),
        "dataset_version": DATASET_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "created_at": _utc_now(),
        "git_commit": _git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "vault_snapshot": _public_snapshot(run_config),
        "source_alias_manifest": {
            "source_alias_count": len(source_aliases),
            "alias_format": "DOC-A01",
            "mapping_file_present": bool(source_aliases),
            "mapping_file_tracked": False,
        },
        "dataset_summary": {
            "total_samples": len(records),
            "formal_samples": len(formal),
            "question_type_counts": dict(sorted(Counter(str(row.get("question_type")) for row in records).items())),
            "expected_answerability_counts": dict(sorted(Counter(str(row.get("expected_answerability")) for row in records).items())),
            "expected_final_action_counts": dict(sorted(Counter(str(row.get("expected_final_action")) for row in records).items())),
            "annotation_status_counts": dict(sorted(Counter(str(row.get("annotation_status")) for row in records).items())),
            "privacy_class_counts": dict(sorted(Counter(str(row.get("privacy_class")) for row in records).items())),
        },
        "actual_final_actions": actual_final_actions,
        "correct_outcomes": {
            "correct_full_answer": len(correct_full_answers),
            "correct_partial_answer": len(correct_partial_answers),
            "correct_abstention": len(correct_abstentions),
            "safe_answer": {
                "numerator": len(safe_answers),
                "denominator": len(expected_positive),
                "value": _rate(len(safe_answers), len(expected_positive)),
                "excluded_infrastructure_failures": sum(
                    row["error_type"] == "infrastructure_failure" and row["expected_final_action"] in {"answer", "partial_answer"}
                    for row in results
                ),
            },
            "compatibility": {
                "expected_abstention": len(correct_abstentions),
            },
        },
        "failure_attribution": failure_attribution_counts,
        "primary_outcome_counts": primary_outcome_counts,
        "primary_outcome_total": len(results),
        "metrics": {
            "total_samples": len(results),
            "completed_samples": len(completed),
            "infrastructure_failures": len(results) - len(completed),
            "infrastructure_failure_rate": _metric(len(results) - len(completed), len(results)),
            "answer_count": actual_final_actions.get("answer", 0),
            "partial_answer_count": actual_final_actions.get("partial_answer", 0),
            "abstain_count": actual_final_actions.get("abstain", 0),
            "unsupported_answer_count": sum(row["error_type"] == "unsupported_answer" for row in results),
            "unsafe_allow_count": sum(row["error_type"] == "answerability_unsafe_allow" for row in results),
            "citation_failure_count": sum(row["error_type"] == "citation_failure" for row in results),
            "grounding_failure_count": sum(row["error_type"] == "grounding_failure" for row in results),
            "safe_answer_coverage": _metric(
                len(safe_answers),
                len(expected_positive),
                excluded_infrastructure_failures=sum(
                    row["error_type"] == "infrastructure_failure" and row["expected_final_action"] in {"answer", "partial_answer"}
                    for row in results
                ),
            ),
            "fully_answerable_answer_rate": _metric(sum(row["expected_answerability"] == "fully_answerable" and row["final_action"] == "answer" and row["error_type"] == "no_failure" for row in completed), sum(row["expected_answerability"] == "fully_answerable" for row in completed)),
            "partial_answer_success_rate": _metric(sum(row["expected_final_action"] == "partial_answer" and row["final_action"] == "partial_answer" and row["error_type"] == "no_failure" for row in completed), len(expected_partial_completed)),
            "false_abstention_rate": _metric(
                false_abstention_numerator,
                len(expected_positive_noninfra),
                excluded_infrastructure_failures=sum(row["error_type"] == "infrastructure_failure" and row["expected_final_action"] in {"answer", "partial_answer"} for row in results),
            ),
            "post_retrieval_false_abstention_rate": _metric(
                post_retrieval_false_abstention_numerator,
                len(post_retrieval_population),
                excluded_infrastructure_failures=sum(row["error_type"] == "infrastructure_failure" and row["expected_final_action"] in {"answer", "partial_answer"} for row in results),
                excluded_retrieval_primary_failures=excluded_retrieval_primary_failures,
            ),
            "false_abstention_breakdown": {
                "fully_answerable_abstained": fully_answerable_abstained,
                "partially_answerable_abstained": partially_answerable_abstained,
                "retrieval_caused_abstention": retrieval_caused_abstention,
                "generation_abstention": generation_abstention,
                "partial_answer_missed": partial_answer_missed,
            },
            "expected_source_recall_at_k": _metric(sum(bool(set(row["expected_source_aliases"]) & set(row["retrieved_source_aliases"])) for row in expected_sources), len(expected_sources)),
            "retrieval_sufficient_rate": _metric(sum(row["retrieval_sufficient"] for row in completed), len(completed)),
            "wrong_source_rate": _metric(sum(row["error_type"] == "retrieval_wrong_source" for row in completed), len(completed)),
            "ranking_failure_count": sum(row["error_type"] == "retrieval_ranking_failure" for row in completed),
            "actual_final_actions": actual_final_actions,
        },
        "representative_sample_ids": {
            key: [row["sample_id"] for row in results if _primary_outcome(row) == key][:5]
            for key in sorted(primary_outcome_counts)
        },
        "privacy": {
            "contains_questions": False,
            "contains_paths": False,
            "contains_evidence_text": False,
            "contains_raw_provider_output": False,
            "contains_secrets": False,
        },
    }
    assert_public_report_safe(report)
    return report


def assert_public_report_safe(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaks = _privacy_leaks(text)
    if leaks:
        raise DogfoodingValidationError(f"Public report contains private-looking content: {sorted(leaks)}")


def write_public_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    summary = report["dataset_summary"]
    safe_answer = metrics["safe_answer_coverage"]
    partial_answer = metrics["partial_answer_success_rate"]
    false_abstention = metrics["false_abstention_rate"]
    post_retrieval_false_abstention = metrics["post_retrieval_false_abstention_rate"]
    false_abstention_breakdown = metrics["false_abstention_breakdown"]
    expected_source_recall = metrics["expected_source_recall_at_k"]
    infra = metrics["infrastructure_failure_rate"]
    lines = [
        "# Real Vault Dogfooding Baseline",
        "",
        "This report is intentionally sanitized. It contains aggregate counts, sample IDs, and failure attribution only.",
        "",
        "## Runtime",
        "",
        f"- Reference runtime: `{report['runtime_version']}`",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Status: `{report['status']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Snapshot ID: `{report['vault_snapshot'].get('snapshot_id', 'not_recorded')}`",
        "",
        "## Dataset",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Formal samples: {summary['formal_samples']}",
        f"- Expected answerability: `{summary['expected_answerability_counts']}`",
        f"- Expected final action: `{summary['expected_final_action_counts']}`",
        f"- Question types: `{summary['question_type_counts']}`",
        "",
        "## Metrics",
        "",
        f"- Completed samples: {metrics['completed_samples']} / {metrics['total_samples']}",
        f"- Infrastructure failures: {metrics['infrastructure_failures']}",
        f"- Actual final actions: `{metrics['actual_final_actions']}`",
        f"- Safe answer coverage: `{safe_answer['numerator']} / {safe_answer['denominator']} = {safe_answer['value']} (excluded infrastructure failures: {safe_answer['excluded_infrastructure_failures']})`",
        f"- Overall false abstention: `{false_abstention['numerator']} / {false_abstention['denominator']} = {false_abstention['value']}`",
        f"- False abstention breakdown: `{false_abstention_breakdown}`",
        f"- Post-retrieval false abstention: `{post_retrieval_false_abstention['numerator']} / {post_retrieval_false_abstention['denominator']} = {post_retrieval_false_abstention['value']} (excluded retrieval primary failures: {post_retrieval_false_abstention['excluded_retrieval_primary_failures']}, excluded infrastructure failures: {post_retrieval_false_abstention['excluded_infrastructure_failures']})`",
        f"- Partial answer success rate: `{partial_answer['numerator']} / {partial_answer['denominator']} = {partial_answer['value']}`",
        f"- Unsupported answers: {metrics['unsupported_answer_count']}",
        f"- Citation failures: {metrics['citation_failure_count']}",
        f"- Grounding failures: {metrics['grounding_failure_count']}",
        f"- Expected source recall@k: `{expected_source_recall['numerator']} / {expected_source_recall['denominator']} = {expected_source_recall['value']}`",
        f"- Infrastructure failure rate: `{infra['numerator']} / {infra['denominator']} = {infra['value']}`",
        "",
        "## Outcome Breakdown",
        "",
        f"- Correct outcomes: `{report['correct_outcomes']}`",
        f"- Primary outcome counts: `{report['primary_outcome_counts']}`",
        f"- Failure attribution: `{report['failure_attribution']}`",
        f"- Representative sample IDs: `{report['representative_sample_ids']}`",
        "",
        "## Privacy",
        "",
        "- Private questions, paths, evidence text, raw provider output, secrets, and knowledge base IDs are excluded.",
        "- `expected_abstention` is retained only as a compatibility alias for `correct_abstention` and is not counted as a failure stage.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_failures(*, results: list[dict[str, Any]], records: list[dict[str, Any]], replay_performed: bool = False) -> dict[str, Any]:
    records_by_id = {row["sample_id"]: row for row in records if isinstance(row.get("sample_id"), str)}
    failure_rows = [row for row in results if row.get("error_type") in {"generation_abstention", "partial_answer_missed", "retrieval_wrong_source", "answerability_unsafe_allow"}]
    diagnostics = [_diagnose_failure_row(row, records_by_id.get(str(row.get("sample_id")), {})) for row in failure_rows]
    root_cause_counts = dict(sorted(Counter(item["diagnosis"]["primary_root_cause"] for item in diagnostics).items()))
    failure_type_counts = dict(sorted(Counter(str(item["primary_outcome"]) for item in diagnostics).items()))
    original_stage_counts = dict(sorted(Counter(str(item["original_failure_stage"]) for item in diagnostics).items()))
    audited_stage_counts = dict(sorted(Counter(str(item["audited_failure_stage"]) for item in diagnostics).items()))
    attribution_differences = [
        {
            "sample_id": item["sample_id"],
            "original_failure_stage": item["original_failure_stage"],
            "audited_failure_stage": item["audited_failure_stage"],
            "original_error_type": item["primary_outcome"],
            "audited_root_cause": item["diagnosis"]["primary_root_cause"],
        }
        for item in diagnostics
        if item["original_failure_stage"] != item["audited_failure_stage"] or item["primary_outcome"] != item["diagnosis"]["primary_root_cause"]
    ]
    field_audit = _diagnostic_field_audit(results)
    generation_rows = [item for item in diagnostics if item["primary_outcome"] == "generation_abstention"]
    partial_rows = [item for item in diagnostics if item["primary_outcome"] == "partial_answer_missed"]
    retrieval_rows = [item for item in diagnostics if item["primary_outcome"] == "retrieval_wrong_source"]
    unsafe_rows = [item for item in diagnostics if item["primary_outcome"] == "answerability_unsafe_allow"]
    return {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "created_at": _utc_now(),
        "replay": {
            "performed": replay_performed,
            "remote_replay_required": False,
            "note": "No remote replay required." if not replay_performed else "Replay diagnostics are marked separately and are not included in TASK-0039 baseline metrics.",
        },
        "input_audit": field_audit,
        "sample_count": len(diagnostics),
        "failure_type_counts": failure_type_counts,
        "original_failure_stage_counts": original_stage_counts,
        "audited_failure_stage_counts": audited_stage_counts,
        "root_cause_counts": root_cause_counts,
        "stage_summaries": {
            "retrieval": _retrieval_diagnosis_summary(retrieval_rows),
            "generation": _generation_diagnosis_summary(generation_rows),
            "partial_answer": _partial_diagnosis_summary(partial_rows),
            "unsafe_allow": _unsafe_allow_summary(unsafe_rows),
        },
        "attribution_differences": attribution_differences,
        "repair_candidates": _repair_candidates(diagnostics),
        "diagnostics": diagnostics,
        "privacy": {
            "contains_questions": False,
            "contains_paths": False,
            "contains_evidence_text": False,
            "contains_raw_provider_output": False,
            "contains_secrets": False,
        },
    }


def write_failure_diagnosis_markdown(path: Path, diagnosis: dict[str, Any], baseline_report: dict[str, Any] | None = None) -> None:
    metrics = (baseline_report or {}).get("metrics", {})
    summary = (baseline_report or {}).get("dataset_summary", {})
    root_counts = diagnosis["root_cause_counts"]
    candidates = diagnosis["repair_candidates"]
    lines = [
        "# Real Vault Failure Diagnosis",
        "",
        "This report is sanitized. It contains sample IDs, reason codes, status labels, counts, and audited categories only.",
        "",
        "## TASK-0039 Baseline Summary",
        "",
        f"- Dataset version: `{diagnosis['dataset_version']}`",
        f"- Total samples: {summary.get('total_samples', 'unknown')}",
        f"- Expected answerability: `{summary.get('expected_answerability_counts', {})}`",
        f"- Actual final actions: `{metrics.get('actual_final_actions', {})}`",
        f"- Safe answer coverage: `{_format_metric(metrics.get('safe_answer_coverage'))}`",
        f"- Post-retrieval false abstention: `{_format_metric(metrics.get('post_retrieval_false_abstention_rate'))}`",
        f"- Unsupported answers: {metrics.get('unsupported_answer_count', 'unknown')}",
        f"- Citation failures: {metrics.get('citation_failure_count', 'unknown')}",
        f"- Grounding failures: {metrics.get('grounding_failure_count', 'unknown')}",
        "",
        "## Audit Scope",
        "",
        f"- Failure samples audited: {diagnosis['sample_count']}",
        f"- Failure type counts: `{diagnosis['failure_type_counts']}`",
        f"- Existing-result field coverage: `{diagnosis['input_audit']['required_field_status']}`",
        f"- Missing fine-grained retrieval traces: `{diagnosis['input_audit']['missing_retrieval_trace_fields']}`",
        f"- Remote replay: `{diagnosis['replay']['note']}`",
        "",
        "## Root Cause Matrix",
        "",
        "| Root cause category | Count |",
        "| --- | ---: |",
    ]
    for key, value in root_counts.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Retrieval Failures",
            "",
            _summary_bullet("Retrieval audited samples", diagnosis["stage_summaries"]["retrieval"]),
            "",
            "## Generation Abstention",
            "",
            _summary_bullet("Generation audited samples", diagnosis["stage_summaries"]["generation"]),
            "",
            "## Partial Answer Missed",
            "",
            _summary_bullet("Partial-answer audited samples", diagnosis["stage_summaries"]["partial_answer"]),
            "",
            "## Unsafe Allow",
            "",
            _summary_bullet("Unsafe-allow audited samples", diagnosis["stage_summaries"]["unsafe_allow"]),
            "",
            "## Attribution Differences",
            "",
        ]
    )
    if diagnosis["attribution_differences"]:
        lines.extend(["| Sample | Original stage | Audited stage | Audited root cause |", "| --- | --- | --- | --- |"])
        for item in diagnosis["attribution_differences"]:
            lines.append(f"| `{item['sample_id']}` | `{item['original_failure_stage']}` | `{item['audited_failure_stage']}` | `{item['audited_root_cause']}` |")
    else:
        lines.append("- No original-vs-audited attribution differences.")
    lines.extend(["", "## Repair Candidates", "", "| Priority | Candidate | Affected | Risk | Scope | Confidence | Next task |", "| ---: | --- | ---: | --- | --- | --- | --- |"])
    for index, item in enumerate(candidates, start=1):
        lines.append(
            f"| {index} | `{item['candidate_id']}` | {item['affected_sample_count']} | `{item['unsupported_answer_risk']}` | `{item['implementation_scope']}` | `{item['confidence']}` | {item['recommended_next_task']} |"
        )
    lines.extend(
        [
            "",
            "## Risks",
            "",
            "- The existing TASK-0039 result file does not contain raw vector, lexical, hybrid, dedup, or rank traces, so Retrieval sub-causes are lower-confidence without explicit replay.",
            "- Post-generation validation refusals preserved zero unsupported answers in the baseline; relaxing those checks is high risk unless tested as a single-variable experiment.",
            "",
            "## Recommended TASK-0041",
            "",
            "Run a minimal generation-contract and partial-answer propagation experiment against the frozen 36-sample baseline, keeping retrieval, answerability thresholds, citation validation, grounding validation, and provider model fixed.",
        ]
    )
    payload = "\n".join(lines) + "\n"
    assert_public_report_safe({"markdown": payload})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _diagnose_failure_row(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("private_answer_payload") if isinstance(row.get("private_answer_payload"), dict) else {}
    model_decision = _model_decision(answer)
    answerability = answer.get("answerability") if isinstance(answer.get("answerability"), dict) else {}
    grounding = answer.get("grounding") if isinstance(answer.get("grounding"), dict) else {}
    expected_aliases = row.get("expected_source_aliases") if isinstance(row.get("expected_source_aliases"), list) else []
    retrieved_aliases = row.get("retrieved_source_aliases") if isinstance(row.get("retrieved_source_aliases"), list) else []
    expected_source_hit = not expected_aliases or bool(set(expected_aliases) & set(retrieved_aliases))
    selected_source_hit = expected_source_hit
    error_type = str(row.get("error_type"))
    root_cause, audited_stage, tags, confidence, fix_layer = _root_cause_for(row=row, answer=answer, model_decision=model_decision)
    contract_failures = model_decision.get("contract_failures") if isinstance(model_decision.get("contract_failures"), list) else []
    unsupported_claims_value = model_decision.get("unsupported_claims")
    unsupported_claim_count = len(unsupported_claims_value) if isinstance(unsupported_claims_value, list) else 0
    citations = model_decision.get("citations") if isinstance(model_decision.get("citations"), list) else []
    return {
        "sample_id": row.get("sample_id"),
        "primary_outcome": error_type,
        "original_failure_stage": row.get("failure_stage"),
        "audited_failure_stage": audited_stage,
        "question_type": row.get("question_type"),
        "expected_answerability": row.get("expected_answerability"),
        "expected_final_action": row.get("expected_final_action"),
        "actual_final_action": row.get("final_action"),
        "retrieval": {
            "expected_source_hit": expected_source_hit,
            "expected_source_rank": "not_available",
            "selected_source_hit": selected_source_hit,
            "evidence_count": len(retrieved_aliases),
            "evidence_sufficient": row.get("retrieval_sufficient"),
            "evidence_scope_match": _evidence_scope_match(row, model_decision),
            "wrong_source_reason": _wrong_source_reason(row),
            "ranking_diagnostic": _ranking_diagnostic(row),
        },
        "answerability": {
            "decision": row.get("decision_status"),
            "decision_reason": row.get("decision_reason"),
            "supported_scope_present": bool(row.get("supported_scope")),
            "unsupported_scope_present": bool(row.get("unsupported_scope")),
            "premise_correction_present": bool(row.get("premise_correction")),
            "decision_matches_annotation": _decision_matches_annotation(row),
            "reason_code": answerability.get("reason_code"),
            "status": answerability.get("status"),
        },
        "generation": {
            "invoked": answer.get("generation_latency_ms") is not None,
            "raw_generation_status": answer.get("status"),
            "explicit_abstention": model_decision.get("decision") == "abstain",
            "structured_output_valid": bool(model_decision),
            "contract_fields_present": _contract_fields_present(model_decision),
            "answer_text_present": bool(model_decision.get("answer")),
            "citations_present": bool(citations),
            "provider_error": bool(answer.get("system_error")),
            "generation_diagnostic": _generation_diagnostic(answer, model_decision, contract_failures),
            "refusal_reason_code": answer.get("refusal_reason_code"),
            "contract_failures": _sanitize_codes(contract_failures),
        },
        "grounding": {
            "citation_valid": row.get("citation_status") == "valid",
            "grounding_valid": row.get("grounding_status") in {"grounded", "disabled"},
            "unsupported_answer": error_type == "unsupported_answer",
            "grounding_reason_code": grounding.get("reason_code"),
            "unsupported_claim_count": unsupported_claim_count,
        },
        "diagnosis": {
            "primary_root_cause": root_cause,
            "contributing_factors": _contributing_factors(row, answer, model_decision),
            "confidence": confidence,
            "recommended_fix_layer": fix_layer,
        },
        "diagnostic_tags": tags,
    }


def _root_cause_for(*, row: dict[str, Any], answer: dict[str, Any], model_decision: dict[str, Any]) -> tuple[str, str, list[str], str, str]:
    error_type = row.get("error_type")
    reason = answer.get("refusal_reason_code")
    contract_failures = model_decision.get("contract_failures") if isinstance(model_decision.get("contract_failures"), list) else []
    if error_type == "generation_abstention":
        if reason in {"unsupported_claims", "ungrounded_answer"} or contract_failures:
            return "post_generation_validation_abstention", "generation", ["provider_generated_answer", str(reason or "contract_failure")], "high", "generation_contract"
        if model_decision.get("decision") == "abstain":
            return "explicit_model_abstention", "generation", ["provider_explicit_abstain"], "high", "provider_behavior"
        if not model_decision:
            return "invalid_json", "generation", ["structured_output_missing"], "medium", "generation_contract"
        return "generation_contract_mismatch", "generation", ["decision_allowed_then_abstained"], "medium", "generation_contract"
    if error_type == "partial_answer_missed":
        if row.get("decision_status") == "refused":
            return "answerability_full_abstention", "answerability", ["partial_expected_decision_refused"], "high", "answerability"
        if model_decision.get("decision") == "abstain":
            return "provider_ignores_partial_contract", "generation", ["partial_contract_provider_abstain"], "high", "generation_contract"
        if reason == "ungrounded_answer":
            return "partial_answer_grounding_failure", "generation", ["partial_answer_failed_grounding"], "high", "generation_contract"
        if reason in {"unsupported_claims", "partial_disclosure_missing", "partial_supported_scope_missing"} or contract_failures:
            return "provider_ignores_partial_contract", "generation", ["partial_contract_post_generation_refusal"], "high", "generation_contract"
        return "partial_scope_not_propagated", "generation", ["partial_scope_unclear"], "medium", "generation_contract"
    if error_type == "retrieval_wrong_source":
        if row.get("final_action") in {"answer", "partial_answer"} and row.get("grounding_status") == "grounded":
            return "source_alias_annotation_issue", "annotation", ["grounded_answer_without_expected_alias_hit"], "medium", "evaluation"
        return "evidence_selection_failure", "retrieval", ["expected_source_not_selected", "raw_rank_trace_missing"], "medium", "retrieval"
    if error_type == "answerability_unsafe_allow":
        if row.get("grounding_status") == "grounded":
            return "question_interpretation_mismatch", "annotation", ["grounded_but_expected_abstain", "annotation_review_required"], "medium", "annotation_review"
        return "answerability_threshold_too_permissive", "answerability", ["expected_abstain_allowed"], "medium", "answerability"
    return str(error_type or "other"), str(row.get("failure_stage") or "unknown"), ["unclassified"], "low", "unknown"


def _diagnostic_field_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = set()
    private_fields = set()
    for row in results:
        fields.update(row)
        answer = row.get("private_answer_payload")
        if isinstance(answer, dict):
            private_fields.update(answer)
    required = [
        "sample_id",
        "expected_answerability",
        "expected_final_action",
        "expected_source_aliases",
        "retrieved_source_aliases",
        "retrieval_status",
        "retrieval_sufficient",
        "selected_evidence_ids",
        "decision_status",
        "decision_reason",
        "supported_scope",
        "unsupported_scope",
        "premise_correction",
        "generation_status",
        "generation_reason",
        "provider_raw_status",
        "citation_status",
        "grounding_status",
        "final_action",
        "failure_stage",
        "error_type",
        "latency_ms",
    ]
    retrieval_trace_fields = ["raw_vector_results", "raw_lexical_results", "raw_hybrid_results", "expected_source_rank", "dedup_diagnostics"]
    return {
        "present_result_fields": sorted(fields),
        "present_private_payload_fields": sorted(private_fields),
        "required_field_status": {field: field in fields for field in required},
        "missing_required_fields": [field for field in required if field not in fields],
        "can_complete_from_existing_results": True,
        "requires_new_diagnostic_script": True,
        "can_reuse_evaluation_module": True,
        "minimal_replay_needed_sample_ids": [],
        "missing_retrieval_trace_fields": [field for field in retrieval_trace_fields if field not in fields],
        "classification_conflicts": [],
    }


def _retrieval_diagnosis_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "expected_source_hit_count": sum(row["retrieval"]["expected_source_hit"] for row in rows),
        "selected_source_hit_count": sum(row["retrieval"]["selected_source_hit"] for row in rows),
        "grounded_answer_despite_alias_miss_count": sum(row["actual_final_action"] in {"answer", "partial_answer"} and row["grounding"]["grounding_valid"] for row in rows),
        "raw_vector_lexical_hybrid_trace_available": False,
        "dominant_root_causes": dict(sorted(Counter(row["diagnosis"]["primary_root_cause"] for row in rows).items())),
    }


def _generation_diagnosis_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "decision_allowed_count": sum(row["answerability"]["decision"] == "allowed" for row in rows),
        "provider_invoked_count": sum(row["generation"]["invoked"] for row in rows),
        "actual_abstain_count": sum(row["actual_final_action"] == "abstain" for row in rows),
        "explicit_model_abstention_count": sum(row["generation"]["explicit_abstention"] for row in rows),
        "structured_output_error_count": sum(not row["generation"]["structured_output_valid"] for row in rows),
        "post_generation_validation_abstention_count": sum(row["diagnosis"]["primary_root_cause"] == "post_generation_validation_abstention" for row in rows),
        "citation_or_grounding_post_failure_count": sum(row["generation"]["refusal_reason_code"] in {"unsupported_claims", "ungrounded_answer"} for row in rows),
        "actually_answerability_stage_count": sum(row["audited_failure_stage"] == "answerability" for row in rows),
        "root_causes": dict(sorted(Counter(row["diagnosis"]["primary_root_cause"] for row in rows).items())),
    }


def _partial_diagnosis_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "answerability_full_abstention_count": sum(row["diagnosis"]["primary_root_cause"] == "answerability_full_abstention" for row in rows),
        "decision_correct_provider_refused_count": sum(row["answerability"]["decision"] == "allowed" and row["actual_final_action"] == "abstain" for row in rows),
        "scope_propagation_issue_count": sum(row["diagnosis"]["primary_root_cause"] in {"partial_scope_not_propagated", "provider_ignores_partial_contract"} for row in rows),
        "evidence_insufficient_count": sum(not row["retrieval"]["evidence_sufficient"] for row in rows),
        "should_split_current_attribution": True,
        "root_causes": dict(sorted(Counter(row["diagnosis"]["primary_root_cause"] for row in rows).items())),
    }


def _unsafe_allow_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "grounded_answer_count": sum(row["grounding"]["grounding_valid"] for row in rows),
        "annotation_review_required_count": sum("annotation_review_required" in row["diagnostic_tags"] for row in rows),
        "root_causes": dict(sorted(Counter(row["diagnosis"]["primary_root_cause"] for row in rows).items())),
    }


def _repair_candidates(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["diagnosis"]["primary_root_cause"] for row in diagnostics)
    specs = {
        "post_generation_validation_abstention": ("generation_abstention_handling", "generation_abstention", "high", "generation contract and validator diagnostics", "high"),
        "provider_ignores_partial_contract": ("partial_answer_contract_propagation", "partial_answer_missed", "medium", "generation contract", "high"),
        "partial_answer_grounding_failure": ("partial_answer_grounding_contract", "partial_answer_missed", "medium", "generation contract", "high"),
        "evidence_selection_failure": ("retrieval_evidence_selection_trace", "retrieval_wrong_source", "medium", "retrieval diagnostics", "medium"),
        "source_alias_annotation_issue": ("annotation_alias_review", "retrieval_wrong_source", "low", "annotation review", "medium"),
        "question_interpretation_mismatch": ("unsafe_allow_annotation_review", "answerability_unsafe_allow", "low", "annotation review", "medium"),
    }
    candidates = []
    for root_cause, count in counts.items():
        candidate_id, failure_type, risk, scope, confidence = specs.get(root_cause, (root_cause, "mixed", "unknown", "unknown", "low"))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "affected_sample_count": count,
                "affected_failure_types": sorted({row["primary_outcome"] for row in diagnostics if row["diagnosis"]["primary_root_cause"] == root_cause}),
                "expected_metric_impact": f"May recover up to {count} audited failure sample(s) if fixed without introducing validation regressions.",
                "unsupported_answer_risk": risk,
                "implementation_scope": scope,
                "confidence": confidence,
                "recommended_next_task": "TASK-0041 single-variable frozen-baseline experiment" if failure_type != "retrieval_wrong_source" else "TASK-0041 retrieval trace audit before parameter changes",
            }
        )
    return sorted(candidates, key=lambda item: (-item["affected_sample_count"], item["unsupported_answer_risk"], item["candidate_id"]))


def _decision_matches_annotation(row: dict[str, Any]) -> bool:
    if row.get("expected_final_action") == "abstain":
        return row.get("decision_status") == "refused"
    return row.get("decision_status") == "allowed"


def _contract_fields_present(model_decision: dict[str, Any]) -> bool:
    return {"decision", "answer", "citations", "reason"}.issubset(model_decision)


def _evidence_scope_match(row: dict[str, Any], model_decision: dict[str, Any]) -> str:
    if row.get("retrieval_status") == "wrong_source":
        return "expected_source_miss"
    if row.get("decision_status") == "allowed" and (row.get("supported_scope") or model_decision.get("supported_scope")):
        return "supported_scope_present"
    return "unknown"


def _wrong_source_reason(row: dict[str, Any]) -> str | None:
    if row.get("retrieval_status") != "wrong_source":
        return None
    if row.get("final_action") in {"answer", "partial_answer"} and row.get("grounding_status") == "grounded":
        return "source_alias_annotation_issue"
    return "expected_source_not_selected"


def _ranking_diagnostic(row: dict[str, Any]) -> str:
    if row.get("retrieval_status") == "wrong_source":
        return "rank_trace_not_available"
    if row.get("retrieval_status") == "hit":
        return "expected_source_selected"
    return str(row.get("retrieval_status") or "unknown")


def _generation_diagnostic(answer: dict[str, Any], model_decision: dict[str, Any], contract_failures: list[Any]) -> str:
    if answer.get("system_error"):
        return "provider_error"
    if model_decision.get("decision") == "abstain":
        return "explicit_model_abstention"
    if answer.get("refusal_reason_code") == "ungrounded_answer":
        return "post_generation_grounding_refusal"
    if answer.get("refusal_reason_code") == "unsupported_claims" or contract_failures:
        return "post_generation_validation_abstention"
    if not model_decision:
        return "structured_output_missing"
    return str(answer.get("status") or "unknown")


def _contributing_factors(row: dict[str, Any], answer: dict[str, Any], model_decision: dict[str, Any]) -> list[str]:
    factors = []
    if row.get("retrieval_status") == "wrong_source":
        factors.append("expected_source_alias_not_retrieved")
    if row.get("decision_status") == "allowed" and row.get("final_action") == "abstain":
        factors.append("answerability_allowed_before_final_refusal")
    if answer.get("refusal_reason_code"):
        factors.append(str(answer["refusal_reason_code"]))
    if model_decision.get("decision") == "answer" and answer.get("status") == "refused":
        factors.append("provider_answer_rejected_after_validation")
    if row.get("expected_final_action") == "abstain" and row.get("final_action") in {"answer", "partial_answer"}:
        factors.append("expected_abstain_but_allowed")
    return factors


def _sanitize_codes(values: list[Any]) -> list[str]:
    return [str(value).split(":", 1)[0] for value in values]


def _summary_bullet(label: str, payload: dict[str, Any]) -> str:
    return f"- {label}: `{payload}`"


def _format_metric(metric: Any) -> str:
    if not isinstance(metric, dict):
        return "unknown"
    return f"{metric.get('numerator')} / {metric.get('denominator')} = {metric.get('value')}"


def _validate_enum(errors: list[str], prefix: str, row: dict[str, Any], field: str, allowed: set[str]) -> None:
    if row.get(field) not in allowed:
        errors.append(f"{prefix}: {field} must be one of {sorted(allowed)}")


def _validate_question_type(errors: list[str], prefix: str, row: dict[str, Any]) -> None:
    value = row.get("question_type")
    values = value if isinstance(value, list) else [value]
    if not values or any(item not in QUESTION_TYPES for item in values):
        errors.append(f"{prefix}: question_type contains an unsupported value")


def _validate_sources(errors: list[str], prefix: str, row: dict[str, Any], aliases: dict[str, Any]) -> None:
    values = row.get("expected_source_aliases")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        errors.append(f"{prefix}: expected_source_aliases must be a string array")
        return
    missing = [value for value in values if value not in aliases]
    if missing:
        errors.append(f"{prefix}: expected_source_aliases missing from alias map: {missing}")


def _validate_scope_contract(errors: list[str], prefix: str, row: dict[str, Any]) -> None:
    answerability = row.get("expected_answerability")
    action = row.get("expected_final_action")
    if answerability == "fully_answerable" and action == "abstain":
        errors.append(f"{prefix}: fully_answerable cannot expect abstain")
    if answerability == "not_answerable" and action == "answer":
        errors.append(f"{prefix}: not_answerable cannot expect answer")
    supported = row.get("expected_supported_scope")
    unsupported = row.get("expected_unsupported_scope")
    if action == "partial_answer" and (not supported or not unsupported):
        errors.append(f"{prefix}: partial_answer requires supported and unsupported scope")
    if not isinstance(row.get("expected_citation_required"), bool):
        errors.append(f"{prefix}: expected_citation_required must be boolean")


def _validate_privacy_text(errors: list[str], prefix: str, row: dict[str, Any]) -> None:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    leaks = _privacy_leaks(text)
    if leaks:
        errors.append(f"{prefix}: contains private-looking values: {sorted(leaks)}")


def _privacy_leaks(text: str) -> set[str]:
    leaks = set()
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        leaks.add("secret")
    if ABSOLUTE_PATH_PATTERN.search(text):
        leaks.add("absolute_path")
    return leaks


def _snapshot_ids(run_config: dict[str, Any]) -> set[str]:
    ids = set()
    value = run_config.get("snapshot_id")
    if isinstance(value, str):
        ids.add(value)
    snapshots = run_config.get("snapshots")
    if isinstance(snapshots, list):
        ids.update(item.get("snapshot_id") for item in snapshots if isinstance(item, dict) and isinstance(item.get("snapshot_id"), str))
    return ids


def _is_under_private(path: Path) -> bool:
    return "evaluation-data/dogfooding/private" in path.as_posix()


def _is_git_tracked(repo_root: Path, path: Path) -> bool:
    rel = path if not path.is_absolute() else path.relative_to(repo_root)
    result = subprocess.run(["git", "ls-files", "--", rel.as_posix()], cwd=repo_root, text=True, capture_output=True, check=False)
    return bool(result.stdout.strip())


def _existing_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["sample_id"]: row for row in read_jsonl(path) if isinstance(row.get("sample_id"), str)}


def _persist_ordered(path: Path, records: list[dict[str, Any]], results_by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [results_by_id[row["sample_id"]] for row in records if row["sample_id"] in results_by_id]
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _with_timeout(fn: Callable[[], dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return fn()

    def handler(_signum, _frame) -> None:
        raise TimeoutError("per-sample timeout exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _reverse_aliases(source_aliases: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {str(value.get("relative_path")): alias for alias, value in source_aliases.items()}


def _actual_action(answer: dict[str, Any], expected_action: str) -> str:
    if answer.get("system_error"):
        return "system_error"
    if answer.get("status") == "answered" or answer.get("answerable") is True:
        return "partial_answer" if expected_action == "partial_answer" else "answer"
    return "abstain"


def _retrieved_aliases(answer: dict[str, Any], reverse_aliases: dict[str, str]) -> list[str]:
    sources = answer.get("retrieved_sources")
    if not isinstance(sources, list):
        sources = []
    aliases = []
    for source in sources:
        alias = reverse_aliases.get(str(source))
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _retrieval_status(expected_aliases: list[str], retrieved_aliases: list[str], retrieved_expected: bool) -> str:
    if not expected_aliases:
        return "not_required"
    if not retrieved_aliases:
        return "empty"
    if retrieved_expected:
        return "hit"
    return "wrong_source"


def _decision_status(answer: dict[str, Any]) -> str:
    answerability = answer.get("answerability") if isinstance(answer.get("answerability"), dict) else {}
    status = answerability.get("status")
    if answer.get("system_error"):
        return "error"
    if answerability.get("answerable") is True or status in {"answerable", "partially_answerable"}:
        return "allowed"
    if answerability.get("answerable") is False or status == "unanswerable":
        return "refused"
    return "unknown"


def _generation_status(answer: dict[str, Any], actual_action: str, decision_status: str) -> str:
    decision = _model_decision(answer)
    if answer.get("system_error"):
        return "error"
    if actual_action in {"answer", "partial_answer"}:
        return "generated"
    if decision_status == "refused":
        return "not_started"
    if decision.get("decision") == "abstain" or answer.get("generation_latency_ms") is not None:
        return "abstained"
    return "not_started"


def _citation_status(answer: dict[str, Any], actual_action: str) -> str:
    if actual_action not in {"answer", "partial_answer"}:
        return "not_applicable"
    citations = answer.get("citations")
    if not isinstance(citations, list) or not citations:
        return "missing"
    return "valid"


def _grounding_status(answer: dict[str, Any], actual_action: str) -> str:
    grounding = answer.get("grounding") if isinstance(answer.get("grounding"), dict) else {}
    if actual_action not in {"answer", "partial_answer"}:
        return str(grounding.get("reason_code") or "not_applicable")
    if grounding.get("valid") is True:
        return "grounded"
    if grounding.get("status") == "disabled":
        return "disabled"
    return str(grounding.get("reason_code") or "invalid")


def _decision_reason(answer: dict[str, Any]) -> str | None:
    answerability = answer.get("answerability") if isinstance(answer.get("answerability"), dict) else {}
    return answerability.get("reason_code") or answer.get("refusal_reason_code")


def _model_decision(answer: dict[str, Any]) -> dict[str, Any]:
    value = answer.get("model_decision")
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _failure_stage(error_type: str) -> str:
    if error_type.startswith("retrieval_") or error_type == "index_freshness_failure":
        return "retrieval"
    if error_type in {"answerability_false_abstention", "answerability_unsafe_allow", "partial_answer_missed"}:
        return "answerability"
    if error_type == "expected_abstention":
        return "expected_outcome"
    if error_type.startswith("generation_"):
        return "generation"
    if error_type == "citation_failure":
        return "citation"
    if error_type in {"grounding_failure", "unsupported_answer"}:
        return "grounding"
    if error_type == "infrastructure_failure":
        return "infrastructure"
    return "none"


def _system_error(code: str) -> dict[str, Any]:
    return {"status": "system_error", "answerable": False, "system_error": True, "system_error_code": code, "retrieved_sources": [], "citations": [], "grounding": {"valid": False, "status": "error", "reason_code": "system_error"}}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metric(
    numerator: int,
    denominator: int,
    *,
    excluded_infrastructure_failures: int | None = None,
    excluded_retrieval_primary_failures: int | None = None,
) -> dict[str, Any]:
    payload = {
        "numerator": numerator,
        "denominator": denominator,
        "value": _rate(numerator, denominator),
    }
    if excluded_infrastructure_failures is not None:
        payload["excluded_infrastructure_failures"] = excluded_infrastructure_failures
    if excluded_retrieval_primary_failures is not None:
        payload["excluded_retrieval_primary_failures"] = excluded_retrieval_primary_failures
    return payload


def _primary_outcome(row: dict[str, Any]) -> str:
    expected_action = row.get("expected_final_action")
    final_action = row.get("final_action")
    error_type = row.get("error_type")
    if expected_action == "answer" and final_action == "answer" and error_type == "no_failure":
        return "correct_full_answer"
    if expected_action == "partial_answer" and final_action == "partial_answer" and error_type == "no_failure":
        return "correct_partial_answer"
    if expected_action == "abstain" and final_action == "abstain" and error_type == "expected_abstention":
        return "correct_abstention"
    return str(error_type)


def _public_path(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "evaluation-data" in parts:
        return "/".join(parts[parts.index("evaluation-data"):])
    return path.name


def _public_snapshot(run_config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "snapshot_id",
        "created_at",
        "git_commit",
        "document_count",
        "markdown_file_count",
        "chunk_count",
        "embedding_model",
        "embedding_dimension",
        "database_schema_version",
        "parser_version",
        "chunking_version",
        "knowledge_base_alias",
        "content_hash_aggregate",
        "index_completed_at",
        "index_run_status",
    }
    return {key: run_config[key] for key in allowed if key in run_config}


def _git(args: list[str]) -> str | None:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
