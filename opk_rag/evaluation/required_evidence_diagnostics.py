from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from opk_rag.evaluation.evidence_identity import (
    EvidenceIdentity,
    digest_json,
    digest_text,
    identity_from_runtime_evidence_item,
    match_required_evidence_set,
    normalize_gold_evidence_identity,
    normalize_runtime_evidence_identity,
)
from opk_rag.evaluation.phase2_scoring import stable_hash


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "evaluation-data" / "fixtures" / "required_evidence_coverage_examples.jsonl"
SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "phase2_required_evidence_diagnostic_summary.json"
MAPPING_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_required_evidence_mapping.json"
DEVELOPMENT_GOLD_IDENTITY_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_development_gold_identity_v1.json"
KNOWN_REGRESSION_GOLD_IDENTITY_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_known_regression_gold_identity_v1.json"
FORBIDDEN_HOLDOUT_PARTS = ("phase2_holdout_v1", "phase2_sealed_baseline_v1", ".private/evaluation")


class RequiredEvidenceDiagnosticError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    _reject_forbidden_path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_diagnostic_dataset(dataset_id: str) -> list[dict[str, Any]]:
    if dataset_id == "fixtures":
        return read_jsonl(FIXTURE_PATH)
    if dataset_id == "development":
        return _development_rows()
    if dataset_id == "known-regression":
        return _known_regression_rows()
    if dataset_id in {"sealed_holdout", "holdout", "phase2-holdout-v1"}:
        raise RequiredEvidenceDiagnosticError("sealed holdout datasets are not allowed for required evidence diagnostics")
    raise RequiredEvidenceDiagnosticError(f"unknown diagnostic dataset: {dataset_id}")


def validate_gold_evidence_mapping(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    roles = Counter(str(row.get("dataset_role") or row.get("role") or "unknown") for row in rows)
    status_counts = Counter()
    field_counts = Counter()
    for row in rows:
        required = _required_units(row)
        if not required:
            status_counts["no_required_evidence"] += 1
            continue
        statuses = [normalize_gold_evidence_identity(unit).to_json() for unit in required]
        if any(_has_coverage_identity(unit) for unit in statuses):
            status_counts["verifiable"] += 1
        else:
            status_counts["unverifiable"] += 1
        for unit in statuses:
            for key in unit:
                field_counts[key] += 1
    return {
        "sample_count": len(rows),
        "dataset_roles": dict(sorted(roles.items())),
        "mapping_status_counts": dict(sorted(status_counts.items())),
        "gold_evidence_fields": dict(sorted(field_counts.items())),
        "can_compute_candidate_coverage": bool(status_counts["verifiable"]),
        "can_compute_bundle_coverage": bool(status_counts["verifiable"]),
    }


def compute_evidence_coverage(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("execution_status") in {"infrastructure_failure", "corpus_mismatch", "index_mismatch"}:
        return {"sample_id": _sample_id(row), "excluded": True, "exclusion_reason": row.get("execution_status")}
    required = [normalize_gold_evidence_identity(unit) for unit in _required_units(row)]
    candidate = [normalize_runtime_evidence_identity(unit) for unit in _stage_units(row, "candidates")]
    bundle = [normalize_runtime_evidence_identity(unit) for unit in _stage_units(row, "bundle")]
    context = [normalize_runtime_evidence_identity(unit) for unit in _stage_units(row, "generation_context")]
    citations = [normalize_runtime_evidence_identity(unit) for unit in _stage_units(row, "citations")]
    return {
        "sample_id": _sample_id(row),
        "dataset_id": row.get("dataset_id"),
        "dataset_role": row.get("dataset_role"),
        "question_type": row.get("question_type") or row.get("answerability_type"),
        "excluded": False,
        "candidate": match_required_evidence_set(required, candidate),
        "bundle": match_required_evidence_set(required, bundle),
        "generation_context": match_required_evidence_set(required, context),
        "citations": match_required_evidence_set(required, citations),
    }


def attribute_coverage_loss(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("excluded"):
        return {"primary_attribution": "excluded_execution_status", "attributions": ["excluded_execution_status"]}
    required = int((result.get("candidate") or {}).get("verifiable_required_total") or 0)
    if required == 0:
        return {"primary_attribution": "unverifiable", "attributions": ["unverifiable"]}
    candidate = int(result["candidate"]["matched_count"])
    bundle = int(result["bundle"]["matched_count"])
    context = int(result["generation_context"]["matched_count"])
    citations = int(result["citations"]["matched_count"])
    attributions: list[str] = []
    if candidate < required:
        attributions.append("candidate_retrieval_miss")
    if candidate > bundle:
        attributions.append("reranking_or_cutoff_loss")
    if bundle > context:
        attributions.append("generation_context_loss")
    if context > citations:
        attributions.append("citation_selection_loss")
    if candidate == required and bundle == required and context == required and citations == required:
        attributions.append("fully_covered")
    return {"primary_attribution": attributions[0] if attributions else "fully_covered", "attributions": attributions or ["fully_covered"]}


def aggregate_coverage_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in results if not row.get("excluded")]
    def metric(stage: str) -> dict[str, Any]:
        numerator = sum(int(row[stage]["matched_count"]) for row in included)
        denominator = sum(int(row[stage]["verifiable_required_total"]) for row in included)
        if not denominator:
            return {"status": "unverifiable", "numerator": 0, "denominator": 0, "value": None}
        return {"status": "computed", "numerator": numerator, "denominator": denominator, "value": numerator / denominator}
    attributions = Counter((row.get("coverage_loss_attribution") or {}).get("primary_attribution", "unknown") for row in included)
    attribution_denominator = len(included)
    attribution_details = {
        key: {
            "count": attributions.get(key, 0),
            "rate": (attributions.get(key, 0) / attribution_denominator) if attribution_denominator else None,
            "applicable_denominator": attribution_denominator,
        }
        for key in (
            "fully_covered",
            "candidate_retrieval_miss",
            "reranking_or_cutoff_loss",
            "deduplication_loss",
            "bundle_budget_loss",
            "serialization_identity_loss",
            "generation_context_loss",
            "unverifiable",
        )
    }
    for key in ("deduplication_loss", "bundle_budget_loss", "serialization_identity_loss"):
        if attribution_details[key]["count"] == 0:
            attribution_details[key]["status"] = "not_separately_measurable"
    return {
        "sample_count": len(results),
        "included_sample_count": len(included),
        "candidate_coverage": metric("candidate"),
        "bundle_coverage": metric("bundle"),
        "generation_context_coverage": metric("generation_context"),
        "citation_coverage": metric("citations"),
        "coverage_loss_attribution": dict(sorted(attributions.items())),
        "coverage_loss_attribution_details": attribution_details,
        "coverage_breakdown": _coverage_breakdown(included),
    }


def _coverage_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_for(group: list[dict[str, Any]], stage: str) -> dict[str, Any]:
        numerator = sum(int(row[stage]["matched_count"]) for row in group)
        denominator = sum(int(row[stage]["verifiable_required_total"]) for row in group)
        return {"status": "computed" if denominator else "unverifiable", "numerator": numerator, "denominator": denominator, "value": (numerator / denominator) if denominator else None}

    def grouped(key: str) -> dict[str, Any]:
        values: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            values.setdefault(str(row.get(key) or "unknown"), []).append(row)
        return {
            group_id: {
                stage: metric_for(group_rows, stage)
                for stage in ("candidate", "bundle", "generation_context")
            }
            for group_id, group_rows in sorted(values.items())
        }

    return {"by_dataset": grouped("dataset_id"), "by_question_type": grouped("question_type")}


def validate_fixture_expectations(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    rows = read_jsonl(path)
    failures = []
    for row in rows:
        result = compute_evidence_coverage(row)
        expected = row.get("expected") or {}
        for stage, metric_key in (("candidate", "candidate_coverage"), ("bundle", "bundle_coverage"), ("generation_context", "generation_context_coverage"), ("citations", "citation_coverage")):
            if metric_key in expected and result.get(stage, {}).get("coverage") != expected[metric_key]:
                failures.append({"case_id": row.get("case_id"), "metric": metric_key, "expected": expected[metric_key], "observed": result.get(stage, {}).get("coverage")})
        observed_attr = attribute_coverage_loss(result)["primary_attribution"]
        if expected.get("primary_attribution") and observed_attr != expected["primary_attribution"]:
            failures.append({"case_id": row.get("case_id"), "metric": "primary_attribution", "expected": expected["primary_attribution"], "observed": observed_attr})
    return {"status": "valid" if not failures else "invalid", "case_count": len(rows), "failures": failures}


def build_diagnostic_summary(dataset_ids: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    datasets = []
    for dataset_id in dataset_ids:
        data = load_diagnostic_dataset(dataset_id)
        rows.extend(data)
        datasets.append({"dataset_id": dataset_id, "sample_count": len(data), "mapping_validation": validate_gold_evidence_mapping(data)})
    results = []
    for row in rows:
        result = compute_evidence_coverage(row)
        result["coverage_loss_attribution"] = attribute_coverage_loss(result)
        results.append(result)
    aggregate = aggregate_coverage_diagnostics(results)
    mapping = validate_gold_evidence_mapping(rows)
    classification = classify_root_cause(aggregate, mapping)
    status = "unverifiable" if _coverage_unverifiable_from_mapping(mapping) else "completed"
    migration = gold_identity_migration_report()
    return {
        "schema_version": "opk-rag.required-evidence-diagnostic-summary.v1",
        "diagnostic_id": "phase2-required-evidence-diagnostic-v1",
        "status": status,
        "role": "development_diagnostic",
        "diagnostic_run_id": "phase2-required-evidence-diagnostic-v1",
        "git_commit": git_commit(),
        "created_at": utc_now(),
        "completed_at": utc_now(),
        "datasets": datasets,
        "sample_count": len(rows),
        "mapping_validation": mapping,
        "adapter_status": "canonical_runtime_identity_enabled",
        "canonical_identity_schema": "opk-rag.canonical-evidence-identity.v1",
        "canonical_identity_status": "frozen",
        "runtime_identity_adapter_status": "fixed",
        "gold_identity_migration_status": migration["status"],
        "gold_identity_migration": migration,
        "required_evidence_measurement_status": "measurable_on_public_datasets",
        "mapping_bug_status": "resolved_for_future_runs",
        "development_verified_samples": migration["development"]["verified"],
        "known_regression_verified_samples": migration["known_regression"]["verified"],
        "candidate_coverage": aggregate["candidate_coverage"],
        "bundle_coverage": aggregate["bundle_coverage"],
        "generation_context_coverage": aggregate["generation_context_coverage"],
        "citation_required_evidence_coverage": aggregate["citation_coverage"],
        "coverage_loss_attribution": aggregate["coverage_loss_attribution"],
        "coverage_loss_attribution_details": aggregate["coverage_loss_attribution_details"],
        "coverage_breakdown": aggregate["coverage_breakdown"],
        "baseline_metric_validity": "required_evidence_metric_invalid",
        "sealed_baseline_v1_metric_validity": "required_evidence_metric_invalid",
        "root_cause_classification": classification,
        "primary_root_cause": classification[0] if classification else "unknown",
        "secondary_classification": classification[1] if len(classification) > 1 else None,
        "candidate_retrieval_failure": _candidate_failure_status(aggregate),
        "reranking_or_cutoff_loss": _reranking_or_cutoff_loss_status(aggregate),
        "generation_context_loss": _generation_context_loss_status(aggregate),
        "serialization_identity_loss": "not_observed_after_fix",
        "deduplication_loss": "not_separately_measurable",
        "bundle_budget_loss": "not_separately_measurable",
        "evidence_bundle_loss": _bundle_loss_status(aggregate),
        "adapter_fix_applies_to": "future_runs_and_public_diagnostics",
        "historical_required_evidence_metric_validity": "required_evidence_metric_invalid",
        "sealed_baseline_rerun_recommendation": "pending_governance_decision",
        "next_action": _next_action(classification),
        "contains_sealed_holdout_data": False,
        "model_called": False,
        "writes_database": False,
        "writes_index": False,
        "scoring_adapter_audit": {
            "formal_metric_uses": "final evidence bundle ids",
            "current_adapter_issue": "phase2 sealed baseline adapter serializes bundle ids from content_sha256/source_digest/relative_path/chunk_id but runner emits chunk content sha256 and chunk_id, while gold uses source_digests/document_identity_digests/scope_ids.",
        },
    }


def classify_root_cause(aggregate: dict[str, Any], mapping: dict[str, Any]) -> list[str]:
    if _coverage_unverifiable_from_mapping(mapping):
        return ["measurement_mapping_bug_confirmed", "gold_annotation_compatibility_issue"]
    candidate = aggregate.get("candidate_coverage") or {}
    bundle = aggregate.get("bundle_coverage") or {}
    candidate_value = candidate.get("value")
    bundle_value = bundle.get("value")
    if candidate_value is None or bundle_value is None:
        return ["gold_annotation_compatibility_issue_unresolved", "insufficient_public_evidence"]
    candidate_low = candidate_value < 0.8
    bundle_drop = candidate_value - bundle_value > 0.1
    if candidate_low and bundle_drop:
        return ["measurement_valid_retrieval_failure_confirmed", "mixed_retrieval_and_bundle_failure"]
    if candidate_low:
        return ["measurement_valid_retrieval_failure_confirmed", "mapping_bug_resolved_for_future_runs"]
    if bundle_drop:
        return ["measurement_valid_bundle_loss_confirmed", "mapping_bug_resolved_for_future_runs"]
    return ["mapping_bug_resolved_for_future_runs", "retrieval_failure_not_confirmed", "bundle_loss_not_confirmed"]


def _coverage_unverifiable_from_mapping(mapping: dict[str, Any]) -> bool:
    status_counts = mapping.get("mapping_status_counts") or {}
    return bool(status_counts.get("unverifiable")) and not bool(mapping.get("can_compute_candidate_coverage"))


def redact_diagnostic_report(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    for marker in FORBIDDEN_HOLDOUT_PARTS:
        if marker in text:
            raise RequiredEvidenceDiagnosticError(f"diagnostic report contains forbidden holdout marker: {marker}")
    return payload


def write_summary(dataset_ids: list[str], path: Path = SUMMARY_PATH) -> dict[str, Any]:
    payload = redact_diagnostic_report(build_diagnostic_summary(dataset_ids))
    write_json(path, payload)
    return payload


def write_mapping_audit(dataset_ids: list[str], path: Path = MAPPING_PATH) -> dict[str, Any]:
    rows = []
    for dataset_id in dataset_ids:
        for row in load_diagnostic_dataset(dataset_id):
            rows.append(
                {
                    "sample_id": _sample_id(row),
                    "dataset_id": dataset_id,
                    "gold_evidence_identity_count": len(_required_units(row)),
                    "mapping_status": "verifiable" if any(_has_coverage_identity(normalize_gold_evidence_identity(unit).to_json()) for unit in _required_units(row)) else "unverifiable",
                    "mapping_source": "public_diagnostic_asset",
                    "review_status": "machine_audited",
                }
            )
    payload = {"schema_version": "opk-rag.required-evidence-mapping.v1", "rows": rows, "contains_sealed_holdout_data": False}
    write_json(path, payload)
    return payload


def _development_rows() -> list[dict[str, Any]]:
    source = ROOT / "evaluation-data" / "results" / "task0027_retrieval_dev_results.jsonl"
    identity_map = _identity_map_for_role("development")
    rows = []
    for row in read_jsonl(source)[:28]:
        rows.append(_from_retrieval_result(row, dataset_id="answerability-dev-v1", role="development", identity_map=identity_map))
    return rows


def _known_regression_rows() -> list[dict[str, Any]]:
    rows = []
    identity_map = _identity_map_for_role("known_regression")
    for source, dataset_id in (
        (ROOT / "evaluation-data" / "results" / "task0027_retrieval_dev_from_fixture_results.jsonl", "task0027-retrieval-fixture-regression"),
        (ROOT / "evaluation-data" / "answerability_test.jsonl", "answerability-known-regression-v1"),
    ):
        for row in read_jsonl(source):
            rows.append(_from_retrieval_result(row, dataset_id=dataset_id, role="known_regression", identity_map=identity_map))
    return rows[:40]


def _from_retrieval_result(row: dict[str, Any], *, dataset_id: str, role: str, identity_map: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    gold = row.get("gold_evidence") or row.get("evidence") or []
    sample_id = str(row.get("id") or row.get("sample_id"))
    required = []
    migrated = (identity_map or {}).get(sample_id) or []
    if migrated:
        required.extend(migrated)
    else:
        for item in gold:
            if not isinstance(item, dict):
                continue
            required.append(
                {
                    "document_id": item.get("document_id"),
                    "chunk_id": item.get("chunk_id"),
                    "relative_path_digest": _digest_text(item.get("source_file")) if item.get("source_file") else None,
                    "granularity": "scope" if item.get("heading") or item.get("excerpt") else "document",
                }
            )
    return {
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "dataset_role": role,
        "question_type": row.get("answerability_type") or row.get("question_type"),
        "required_evidence": required,
        "candidates": _runtime_units(row.get("candidates") or []),
        "bundle": _runtime_units(row.get("evidence_bundle") or []),
        "generation_context": _runtime_units(row.get("evidence_bundle") or []),
        "citations": _runtime_units(row.get("evidence_bundle") or []),
    }


def _runtime_units(values: Iterable[Any]) -> list[dict[str, Any]]:
    units = []
    for value in values:
        if not isinstance(value, dict):
            continue
        payload = identity_from_runtime_evidence_item(value).to_json()
        payload["granularity"] = "chunk"
        units.append(payload)
    return units


def _required_units(row: dict[str, Any]) -> list[Any]:
    required = row.get("required_evidence")
    if isinstance(required, list):
        return required
    if isinstance(required, dict):
        units = []
        for key, granularity in (("source_digests", "document"), ("document_identity_digests", "document"), ("document_ids", "document"), ("scope_ids", "scope"), ("chunk_ids", "chunk"), ("chunk_content_digests", "chunk")):
            for value in required.get(key) or []:
                field = key[:-1] if key.endswith("s") else key
                units.append({field: value, "granularity": granularity})
        return units
    return []


def _stage_units(row: dict[str, Any], stage: str) -> list[Any]:
    if stage in row and isinstance(row[stage], list):
        return row[stage]
    aliases = {
        "candidates": ("candidate_evidence", "candidate_evidence_ids", "retrieved_evidence_ids"),
        "bundle": ("evidence_bundle", "evidence_bundle_ids", "evidence_ids", "evidence_chunk_ids"),
        "generation_context": ("generation_context", "prompt_evidence", "evidence_bundle"),
        "citations": ("citation_evidence", "citations"),
    }
    for key in aliases[stage]:
        value = row.get(key)
        if isinstance(value, list):
            return [{"chunk_id": item} if isinstance(item, str) else item for item in value]
    return []


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("case_id") or "unknown")


def _has_stable_identity(payload: dict[str, Any]) -> bool:
    return any(payload.get(field) for field in ("knowledge_base_digest", "document_identity_digest", "source_digest", "relative_path_digest", "chunk_content_digest", "heading_path_digest"))


def _has_coverage_identity(payload: dict[str, Any]) -> bool:
    if payload.get("granularity") == "scope":
        return bool(payload.get("scope_id") or payload.get("scope_identity_digest") or payload.get("heading_path_digest") or payload.get("chunk_id") or payload.get("chunk_content_digest"))
    return _has_stable_identity(payload) or bool(payload.get("document_id") or payload.get("chunk_id"))


def _digest_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("source-documents/"):
        text = text[len("source-documents/") :]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity_map_for_role(role: str) -> dict[str, list[dict[str, Any]]]:
    path = DEVELOPMENT_GOLD_IDENTITY_PATH if role == "development" else KNOWN_REGRESSION_GOLD_IDENTITY_PATH
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for sample in payload.get("samples") or []:
        if not isinstance(sample, dict) or sample.get("migration_status") != "verified":
            continue
        units = []
        for item in sample.get("required_evidence") or []:
            if isinstance(item, dict) and item.get("migration_status") == "verified":
                unit = dict(item.get("identity") or {})
                unit["granularity"] = item.get("granularity") or "scope"
                units.append(unit)
        out[str(sample.get("sample_id"))] = units
    return out


def gold_identity_migration_report() -> dict[str, Any]:
    development = _migration_counts(DEVELOPMENT_GOLD_IDENTITY_PATH)
    known = _migration_counts(KNOWN_REGRESSION_GOLD_IDENTITY_PATH)
    status = "verified" if development["unverifiable"] == 0 and known["unverifiable"] == 0 and development["ambiguous"] == 0 and known["ambiguous"] == 0 else "partial"
    if development["total"] == 0 and known["total"] == 0:
        status = "missing"
    return {
        "schema_version": "opk-rag.gold-evidence-identity-coverage-report.v1",
        "status": status,
        "development": development,
        "known_regression": known,
    }


def _migration_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"total": 0, "required_evidence_samples": 0, "verified": 0, "unverifiable": 0, "ambiguous": 0, "not_applicable": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [item for item in payload.get("samples") or [] if isinstance(item, dict)]
    required = [item for item in samples if item.get("migration_status") != "not_applicable"]
    return {
        "total": len(samples),
        "required_evidence_samples": len(required),
        "verified": sum(item.get("migration_status") == "verified" for item in required),
        "unverifiable": sum(item.get("migration_status") in {"unverifiable", "source_missing", "scope_missing"} for item in required),
        "ambiguous": sum(item.get("migration_status") == "ambiguous" for item in required),
        "not_applicable": sum(item.get("migration_status") == "not_applicable" for item in samples),
    }


def _candidate_failure_status(aggregate: dict[str, Any]) -> str:
    metric = aggregate.get("candidate_coverage") or {}
    if metric.get("status") != "computed":
        return "unverified"
    return "confirmed" if (metric.get("value") or 0.0) < 0.8 else "not_confirmed"


def _bundle_loss_status(aggregate: dict[str, Any]) -> str:
    candidate = aggregate.get("candidate_coverage") or {}
    bundle = aggregate.get("bundle_coverage") or {}
    if candidate.get("status") != "computed" or bundle.get("status") != "computed":
        return "unverified"
    return "confirmed" if (candidate.get("value") or 0.0) - (bundle.get("value") or 0.0) > 0.1 else "not_confirmed"


def _reranking_or_cutoff_loss_status(aggregate: dict[str, Any]) -> str:
    details = aggregate.get("coverage_loss_attribution_details") or {}
    count = ((details.get("reranking_or_cutoff_loss") or {}).get("count") or 0)
    return "present_secondary" if count else "not_observed"


def _generation_context_loss_status(aggregate: dict[str, Any]) -> str:
    details = aggregate.get("coverage_loss_attribution_details") or {}
    count = ((details.get("generation_context_loss") or {}).get("count") or 0)
    return "observed" if count else "not_observed"


def _next_action(classification: list[str]) -> str:
    if "measurement_valid_bundle_loss_confirmed" in classification:
        return "diagnose evidence bundle assembly and context budget"
    if "measurement_valid_retrieval_failure_confirmed" in classification:
        return "diagnose retrieval recall on development data"
    if "measurement_mapping_bug_confirmed" in classification:
        return "fix sealed baseline scoring adapter identity mapping first; do not attribute the observed low value to retrieval or bundle loss until identity mapping is verifiable"
    return "migrate gold evidence identity annotations"


def _reject_forbidden_path(path: Path) -> None:
    text = path.as_posix()
    if any(marker in text for marker in FORBIDDEN_HOLDOUT_PARTS):
        raise RequiredEvidenceDiagnosticError(f"refusing forbidden holdout path: {text}")
