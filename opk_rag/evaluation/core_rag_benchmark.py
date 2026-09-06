from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from opk_rag.runtime.dotenv import load_project_env


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ID = "core-rag-benchmark-v1"
BENCHMARK_FAMILY = "core-rag"
BENCHMARK_VERSION = "1.0.0"
BENCHMARK_DIR = ROOT / "evaluation-data" / BENCHMARK_ID
RESULTS_DIR = ROOT / "evaluation-data" / "results" / BENCHMARK_ID
ANNOTATION_REVIEW_REPORT = BENCHMARK_DIR / "annotation_review_report.json"
OWNER_REVIEW_PACKAGE = BENCHMARK_DIR / "annotation_owner_review_package.jsonl"
OWNER_REVIEW_DECISIONS = BENCHMARK_DIR / "annotation_owner_review.jsonl"
OWNER_REVIEW_VALIDATION = BENCHMARK_DIR / "annotation_owner_review_validation.json"

MANIFEST_SCHEMA = "opk-rag.core-rag-benchmark-manifest.v1"
CORPUS_BINDING_SCHEMA = "opk-rag.core-rag-corpus-binding.v1"
CHUNKING_BINDING_SCHEMA = "opk-rag.core-rag-chunking-binding.v1"
QUESTION_SCHEMA = "opk-rag.core-rag-question.v1"
ANNOTATION_SCHEMA = "opk-rag.core-rag-annotation.v1"
OWNER_REVIEW_PACKAGE_SCHEMA = "opk-rag.core-rag-owner-review-package.v1"
OWNER_REVIEW_DECISION_SCHEMA = "opk-rag.core-rag-owner-review-decision.v1"
SPLITS_SCHEMA = "opk-rag.core-rag-splits.v1"
RUNTIME_SCHEMA = "opk-rag.core-rag-reference-runtime.v1"
REPORT_CONTRACT_SCHEMA = "opk-rag.core-rag-report-contract.v1"
RUN_REPORT_SCHEMA = "opk-rag.core-rag-baseline-report.v1"

DEV_SPLIT = "development"
KNOWN_REGRESSION_SPLIT = "known-regression"
VALID_SPLITS = {DEV_SPLIT, KNOWN_REGRESSION_SPLIT}
PUBLICATION_STATES = {"implementation_complete", "acceptance_blocked", "published"}
EXPECTED_DEEPSEEK_MODEL = "deepseek-v4-flash"
EXPECTED_SUPABASE_HOST_SUFFIX = ".supabase.co"
EXPECTED_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EXPECTED_EMBEDDING_DIMENSION = 1024
EXPECTED_NORMALIZATION_POLICY = "l2"
EXPECTED_DISTANCE_METRIC = "cosine"
ANSWERABLE_LABELS = {"answerable", "partially_answerable"}
UNANSWERABLE_LABELS = {"unanswerable"}
EXPECTED_ACTION_BY_LABEL = {
    "answerable": "answer",
    "partially_answerable": "partial_answer",
    "unanswerable": "abstain",
}
EXPECTED_ACTIONS_BY_LABEL = {
    "answerable": {"answer"},
    "partially_answerable": {"partial_answer"},
    "unanswerable": {"abstain", "correct_premise"},
}
QUESTION_TYPE_MAP = {
    "fully_answerable": "fully_answerable",
    "partially_answerable": "partially_answerable",
    "no_evidence": "no_evidence",
    "related_topic_missing_fact": "no_evidence",
    "false_premise": "false_premise",
    "unsupported_inference": "no_evidence",
    "conflicting_evidence": "conflicting_evidence",
}
REFUSAL_REASON_BY_TYPE = {
    "no_evidence": "no_evidence",
    "related_topic_missing_fact": "missing_fact",
    "false_premise": "false_premise",
    "unsupported_inference": "unsupported_inference",
    "conflicting_evidence": "conflicting_evidence",
}
SECRET_PATTERNS = (
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
    re.compile(r"authorization\s*:\s*bearer\s+[A-Za-z0-9_\-.]{12,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@", re.IGNORECASE),
)
PRIVATE_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|data|mnt|Volumes|var|private)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
)
RAW_PRIVATE_KEYS = {
    "content",
    "raw_content",
    "document_content",
    "supporting_excerpt",
    "raw_generation_text",
    "raw_output",
    "answer",
    "prompt",
    "messages",
    "authorization",
    "api_key",
    "token",
    "password",
}

OWNER_REVIEW_DECISIONS_ALLOWED = {"approve", "approve_with_edits", "reject_annotation", "needs_source_check", "pending"}
OWNER_REVIEW_BLOCKING_DECISIONS = {"reject_annotation", "needs_source_check", "pending"}
OWNER_REVIEW_LEAKAGE_KEYS = {
    "model_answer",
    "model_score",
    "baseline_pass_fail",
    "pass_fail",
    "final_action",
    "aggregate_accuracy",
    "accuracy",
    "generated_answer",
}
PROVENANCE_CATEGORY_MAP = {
    "migrated_unchanged": "migrated_unchanged",
    "migrated_unchanged_with_normalized_field_names": "migrated_normalized",
    "deterministically_derived_from_authoritative_source_locator": "deterministic_locator_derived",
    "newly_inferred_semantic_annotation": "newly_inferred_semantic",
}
CLAIM_PLACEHOLDER_VALUES = {None, "", "unknown", "todo", "tbd", "placeholder"}

CORE_FILES = {
    "corpus_binding": "corpus_binding.json",
    "chunking_binding": "chunking_binding.json",
    "question_set": "question_set.jsonl",
    "annotations": "annotations.jsonl",
    "splits": "splits.json",
    "reference_runtime_v1": "reference_runtime_v1.json",
    "report_contract": "report_contract.json",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    status: str
    exit_code: int
    publication_state: str | None
    issues: tuple[Issue, ...]
    benchmark_manifest_file_sha256: str | None = None
    benchmark_manifest_canonical_digest: str | None = None
    privacy_scan: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "publication_state": self.publication_state,
            "publication_eligible": self.status == "valid" and self.publication_state == "published",
            "benchmark_manifest_file_sha256": self.benchmark_manifest_file_sha256,
            "benchmark_manifest_canonical_digest": self.benchmark_manifest_canonical_digest,
            "privacy_scan": self.privacy_scan,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{relative_path(path)}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{relative_path(path)}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


def repo_relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    return relative_path(resolved) if resolved.is_relative_to(ROOT) else resolved.as_posix()


def core_path(name: str) -> Path:
    return BENCHMARK_DIR / CORE_FILES[name]


def build_all_artifacts(*, publication_state: str = "implementation_complete") -> dict[str, Any]:
    if publication_state not in PUBLICATION_STATES:
        raise ValueError(f"invalid publication_state: {publication_state}")
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    dev_rows = read_jsonl(ROOT / "evaluation-data" / "answerability_dev.jsonl")
    known_rows = read_jsonl(ROOT / "evaluation-data" / "answerability_test.jsonl")
    dev_gold = _gold_by_sample(ROOT / "evaluation-data" / "diagnostics" / "phase2_development_gold_identity_v1.json")
    known_gold_all = _gold_by_sample(ROOT / "evaluation-data" / "diagnostics" / "phase2_known_regression_gold_identity_v1.json")

    normalized = []
    annotations = []
    for source_row in dev_rows:
        normalized.append(_normalize_question(source_row, dataset_role=DEV_SPLIT))
        annotations.append(_normalize_annotation(source_row, dataset_role=DEV_SPLIT, gold_record=dev_gold.get(source_row["id"])))
    for source_row in known_rows:
        normalized.append(_normalize_question(source_row, dataset_role=KNOWN_REGRESSION_SPLIT))
        annotations.append(_normalize_annotation(source_row, dataset_role=KNOWN_REGRESSION_SPLIT, gold_record=known_gold_all.get(source_row["id"])))

    write_json(core_path("corpus_binding"), _build_corpus_binding())
    write_json(core_path("chunking_binding"), _build_chunking_binding())
    write_jsonl(core_path("question_set"), normalized)
    write_jsonl(core_path("annotations"), annotations)
    write_json(core_path("splits"), _build_splits(normalized))
    write_json(core_path("reference_runtime_v1"), _build_reference_runtime())
    write_json(core_path("report_contract"), _build_report_contract())
    write_json(ANNOTATION_REVIEW_REPORT, _build_annotation_review_report(annotations))
    manifest = _build_manifest(publication_state=publication_state)
    write_json(BENCHMARK_DIR / "benchmark_manifest.json", manifest)
    _update_registry(manifest)
    return manifest


def verify_core_benchmark() -> VerificationResult:
    issues: list[Issue] = []
    manifest_path = BENCHMARK_DIR / "benchmark_manifest.json"
    if not manifest_path.exists():
        return VerificationResult(
            "invalid",
            1,
            None,
            (Issue("error", "manifest_missing", "Core RAG benchmark manifest is missing.", relative_path(manifest_path)),),
        )

    manifest = read_json(manifest_path)
    manifest_file_sha = file_digest(manifest_path)
    manifest_canonical_digest = canonical_manifest_digest(manifest)
    publication_state = manifest.get("publication_state")
    _validate_manifest(manifest, issues)
    question_rows = _load_required_jsonl("question_set", issues)
    annotation_rows = _load_required_jsonl("annotations", issues)
    splits = _load_required_json("splits", issues)
    annotation_review = read_json(ANNOTATION_REVIEW_REPORT) if ANNOTATION_REVIEW_REPORT.exists() else None
    _validate_questions(question_rows, issues)
    _validate_annotations(question_rows, annotation_rows, issues)
    if isinstance(splits, dict):
        _validate_splits(question_rows, splits, issues)
    _validate_component_digests(manifest, issues)
    if not annotation_review:
        issues.append(Issue("error", "annotation_review_report_missing", "Annotation review report is missing.", relative_path(ANNOTATION_REVIEW_REPORT)))
    elif publication_state == "published" and annotation_review.get("status") != "owner_review_complete":
        issues.append(Issue("error", "owner_annotation_review_required", "Published benchmark requires completed owner annotation review.", relative_path(ANNOTATION_REVIEW_REPORT)))
    _validate_registry(issues)
    privacy_scan = scan_core_artifact_privacy()
    for finding in privacy_scan["findings"]:
        issues.append(Issue("error", "privacy_scan_failure", finding["message"], finding["path"]))
    if privacy_scan["status"] != "pass":
        issues.append(Issue("error", "publication_eligible_false_privacy", "Privacy scan failure makes publication ineligible."))

    for command, code in (
        (["uv", "run", "python", "scripts/manage_phase2_corpus_snapshot.py", "verify", "--json"], "corpus_snapshot_gate"),
        (["uv", "run", "python", "scripts/reconcile_phase2_chunking_and_source_spans.py", "verify", "--json"], "chunking_source_span_gate"),
    ):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            issues.append(Issue("error", code, result.stdout.strip() or result.stderr.strip()))

    status = "valid" if not any(issue.severity == "error" for issue in issues) else "invalid"
    return VerificationResult(status, 0 if status == "valid" else 1, publication_state, tuple(issues), manifest_file_sha, manifest_canonical_digest, privacy_scan)


def run_official_baseline(
    *,
    benchmark_id: str,
    benchmark_version: str,
    split: str,
    reference_runtime: str,
    output_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    load_result = load_project_env()
    if benchmark_id != BENCHMARK_ID:
        raise ValueError(f"unsupported benchmark id: {benchmark_id}")
    if benchmark_version != BENCHMARK_VERSION:
        raise ValueError(f"unsupported benchmark version: {benchmark_version}")
    if reference_runtime != "v1":
        raise ValueError(f"unsupported reference runtime: {reference_runtime}")
    if split not in VALID_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    verification = verify_core_benchmark()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = read_json(core_path("reference_runtime_v1"))
    rows = [row for row in read_jsonl(core_path("question_set")) if row["dataset_role"] == split]
    run_id = run_id or f"{BENCHMARK_ID}-{split}-{utc_now().replace(':', '').replace('-', '')}"
    run_dir = output_root / "candidates" / run_id
    if run_dir.exists():
        raise FileExistsError(f"candidate run directory already exists: {repo_relative_or_absolute(run_dir)}")
    run_dir.mkdir(parents=True)
    runtime_validation = validate_reference_runtime()
    blockers = list(runtime_validation["failures"])
    sample_rows = []
    legacy_report: dict[str, Any] | None = None
    legacy_result_path: str | None = None
    legacy_report_path: str | None = None
    legacy_markdown_path: str | None = None
    runner_exit_code: int | None = None
    if blockers:
        status = "acceptance_blocked"
        sample_rows = []
    else:
        legacy = _run_legacy_answerability_baseline(split=split, output_root=Path("/tmp/opk-rag-core-benchmark-private-runs"), run_id=run_id)
        runner_exit_code = legacy["exit_code"]
        legacy_report = read_json(legacy["report_json"])
        legacy_result_path = repo_relative_or_absolute(legacy["results"])
        legacy_report_path = repo_relative_or_absolute(legacy["report_json"])
        legacy_markdown_path = repo_relative_or_absolute(legacy["markdown_report"])
        legacy_results = read_jsonl(legacy["results"])
        annotations_by_id = {row["sample_id"]: row for row in read_jsonl(core_path("annotations"))}
        sample_rows = [
            sanitize_report_payload(_core_sample_result(row, annotation=annotations_by_id[row["id"]], split=split, run_id=run_id))
            for row in legacy_results
        ]
        blockers = _blockers_from_legacy_run(legacy_report, runner_exit_code=runner_exit_code)
        status = "completed" if not blockers else "acceptance_blocked"

    runtime_fingerprints = build_runtime_fingerprints(runtime_validation)
    aggregate = {
        "schema_version": "opk-rag.core-rag-baseline-aggregate.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "split": split,
        "run_id": run_id,
        "run_output_dir": repo_relative_or_absolute(run_dir),
        "status": status,
        "publication_eligible": status == "completed" and not blockers,
        "reference_runtime": "core-rag-reference-runtime-v1",
        "verification_status": verification.status,
        "runtime_integrity": runtime_validation,
        "sample_count": len(sample_rows),
        "completed_sample_count": sum(1 for row in sample_rows if row["execution_status"] == "completed"),
        "infrastructure_failure_count": sum(1 for row in sample_rows if row["execution_status"] == "infrastructure_failure"),
        "infrastructure_failures": blockers,
        "legacy_runner_exit_code": runner_exit_code,
        "legacy_results_path": legacy_result_path,
        "legacy_report_path": legacy_report_path,
        "legacy_markdown_report_path": legacy_markdown_path,
        "metrics": _compute_core_baseline_metrics(sample_rows),
        "environment_load": {
            "found": load_result.found,
            "loaded": load_result.loaded,
            "loaded_keys": list(load_result.loaded_keys),
            "skipped_existing_keys": list(load_result.skipped_existing_keys),
        },
        "remaining_acceptance_command": _official_command(split, output_root),
        "runtime_manifest_digest": file_digest(core_path("reference_runtime_v1")),
        "benchmark_manifest_file_sha256": verification.benchmark_manifest_file_sha256,
        "benchmark_manifest_canonical_digest": verification.benchmark_manifest_canonical_digest,
        "runtime_fingerprints": runtime_fingerprints,
        "privacy_scan": {"status": "pending_until_files_written"},
        "generated_at": utc_now(),
    }
    aggregate["publication_eligible"] = aggregate["publication_eligible"] and verification.status == "valid" and _owner_annotation_review_complete()
    report = _markdown_baseline_report(aggregate, runtime_manifest)

    write_jsonl(run_dir / f"{split}_baseline_results.jsonl", sample_rows)
    write_json(run_dir / f"{split}_baseline_aggregate.json", sanitize_report_payload(aggregate))
    write_json(run_dir / f"{split}_reference_runtime_v1.json", sanitize_report_payload(runtime_manifest))
    (run_dir / f"{split}_baseline_report.md").write_text(sanitize_markdown(report), encoding="utf-8")
    privacy_scan = scan_paths_for_privacy([run_dir])
    aggregate["privacy_scan"] = privacy_scan
    aggregate["publication_eligible"] = aggregate["publication_eligible"] and privacy_scan["status"] == "pass" and _owner_annotation_review_complete()
    write_json(run_dir / f"{split}_baseline_aggregate.json", sanitize_report_payload(aggregate))
    return aggregate


def _gold_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {sample["sample_id"]: sample for sample in payload.get("samples", [])}


def _normalize_question(row: dict[str, Any], *, dataset_role: str) -> dict[str, Any]:
    sample_id = row["id"]
    source_split = row.get("split")
    return {
        "schema_version": QUESTION_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "sample_id": sample_id,
        "source_sample_id": sample_id,
        "legacy_eval_id": row.get("legacy_eval_id"),
        "dataset_role": dataset_role,
        "split_id": dataset_role,
        "source_split": source_split,
        "question": row["question"],
        "question_digest": text_digest(row["question"]),
        "answerability_label": _answerability_label(row),
        "expected_action": _expected_action(row),
        "question_type": _question_type(row),
        "source_corpus_id": "phase2-corpus-v1",
        "source_corpus_digest": _source_corpus_digest(),
        "corpus_version": "phase2-corpus-v1",
    }


def _normalize_annotation(row: dict[str, Any], *, dataset_role: str, gold_record: dict[str, Any] | None) -> dict[str, Any]:
    answerability = _answerability_label(row)
    question_type = _question_type(row)
    required_evidence = [] if answerability in UNANSWERABLE_LABELS else _required_evidence(gold_record)
    required_claims = [
        {
            "claim_id": f"{row['id']}-required-claim-{index:02d}",
            "text": text,
            "provenance": _provenance("legacy_required_facts", review_status="owner_review_required", classification="newly_inferred_semantic_annotation"),
        }
        for index, text in enumerate(row.get("required_facts") or [], start=1)
    ]
    required_claim_state = _claim_field_state(row, "required_claims", "required_facts")
    forbidden_claims = [
        {
            "claim_id": f"{row['id']}-forbidden-claim-{index:02d}",
            "text": text,
            "provenance": _provenance("legacy_unsupported_subquestions", review_status="owner_review_required", classification="newly_inferred_semantic_annotation"),
        }
        for index, text in enumerate(row.get("unsupported_subquestions") or [], start=1)
    ]
    forbidden_claim_state = _claim_field_state(row, "forbidden_claims", "unsupported_subquestions")
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "sample_id": row["id"],
        "source_sample_id": row["id"],
        "dataset_role": dataset_role,
        "split_id": dataset_role,
        "question": row["question"],
        "question_digest": text_digest(row["question"]),
        "answerability_label": answerability,
        "expected_action": _expected_action(row),
        "question_type": question_type,
        "source_corpus_id": "phase2-corpus-v1",
        "source_corpus_digest": _source_corpus_digest(),
        "required_evidence": required_evidence,
        "required_claims": required_claims,
        "forbidden_claims": forbidden_claims,
        "claim_field_states": {
            "required_claims": required_claim_state,
            "forbidden_claims": forbidden_claim_state,
        },
        "citation_policy": {
            "policy_id": "core-rag-citation-policy-v1",
            "requires_citations_for_substantive_answers": True,
            "allowed_citation_format": "[C<number>]",
            "source": "phase2-dogfooding-scoring-v1",
        },
        "grounding_policy": {
            "policy_id": "core-rag-grounding-policy-v1",
            "unsupported_material_claims_allowed": False,
            "source": "phase2-dogfooding-scoring-v1",
        },
        "annotation_provenance": {
            "source_dataset": "evaluation-data/answerability_dev.jsonl"
            if dataset_role == DEV_SPLIT
            else "evaluation-data/answerability_test.jsonl",
            "source_record_digest": stable_hash(row),
            "normalization_method": "legacy-answerability-to-core-rag-v1",
            "semantic_mutation": False,
            "classification": "migrated_unchanged_with_normalized_field_names",
            "review_status": "owner_review_required",
            "reviewer_type": None,
            "reviewer_independence": "not_independent",
            "reviewed_at": None,
        },
    }
    if answerability in UNANSWERABLE_LABELS:
        annotation["refusal_reason"] = {
            "reason_code": REFUSAL_REASON_BY_TYPE.get(row.get("answerability_type"), "no_evidence"),
            "source_reason": row.get("expected_reason"),
            "expected_behavior": "abstain_without_substantive_answer",
            "provenance": _provenance("legacy_expected_reason_and_answerability_type", review_status="owner_review_required", classification="newly_inferred_semantic_annotation"),
        }
    else:
        annotation["refusal_reason"] = None
    return annotation


def _required_evidence(gold_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not gold_record:
        return []
    required = []
    for index, item in enumerate(gold_record.get("required_evidence") or [], start=1):
        required.append(
            {
                "evidence_id": f"{gold_record['sample_id']}-gold-{index:02d}",
                "granularity": item.get("granularity"),
                "identity": item.get("identity") or {},
                "legacy_source_reference": item.get("legacy_source_reference"),
                "provenance": {
                    "origin": "phase2_gold_identity_map",
                    "migration_method": item.get("migration_method"),
                    "classification": "deterministically_derived_from_authoritative_source_locator",
                    "locator_status": "migrated",
                    "migration_status": item.get("migration_status"),
                    "migration_confidence": item.get("migration_confidence"),
                    "review_status": "owner_review_required",
                    "reviewed_at": None,
                    "reviewer_type": None,
                    "reviewer_independence": item.get("reviewer_independence"),
                    "review_reason": item.get("review_reason"),
                },
            }
        )
    return required


def _provenance(origin: str, *, review_status: str, classification: str) -> dict[str, Any]:
    return {
        "origin": origin,
        "migration_method": "legacy-answerability-normalization",
        "classification": classification,
        "review_status": review_status,
        "reviewer_type": None if review_status == "owner_review_required" else "agent",
        "reviewer_independence": "not_independent",
        "reviewed_at": None,
        "semantic_mutation": False,
    }


def _claim_field_state(row: dict[str, Any], field_name: str, legacy_field: str) -> dict[str, Any]:
    values = row.get(legacy_field) or []
    empty = len(values) == 0
    return {
        "field_path": field_name,
        "field_state": "empty_by_design" if empty else "concrete_semantic_annotation",
        "empty_reason": f"Authoritative legacy `{legacy_field}` collection is empty; no {field_name} are defined for this sample."
        if empty
        else None,
        "not_applicable": empty,
        "provenance_source": {
            "source_artifact": "evaluation-data/answerability_dev.jsonl"
            if row.get("split") == "dev"
            else "evaluation-data/answerability_test.jsonl",
            "source_field": legacy_field,
            "source_record_digest": stable_hash(row),
        },
        "semantic_meaning_changed": False,
        "review_visibility": "sample_level_owner_review",
        "owner_input_required": False,
    }


def _answerability_label(row: dict[str, Any]) -> str:
    value = row.get("expected_answerability")
    if value in {"answerable", "partially_answerable", "unanswerable"}:
        return value
    raise ValueError(f"{row.get('id')}: unsupported expected_answerability {value!r}")


def _expected_action(row: dict[str, Any]) -> str:
    source_action = row.get("expected_action")
    if source_action in {"answer", "partial_answer", "abstain", "correct_premise"}:
        return source_action
    return EXPECTED_ACTION_BY_LABEL[_answerability_label(row)]


def _question_type(row: dict[str, Any]) -> str:
    value = row.get("answerability_type")
    if value not in QUESTION_TYPE_MAP:
        raise ValueError(f"{row.get('id')}: unsupported answerability_type {value!r}")
    return QUESTION_TYPE_MAP[value]


def _source_corpus_digest() -> str:
    return read_json(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json")["source_content_digest"]


def _build_corpus_binding() -> dict[str, Any]:
    snapshot_path = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
    snapshot = read_json(snapshot_path)
    return {
        "schema_version": CORPUS_BINDING_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "corpus_binding_id": "core-rag-benchmark-v1-corpus-binding",
        "source_corpus_id": "phase2-corpus-v1",
        "source_snapshot_path": relative_path(snapshot_path),
        "source_snapshot_digest": file_digest(snapshot_path),
        "source_content_digest": snapshot["source_content_digest"],
        "markdown_file_count": snapshot["markdown_file_count"],
        "total_byte_count": snapshot["total_byte_count"],
        "private_manifest_required": snapshot["private_manifest_required"],
        "private_manifest_present_at_freeze": snapshot["private_manifest_present_at_freeze"],
        "indexed_active_document_count": snapshot.get("indexed_active_document_count"),
        "indexed_chunk_count": snapshot.get("indexed_chunk_count"),
        "indexed_corpus_status": snapshot.get("indexed_corpus_status"),
        "binding_policy": "reuse-phase2-corpus-v1-without-content-mutation",
        "drift_gate": "uv run python scripts/manage_phase2_corpus_snapshot.py verify --json",
    }


def _build_chunking_binding() -> dict[str, Any]:
    contract_path = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_contract_v1.json"
    report_path = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_report_v1.json"
    span_contract_path = ROOT / "evaluation-data" / "diagnostics" / "phase2_source_evidence_span_contract_v1.json"
    span_report_path = ROOT / "evaluation-data" / "diagnostics" / "phase2_source_evidence_span_resolution_v1.json"
    report = read_json(report_path)
    span_contract = read_json(span_contract_path)
    return {
        "schema_version": CHUNKING_BINDING_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "chunking_binding_id": "core-rag-benchmark-v1-production-chunking-binding",
        "production_chunking_contract": {
            "parser": "opk_rag.chunking.parser.parse_markdown_file",
            "chunker": "opk_rag.chunking.chunker.chunk_markdown_document",
            "config": {"target_size": 1200, "max_size": 1800, "overlap": 0},
        },
        "phase2_contract_path": relative_path(contract_path),
        "phase2_contract_digest": file_digest(contract_path),
        "phase2_report_path": relative_path(report_path),
        "phase2_report_digest": file_digest(report_path),
        "source_span_contract_path": relative_path(span_contract_path),
        "source_span_contract_digest": file_digest(span_contract_path),
        "source_span_report_path": relative_path(span_report_path),
        "source_span_report_digest": file_digest(span_report_path),
        "production_chunk_count": report.get("production_chunk_count"),
        "shadow_c0_chunk_count": report.get("shadow_c0_chunk_count"),
        "production_searchable_document_count": report.get("production_searchable_document_count"),
        "legacy_missing_line_range_chunk_count": report.get("legacy_missing_line_range_chunk_count"),
        "required_evidence_unit_count": span_contract.get("required_evidence_unit_count"),
        "drift_gate": "uv run python scripts/reconcile_phase2_chunking_and_source_spans.py verify --json",
    }


def _build_splits(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows: dict[str, list[str]] = {DEV_SPLIT: [], KNOWN_REGRESSION_SPLIT: []}
    for row in question_rows:
        split_rows[row["dataset_role"]].append(row["sample_id"])
    return {
        "schema_version": SPLITS_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "split_manifest_id": "core-rag-benchmark-v1-splits",
        "roles": {
            DEV_SPLIT: {
                "source": "evaluation-data/answerability_dev.jsonl",
                "semantic_role": "public/exposed diagnostics and implementation work",
                "sample_ids": split_rows[DEV_SPLIT],
            },
            KNOWN_REGRESSION_SPLIT: {
                "source": "evaluation-data/answerability_test.jsonl",
                "semantic_role": "public/exposed known regression prevention; not sealed test",
                "sample_ids": split_rows[KNOWN_REGRESSION_SPLIT],
            },
        },
        "governance": {
            "acceptance_holdout_present": False,
            "known_regression_mutation_requires_version_record": True,
            "holdout_v1_excluded_after_exposure": True,
        },
    }


def _build_reference_runtime() -> dict[str, Any]:
    embedding_contract = {
        "embedding_provider": "local_qwen",
        "embedding_model": EXPECTED_EMBEDDING_MODEL,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "normalization_policy": EXPECTED_NORMALIZATION_POLICY,
        "distance_metric": EXPECTED_DISTANCE_METRIC,
        "input_template_version": "embedding-input-v1",
    }
    index_contract = {
        "retrieval_backend": "remote_supabase_postgresql_pgvector",
        "knowledge_base": "phase2-production-knowledge-base",
        **embedding_contract,
    }
    return {
        "schema_version": RUNTIME_SCHEMA,
        "runtime_id": "core-rag-reference-runtime-v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "runtime_contract": "Local Markdown Vault + Remote Supabase PostgreSQL/pgvector + Remote DeepSeek",
        "required_components": {
            "local_markdown_vault": {
                "required": True,
                "configuration_env_names": ["OPK_RAG_VAULT_PATH"],
                "corpus_id": "phase2-corpus-v1",
                "source_content_digest": _source_corpus_digest(),
            },
            "remote_supabase_postgresql_pgvector": {
                "required": True,
                "configuration_env_names": [
                    "SUPABASE_URL",
                    "SUPABASE_SERVICE_ROLE_KEY",
                    "OPK_RAG_DATABASE_URL",
                ],
                "schema_contract_sources": [
                    "supabase/migrations",
                    "evaluation-data/dogfooding/phase2_corpus_snapshot_public.json",
                ],
                "secrets_stored": False,
                "deployment_role": "remote_supabase_project",
                "retrieval_backend": "PostgreSQL + pgvector",
                "knowledge_base_id": "phase2-production-knowledge-base",
                "embedding_contract": embedding_contract,
                "index_configuration_fingerprint": stable_hash(index_contract),
            },
            "remote_deepseek": {
                "required": True,
                "configuration_env_names": [
                    "OPK_RAG_LLM_BASE_URL",
                    "OPK_RAG_LLM_API_KEY",
                    "OPK_RAG_LLM_MODEL",
                    "OPK_RAG_LLM_ALLOW_REMOTE",
                    "OPK_RAG_LLM_RESPONSE_FORMAT",
                    "OPK_RAG_LLM_THINKING_MODE",
                ],
                "provider": "DeepSeek OpenAI-compatible",
                "expected_model_id": EXPECTED_DEEPSEEK_MODEL,
                "thinking_mode": "disabled",
                "secrets_stored": False,
            },
        },
        "forbidden_official_substitutes": [
            "SQLite",
            "local PostgreSQL",
            "mock provider",
            "local LLM",
            "Ollama",
            "fallback provider",
            "legacy source-documents knowledge base",
            "all-local runtime",
        ],
        "fingerprints": {
            "benchmark_manifest": "deferred-to-benchmark-manifest",
            "corpus_binding": file_digest(core_path("corpus_binding")) if core_path("corpus_binding").exists() else None,
            "chunking_binding": file_digest(core_path("chunking_binding")) if core_path("chunking_binding").exists() else None,
            "scoring_contract": file_digest(ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"),
            "code_revision": _git_head(),
        },
    }


def _build_report_contract() -> dict[str, Any]:
    return {
        "schema_version": REPORT_CONTRACT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "report_contract_id": "core-rag-benchmark-v1-report-contract",
        "required_outputs": [
            "per-sample JSONL",
            "aggregate JSON",
            "Markdown report",
            "non-secret runtime manifest",
        ],
        "required_identity_fields": [
            "benchmark_id",
            "benchmark_version",
            "split",
            "run_id",
            "reference_runtime",
            "benchmark_manifest_file_sha256",
            "benchmark_manifest_canonical_digest",
            "runtime_manifest_digest",
            "runtime_fingerprints",
            "code_revision",
            "dirty_working_tree",
        ],
        "publication_policy": {
            "published_requires_development_zero_infrastructure_failures": True,
            "published_requires_known_regression_zero_infrastructure_failures": True,
            "static_validation_alone_may_publish": False,
        },
        "redaction_policy": {
            "store_env_var_names_only": True,
            "store_secret_values": False,
            "store_private_absolute_vault_paths": False,
            "store_raw_private_document_content": False,
        },
    }


def _build_manifest(*, publication_state: str) -> dict[str, Any]:
    component_paths = {name: core_path(name) for name in CORE_FILES}
    component_digests = {name: file_digest(path) for name, path in component_paths.items()}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_family": BENCHMARK_FAMILY,
        "semantic_version": BENCHMARK_VERSION,
        "publication_state": publication_state,
        "publication_eligible": publication_state == "published",
        "publication_blockers": [] if publication_state == "published" else ["owner_annotation_review_required"],
        "core_capabilities": [
            "retrieval",
            "answerability",
            "grounded_generation",
            "citation_validation",
            "unsupported_claim_validation",
            "partial_answer",
            "abstention",
            "corpus_parity",
            "chunking_parity",
        ],
        "future_extension_boundaries": {
            "Rerank": "separate benchmark family or explicit future version required",
            "Agent RAG": "separate benchmark family or explicit future version required",
            "GraphRAG": "separate benchmark family or explicit future version required",
            "Agent GraphRAG": "separate benchmark family or explicit future version required",
        },
        "component_paths": {name: relative_path(path) for name, path in component_paths.items()},
        "component_digests": component_digests,
        "scoring_contract_path": "evaluation-data/dogfooding/phase2_scoring_contract.json",
        "scoring_contract_digest": file_digest(ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"),
        "code_revision": _git_head(),
        "dirty_working_tree": _git_dirty(),
        "digest_semantics": {
            "benchmark_manifest_file_sha256": "SHA-256 of the exact benchmark_manifest.json bytes.",
            "benchmark_manifest_canonical_digest": "SHA-256 of the canonical JSON object parsed from benchmark_manifest.json.",
        },
        "versioning_policy": {
            "patch": "non-semantic metadata, command text, or report formatting only",
            "minor": "Development samples or non-authoritative diagnostics",
            "major_or_new_id": "breaking semantic changes",
            "known_regression_mutation": "requires version update, digest update, and change record",
        },
    }


def _update_registry(manifest: dict[str, Any]) -> None:
    path = ROOT / "evaluation-data" / "benchmark_registry.json"
    registry = read_json(path)
    entry = {
        "benchmark_id": BENCHMARK_ID,
        "role": "known_regression",
        "source_files": [f"evaluation-data/{BENCHMARK_ID}/question_set.jsonl"],
        "exposure_status": "exposed",
        "mutable": False,
        "intended_use": ["core non-agent RAG regression baseline", "future rerank and agentic experiment comparison foundation"],
        "forbidden_use": ["sealed generalization claim", "acceptance holdout substitute"],
        "contains_private_data": False,
        "sample_count": len(read_jsonl(core_path("question_set"))),
        "content_checksum": _content_checksum([core_path("question_set")]),
        "status": manifest["publication_state"],
        "publication_eligible": manifest.get("publication_eligible") is True,
        "semantic_version": BENCHMARK_VERSION,
        "manifest_path": f"evaluation-data/{BENCHMARK_ID}/benchmark_manifest.json",
        "benchmark_manifest_file_sha256": file_digest(BENCHMARK_DIR / "benchmark_manifest.json") if (BENCHMARK_DIR / "benchmark_manifest.json").exists() else None,
        "benchmark_manifest_canonical_digest": canonical_manifest_digest(manifest),
        "notes": "Core RAG Benchmark v1 normalized from exposed Development and Known Regression inputs.",
    }
    benchmarks = [item for item in registry.get("benchmarks", []) if item.get("benchmark_id") != BENCHMARK_ID]
    benchmarks.append(entry)
    registry["benchmarks"] = benchmarks
    registry["generated_at"] = utc_now()
    write_json(path, registry)


def _validate_manifest(manifest: dict[str, Any], issues: list[Issue]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        issues.append(Issue("error", "manifest_schema", "Invalid benchmark manifest schema.", "benchmark_manifest.json"))
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        issues.append(Issue("error", "benchmark_id", "Invalid benchmark id.", "benchmark_manifest.json"))
    if manifest.get("semantic_version") != BENCHMARK_VERSION:
        issues.append(Issue("error", "benchmark_version", "Invalid benchmark version.", "benchmark_manifest.json"))
    if manifest.get("publication_state") not in PUBLICATION_STATES:
        issues.append(Issue("error", "publication_state", "Invalid publication state.", "benchmark_manifest.json"))
    if manifest.get("publication_state") == "published" and manifest.get("publication_eligible") is not True:
        issues.append(Issue("error", "published_not_eligible", "Published state requires publication_eligible=true.", "benchmark_manifest.json"))
    if manifest.get("publication_state") != "published" and manifest.get("publication_eligible") is True:
        issues.append(Issue("error", "blocked_marked_eligible", "Non-published state cannot be publication_eligible.", "benchmark_manifest.json"))


def _load_required_json(name: str, issues: list[Issue]) -> Any | None:
    path = core_path(name)
    if not path.exists():
        issues.append(Issue("error", f"{name}_missing", "Required Core v1 artifact is missing.", relative_path(path)))
        return None
    payload = read_json(path)
    schema_field = {
        "corpus_binding": CORPUS_BINDING_SCHEMA,
        "chunking_binding": CHUNKING_BINDING_SCHEMA,
        "splits": SPLITS_SCHEMA,
        "reference_runtime_v1": RUNTIME_SCHEMA,
        "report_contract": REPORT_CONTRACT_SCHEMA,
        "annotation_review_report": "opk-rag.core-rag-annotation-review-report.v1",
    }.get(name)
    if schema_field and payload.get("schema_version") != schema_field:
        issues.append(Issue("error", f"{name}_schema", "Required artifact has invalid schema.", relative_path(path)))
    return payload


def _load_required_jsonl(name: str, issues: list[Issue]) -> list[dict[str, Any]]:
    path = core_path(name)
    if not path.exists():
        issues.append(Issue("error", f"{name}_missing", "Required Core v1 artifact is missing.", relative_path(path)))
        return []
    rows = read_jsonl(path)
    expected_schema = QUESTION_SCHEMA if name == "question_set" else ANNOTATION_SCHEMA
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != expected_schema:
            issues.append(Issue("error", f"{name}_schema", f"Invalid schema at row {index}.", relative_path(path)))
    return rows


def _validate_questions(rows: list[dict[str, Any]], issues: list[Issue]) -> None:
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        issues.append(Issue("error", "duplicate_sample_id", "Duplicate sample_id exists in question_set.", relative_path(core_path("question_set"))))
    if len(rows) != 40:
        issues.append(Issue("error", "sample_count", f"Expected 40 official samples, found {len(rows)}.", relative_path(core_path("question_set"))))
    questions = [row.get("question") for row in rows]
    if len(questions) != len(set(questions)):
        issues.append(Issue("error", "duplicate_question", "Exact duplicate question text exists.", relative_path(core_path("question_set"))))
    for row in rows:
        for field in (
            "sample_id",
            "dataset_role",
            "split_id",
            "question",
            "question_digest",
            "answerability_label",
            "expected_action",
            "question_type",
            "source_corpus_id",
            "source_corpus_digest",
            "benchmark_version",
        ):
            if field not in row:
                issues.append(Issue("error", "question_required_field", f"{row.get('sample_id')} missing {field}.", relative_path(core_path("question_set"))))
        if row.get("question_digest") != text_digest(str(row.get("question"))):
            issues.append(Issue("error", "question_digest_mismatch", f"{row.get('sample_id')} question digest mismatch.", relative_path(core_path("question_set"))))
        if row.get("dataset_role") not in VALID_SPLITS:
            issues.append(Issue("error", "invalid_dataset_role", f"{row.get('sample_id')} has invalid dataset role.", relative_path(core_path("question_set"))))
        allowed_actions = EXPECTED_ACTIONS_BY_LABEL.get(row.get("answerability_label"), set())
        if row.get("expected_action") not in allowed_actions:
            issues.append(Issue("error", "expected_action_mismatch", f"{row.get('sample_id')} expected action does not match label.", relative_path(core_path("question_set"))))


def _validate_annotations(question_rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]], issues: list[Issue]) -> None:
    question_ids = {row["sample_id"]: row for row in question_rows if "sample_id" in row}
    annotation_ids = {row.get("sample_id") for row in annotation_rows}
    if set(question_ids) != annotation_ids:
        issues.append(Issue("error", "annotation_question_mismatch", "Question and annotation sample_id sets differ.", relative_path(core_path("annotations"))))
    for row in annotation_rows:
        sample_id = row.get("sample_id")
        question = question_ids.get(sample_id)
        if not question:
            continue
        for field in (
            "required_evidence",
            "required_claims",
            "forbidden_claims",
            "citation_policy",
            "grounding_policy",
            "annotation_provenance",
        ):
            if field not in row:
                issues.append(Issue("error", "annotation_required_field", f"{sample_id} missing {field}.", relative_path(core_path("annotations"))))
        if question["answerability_label"] in ANSWERABLE_LABELS and not row.get("required_evidence"):
            issues.append(Issue("error", "gold_evidence_missing", f"{sample_id} answerable sample lacks required evidence.", relative_path(core_path("annotations"))))
        if question["answerability_label"] in UNANSWERABLE_LABELS and not row.get("refusal_reason"):
            issues.append(Issue("error", "refusal_reason_missing", f"{sample_id} unanswerable sample lacks refusal reason.", relative_path(core_path("annotations"))))
        for evidence in row.get("required_evidence") or []:
            provenance = evidence.get("provenance") or {}
            identity = evidence.get("identity") or {}
            if not any(identity.get(key) for key in ("relative_path_digest", "scope_identity_digest", "source_digest", "document_identity_digest", "chunk_content_digest")):
                issues.append(Issue("error", "gold_identity_unstable", f"{sample_id} evidence lacks stable identity.", relative_path(core_path("annotations"))))
            if not provenance.get("origin") or not provenance.get("review_status"):
                issues.append(Issue("error", "gold_provenance_missing", f"{sample_id} evidence lacks provenance.", relative_path(core_path("annotations"))))
        for claim_group in ("required_claims", "forbidden_claims"):
            claims = row.get(claim_group)
            if not isinstance(claims, list):
                issues.append(Issue("error", "claim_collection_not_array", f"{sample_id} {claim_group} must be an array.", relative_path(core_path("annotations"))))
                continue
            state = ((row.get("claim_field_states") or {}).get(claim_group) or {})
            if not claims:
                if state.get("field_state") == "unresolved_missing_annotation":
                    if state.get("owner_input_required") is not True:
                        issues.append(Issue("error", "unresolved_field_not_marked_owner_input_required", f"{sample_id} {claim_group} unresolved field must require owner input.", relative_path(core_path("annotations"))))
                elif state.get("field_state") not in {"empty_by_design", "not_applicable"} or not state.get("empty_reason") or not state.get("provenance_source"):
                    issues.append(Issue("error", "empty_claim_field_metadata_missing", f"{sample_id} {claim_group} empty collection lacks explicit state, reason, or provenance.", relative_path(core_path("annotations"))))
            for claim in claims:
                if _is_placeholder_claim(claim.get("text") if isinstance(claim, dict) else claim):
                    issues.append(Issue("error", "claim_placeholder_value", f"{sample_id} {claim_group} contains a null or placeholder claim.", relative_path(core_path("annotations"))))
                    continue
                if "provenance" not in claim:
                    issues.append(Issue("error", "claim_provenance_missing", f"{sample_id} {claim_group} lacks provenance.", relative_path(core_path("annotations"))))


def _is_placeholder_claim(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in CLAIM_PLACEHOLDER_VALUES
    return False


def _validate_splits(question_rows: list[dict[str, Any]], splits: dict[str, Any], issues: list[Issue]) -> None:
    roles = splits.get("roles") or {}
    expected = {split: [row["sample_id"] for row in question_rows if row.get("dataset_role") == split] for split in VALID_SPLITS}
    for split, sample_ids in expected.items():
        recorded = (roles.get(split) or {}).get("sample_ids")
        if recorded != sample_ids:
            issues.append(Issue("error", "split_manifest_mismatch", f"{split} split sample ids do not match question_set.", relative_path(core_path("splits"))))
    if set(expected[DEV_SPLIT]) & set(expected[KNOWN_REGRESSION_SPLIT]):
        issues.append(Issue("error", "cross_split_id_leakage", "Development and Known Regression share sample ids.", relative_path(core_path("splits"))))


def _validate_component_digests(manifest: dict[str, Any], issues: list[Issue]) -> None:
    component_digests = manifest.get("component_digests") or {}
    for name in CORE_FILES:
        path = core_path(name)
        if path.exists() and component_digests.get(name) != file_digest(path):
            issues.append(Issue("error", "component_digest_mismatch", f"{name} digest mismatch.", "benchmark_manifest.json"))


def _validate_registry(issues: list[Issue]) -> None:
    registry = read_json(ROOT / "evaluation-data" / "benchmark_registry.json")
    entries = [item for item in registry.get("benchmarks", []) if item.get("benchmark_id") == BENCHMARK_ID]
    if len(entries) != 1:
        issues.append(Issue("error", "registry_entry", "Benchmark registry must contain exactly one Core v1 entry.", "evaluation-data/benchmark_registry.json"))
    elif entries[0].get("status") == "published" and entries[0].get("publication_eligible") is not True:
        issues.append(Issue("error", "registry_published_not_eligible", "Published registry entry requires publication_eligible=true.", "evaluation-data/benchmark_registry.json"))


def _validate_no_secrets(issues: list[Issue]) -> None:
    return None


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    return stable_hash(manifest)


def sanitize_report_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in RAW_PRIVATE_KEYS or any(secret in normalized_key for secret in ("secret", "credential", "authorization", "api_key", "password", "token")):
                sanitized[key] = "[REDACTED]"
            elif normalized_key in {"absolute_path", "vault_path", "base_url", "database_url"} or normalized_key.endswith("_path"):
                sanitized[key] = "[REDACTED_PATH]"
            else:
                sanitized[key] = sanitize_report_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_report_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_report_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def sanitize_markdown(text: str) -> str:
    return _redact_text(text)


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    for pattern in PRIVATE_ABSOLUTE_PATH_PATTERNS:
        redacted = pattern.sub("[REDACTED_PRIVATE_PATH]", redacted)
    return redacted


def scan_core_artifact_privacy() -> dict[str, Any]:
    paths = [BENCHMARK_DIR, RESULTS_DIR, ROOT / "docs" / "CORE_RAG_BENCHMARK_V1.md", ROOT / "docs" / "CORE_RAG_BENCHMARK_V1_FREEZE_REPORT.md"]
    return scan_paths_for_privacy(paths)


def scan_paths_for_privacy(paths: Iterable[Path]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".md"})
        elif path.exists() and path.suffix.lower() in {".json", ".jsonl", ".md"}:
            files.append(path)
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"path": repo_relative_or_absolute(path), "code": "secret_like_text", "message": "Artifact contains secret-like text."})
                break
        for pattern in PRIVATE_ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                findings.append({"path": repo_relative_or_absolute(path), "code": "private_absolute_path", "message": "Artifact contains private absolute path text."})
                break
        if path.suffix.lower() in {".json", ".jsonl"}:
            findings.extend(_scan_structured_raw_content(path))
    return {"status": "pass" if not findings else "fail", "file_count": len(set(files)), "findings": findings}


def _scan_structured_raw_content(path: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    rows: list[Any]
    try:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = [read_json(path)]
    except Exception:
        return findings

    def walk(value: Any, breadcrumb: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{breadcrumb}.{key}" if breadcrumb else str(key)
                if str(key).lower() in {"content", "raw_content", "document_content", "supporting_excerpt"} and isinstance(item, str) and len(item.strip()) > 80:
                    findings.append({"path": repo_relative_or_absolute(path), "code": "raw_private_content", "message": f"Artifact contains raw private content at {child}."})
                    return
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value[:50]):
                walk(item, f"{breadcrumb}[{index}]")

    for row in rows[:50]:
        walk(row, "")
    return findings


def validate_reference_runtime() -> dict[str, Any]:
    load_project_env()
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def record(component: str, code: str, passed: bool, **details: Any) -> None:
        payload = {"component": component, "code": code, "passed": passed, **sanitize_report_payload(details)}
        checks.append(payload)
        if not passed:
            failures.append({"component": component, "code": code, **sanitize_report_payload(details)})

    private_manifest_path = ROOT / ".private" / "evaluation" / "phase2_corpus_manifest.json"
    private_manifest = read_json(private_manifest_path) if private_manifest_path.exists() else {}
    vault = private_manifest.get("vault") if isinstance(private_manifest, dict) else {}
    record("local_markdown_vault", "private_corpus_manifest_present", private_manifest_path.exists(), path=".private/evaluation/phase2_corpus_manifest.json")
    record("local_markdown_vault", "corpus_id_phase2_corpus_v1", private_manifest.get("snapshot_id") == "phase2-corpus-v1", observed=private_manifest.get("snapshot_id"))
    record("local_markdown_vault", "frozen_corpus_digest_match", (vault or {}).get("source_content_digest") == _source_corpus_digest(), observed=(vault or {}).get("source_content_digest"), expected=_source_corpus_digest())

    vault_path = os.environ.get("OPK_RAG_VAULT_PATH", "").strip()
    record("local_markdown_vault", "vault_env_present", bool(vault_path), env_name="OPK_RAG_VAULT_PATH")
    if vault_path:
        record("local_markdown_vault", "legacy_source_documents_not_used", "source-documents" not in Path(vault_path).parts, vault_identity_digest=(vault or {}).get("identity_digest"))

    database_url = (os.environ.get("OPK_RAG_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    parsed_db = urlparse(database_url)
    parsed_supabase = urlparse(supabase_url)
    db_host = parsed_db.hostname or ""
    supabase_host = parsed_supabase.hostname or ""
    record("remote_supabase", "database_url_present", bool(database_url), env_name="OPK_RAG_DATABASE_URL or DATABASE_URL")
    record("remote_supabase", "supabase_url_present", bool(supabase_url), env_name="SUPABASE_URL")
    record("remote_supabase", "remote_postgresql", parsed_db.scheme in {"postgres", "postgresql"} and db_host not in {"", "localhost", "127.0.0.1", "::1"}, host_fingerprint=text_digest(db_host) if db_host else None)
    record("remote_supabase", "remote_supabase_project", supabase_host.endswith(EXPECTED_SUPABASE_HOST_SUFFIX), host_fingerprint=text_digest(supabase_host) if supabase_host else None)
    record("remote_supabase", "sqlite_not_active", not database_url.lower().startswith("sqlite"))
    record("remote_supabase", "knowledge_base_not_legacy_source_documents", "source-documents" not in database_url.lower())

    try:
        from opk_rag.embedding.config import build_configuration_fingerprint, load_embedding_config

        embedding_config = load_embedding_config()
        record("embedding", "embedding_model_contract", embedding_config.model_name == EXPECTED_EMBEDDING_MODEL, observed=embedding_config.model_name, expected=EXPECTED_EMBEDDING_MODEL)
        record("embedding", "embedding_dimension_contract", embedding_config.dimension == EXPECTED_EMBEDDING_DIMENSION, observed=embedding_config.dimension, expected=EXPECTED_EMBEDDING_DIMENSION)
        record("embedding", "embedding_normalization_contract", embedding_config.normalize is True, observed=embedding_config.normalize)
        record("embedding", "embedding_distance_metric_contract", embedding_config.distance_metric == EXPECTED_DISTANCE_METRIC, observed=embedding_config.distance_metric, expected=EXPECTED_DISTANCE_METRIC)
        index_fingerprint = build_configuration_fingerprint(embedding_config)
    except Exception as exc:
        index_fingerprint = None
        record("embedding", "embedding_contract_load", False, error_type=type(exc).__name__)

    base_url = os.environ.get("OPK_RAG_LLM_BASE_URL", "").strip()
    model = (os.environ.get("OPK_RAG_LLM_MODEL") or os.environ.get("OPK_RAG_LLM_MODEL_ID") or "").strip()
    allow_remote = os.environ.get("OPK_RAG_LLM_ALLOW_REMOTE", "").strip().lower()
    record("remote_deepseek", "provider_env_present", bool(base_url) and bool(os.environ.get("OPK_RAG_LLM_API_KEY")), env_names=["OPK_RAG_LLM_BASE_URL", "OPK_RAG_LLM_API_KEY"])
    record("remote_deepseek", "remote_endpoint", bool(base_url) and "localhost" not in base_url and "127.0.0.1" not in base_url and "ollama" not in base_url.lower(), base_url_identity=_url_identity(base_url))
    record("remote_deepseek", "remote_allowed", allow_remote in {"1", "true", "yes", "on"}, env_name="OPK_RAG_LLM_ALLOW_REMOTE")
    record("remote_deepseek", "expected_model_identity", model == EXPECTED_DEEPSEEK_MODEL, observed=model or None, expected=EXPECTED_DEEPSEEK_MODEL)
    if base_url and os.environ.get("OPK_RAG_LLM_API_KEY") and model == EXPECTED_DEEPSEEK_MODEL and not failures:
        record("remote_deepseek", "provider_preflight", _deepseek_preflight(base_url, os.environ["OPK_RAG_LLM_API_KEY"], model), model=model)
    else:
        record("remote_deepseek", "provider_preflight", False, skipped_reason="missing_or_invalid_prior_runtime_identity")
    record("reference_runtime", "fallback_provider_not_active", not _env_truthy("OPK_RAG_ENABLE_FALLBACK_PROVIDER") and not _env_truthy("OPK_RAG_LLM_FALLBACK_ENABLED"))

    return {
        "schema_version": "opk-rag.core-rag-runtime-integrity.v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "index_configuration_fingerprint": index_fingerprint,
    }


def _deepseek_preflight(base_url: str, api_key: str, model: str) -> bool:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "Return OK."}], "max_tokens": 1, "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def _url_identity(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {"scheme": parsed.scheme, "host_sha256": text_digest(parsed.hostname or ""), "endpoint_type": "remote" if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} else "loopback"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_runtime_fingerprints(runtime_validation: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(BENCHMARK_DIR / "benchmark_manifest.json")
    return {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_manifest_file_sha256": file_digest(BENCHMARK_DIR / "benchmark_manifest.json"),
        "benchmark_manifest_canonical_digest": canonical_manifest_digest(manifest),
        "corpus_id": "phase2-corpus-v1",
        "corpus_digest": _source_corpus_digest(),
        "chunking_contract_id": "core-rag-benchmark-v1-production-chunking-binding",
        "chunking_contract_digest": file_digest(core_path("chunking_binding")),
        "annotation_digest": file_digest(core_path("annotations")),
        "scoring_contract_identity": {
            "path": "evaluation-data/dogfooding/phase2_scoring_contract.json",
            "sha256": file_digest(ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"),
        },
        "supabase_remote_deployment_role": "remote_supabase_project",
        "schema_or_migration_contract": "supabase/migrations",
        "knowledge_base_fingerprint": text_digest("phase2-production-knowledge-base"),
        "index_configuration_fingerprint": runtime_validation.get("index_configuration_fingerprint"),
        "embedding_model": EXPECTED_EMBEDDING_MODEL,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "normalization_policy": EXPECTED_NORMALIZATION_POLICY,
        "distance_metric": EXPECTED_DISTANCE_METRIC,
        "deepseek_provider": "DeepSeek OpenAI-compatible",
        "deepseek_model_id": EXPECTED_DEEPSEEK_MODEL,
        "thinking_mode": os.environ.get("OPK_RAG_LLM_THINKING_MODE", "disabled"),
        "parameter_policy": "temperature=0, top_p=1, max_tokens=1024, json_schema",
        "prompt_generation_contract": "answer-response-v3+answer-prompt-v4+evidence-context-v1",
        "code_revision": _git_head(),
        "dirty_working_tree": _git_dirty(),
    }


def _build_annotation_review_report(annotation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_owner_review: list[dict[str, str]] = []
    classification_counts: dict[str, int] = {}
    intentional_empty_fields: list[dict[str, Any]] = []
    legacy_rows = _legacy_rows_by_sample()
    for row in annotation_rows:
        sample_id = row["sample_id"]
        for record in _annotation_field_records(row, legacy_rows[sample_id]):
            classification = record.get("raw_provenance_classification") or "unknown"
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
            if record.get("review_required"):
                required_owner_review.append({"sample_id": sample_id, "origin": str(record.get("source_field")), "classification": classification, "field_path": record["field_path"]})
        intentional_empty_fields.extend(_intentional_empty_field_records(row))
    return {
        "schema_version": "opk-rag.core-rag-annotation-review-report.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "owner_review_required" if required_owner_review else "owner_review_complete",
        "classification_counts": classification_counts,
        "required_owner_review_count": len(required_owner_review),
        "required_owner_review": required_owner_review,
        "intentional_empty_field_count": len(intentional_empty_fields),
        "intentional_empty_fields": intentional_empty_fields,
    }


def _owner_annotation_review_complete() -> bool:
    if not ANNOTATION_REVIEW_REPORT.exists():
        return False
    return read_json(ANNOTATION_REVIEW_REPORT).get("status") == "owner_review_complete"


def build_owner_review_artifacts() -> dict[str, Any]:
    questions = read_jsonl(core_path("question_set"))
    annotations = read_jsonl(core_path("annotations"))
    legacy_rows = _legacy_rows_by_sample()
    annotation_by_id = {row["sample_id"]: row for row in annotations}
    package_rows = [
        _owner_review_package_row(question, annotation_by_id[question["sample_id"]], legacy_rows[question["sample_id"]])
        for question in questions
    ]
    decision_rows = [_pending_owner_decision(row) for row in package_rows]
    write_jsonl(OWNER_REVIEW_PACKAGE, package_rows)
    write_jsonl(OWNER_REVIEW_DECISIONS, decision_rows)
    validation = validate_owner_review(write_report=False)
    write_json(OWNER_REVIEW_VALIDATION, validation)
    return validation


def validate_owner_review(*, write_report: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    questions = read_jsonl(core_path("question_set"))
    annotations = read_jsonl(core_path("annotations"))
    package_rows = read_jsonl(OWNER_REVIEW_PACKAGE) if OWNER_REVIEW_PACKAGE.exists() else []
    decision_rows = read_jsonl(OWNER_REVIEW_DECISIONS) if OWNER_REVIEW_DECISIONS.exists() else []
    official_ids = [row["sample_id"] for row in questions]
    annotation_by_id = {row["sample_id"]: row for row in annotations}
    package_by_id = _unique_rows_by_sample(package_rows, "review_package", issues)
    decision_by_id = _unique_rows_by_sample(decision_rows, "owner_decision", issues)

    if len(package_rows) != 40:
        issues.append({"code": "review_package_sample_count", "message": f"Expected 40 review package rows, found {len(package_rows)}."})
    if set(package_by_id) != set(official_ids):
        issues.append({"code": "review_package_sample_ids", "message": "Review package sample IDs do not match official samples."})
    if set(decision_by_id) != set(official_ids):
        issues.append({"code": "owner_decision_sample_ids", "message": "Owner decision sample IDs do not match official samples."})

    audit = audit_annotation_review_inventory()
    reviewed_items = 0
    approved = approved_with_edits = rejected = needs_source_check = unresolved = 0
    for sample_id in official_ids:
        annotation = annotation_by_id.get(sample_id) or {}
        expected_digest = stable_hash(annotation)
        package = package_by_id.get(sample_id)
        decision = decision_by_id.get(sample_id)
        if package:
            if package.get("pre_review_annotation_digest") != expected_digest:
                issues.append({"code": "package_annotation_digest_mismatch", "sample_id": sample_id, "message": "Review package is not bound to the current annotation digest."})
            _validate_no_leakage(package, f"review_package:{sample_id}", issues)
            _validate_claim_field_contract(sample_id, annotation, package, issues)
            _validate_bounded_evidence_contract(sample_id, annotation, package, issues)
        if not decision:
            unresolved += 1
            continue
        decision_value = decision.get("decision")
        if decision.get("pre_review_annotation_digest") != expected_digest:
            issues.append({"code": "decision_annotation_digest_mismatch", "sample_id": sample_id, "message": "Owner decision is not bound to the current annotation digest."})
        if decision.get("review_role") != "benchmark_owner":
            issues.append({"code": "review_role", "sample_id": sample_id, "message": "Owner decision must use review_role=benchmark_owner."})
        if decision_value not in OWNER_REVIEW_DECISIONS_ALLOWED:
            issues.append({"code": "decision_invalid", "sample_id": sample_id, "message": "Owner decision is missing or invalid."})
            unresolved += 1
            continue
        if decision_value == "pending":
            unresolved += 1
        elif decision_value == "approve":
            approved += 1
            reviewed_items += len(decision.get("reviewed_fields") or [])
            _validate_reviewed_field_coverage(sample_id, package, decision, issues)
            if _decision_is_generated_approval(decision):
                issues.append({"code": "automatic_owner_approval", "sample_id": sample_id, "message": "Approval cannot be marked as automatically generated."})
        elif decision_value == "approve_with_edits":
            approved_with_edits += 1
            reviewed_items += len(decision.get("reviewed_fields") or [])
            _validate_reviewed_field_coverage(sample_id, package, decision, issues)
            edits = decision.get("explicit_edits") or {}
            if not isinstance(edits, dict) or not edits:
                issues.append({"code": "approve_with_edits_missing_values", "sample_id": sample_id, "message": "approve_with_edits requires explicit replacement values."})
            if not str(decision.get("review_note") or "").strip():
                issues.append({"code": "approve_with_edits_missing_note", "sample_id": sample_id, "message": "approve_with_edits requires a review note."})
            _validate_annotation_edits(sample_id, edits, issues)
            if _decision_is_generated_approval(decision):
                issues.append({"code": "automatic_owner_approval", "sample_id": sample_id, "message": "Approval cannot be marked as automatically generated."})
        elif decision_value == "reject_annotation":
            rejected += 1
            rejected_fields = decision.get("rejected_fields") or []
            if not rejected_fields or not str(decision.get("review_note") or "").strip():
                issues.append({"code": "reject_missing_fields_or_reason", "sample_id": sample_id, "message": "reject_annotation requires rejected fields and a reason."})
        elif decision_value == "needs_source_check":
            needs_source_check += 1
        _validate_no_leakage(decision, f"owner_decision:{sample_id}", issues)

    privacy_scan = scan_paths_for_privacy([OWNER_REVIEW_PACKAGE, OWNER_REVIEW_DECISIONS, OWNER_REVIEW_VALIDATION])
    for finding in privacy_scan["findings"]:
        issues.append({"code": "privacy_scan_failure", **finding})
    remaining_review_required = audit["review_required_item_count"] - reviewed_items
    has_publication_blocker = bool(rejected or needs_source_check or unresolved or remaining_review_required > 0)
    if has_publication_blocker:
        issues.append({"code": "publication_blocked_owner_review", "message": "Publication remains blocked until every required owner decision is supplied and valid."})
    manifest_path = BENCHMARK_DIR / "benchmark_manifest.json"
    manifest_state = read_json(manifest_path).get("publication_state") if manifest_path.exists() else "acceptance_blocked"
    result = {
        "schema_version": "opk-rag.core-rag-owner-review-validation.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "valid" if not [issue for issue in issues if issue.get("code") != "publication_blocked_owner_review"] else "invalid",
        "publication_eligible": not has_publication_blocker and manifest_state == "published",
        "publication_state": manifest_state if not has_publication_blocker else "acceptance_blocked",
        "total_official_samples": len(official_ids),
        "reviewed_samples": approved + approved_with_edits + rejected + needs_source_check,
        "approved_samples": approved,
        "approved_with_edits_samples": approved_with_edits,
        "rejected_samples": rejected,
        "needs_source_check_samples": needs_source_check,
        "unresolved_samples": unresolved,
        "reviewed_annotation_items": reviewed_items,
        "remaining_review_required_items": max(0, remaining_review_required),
        "annotation_inventory": audit,
        "privacy_scan": privacy_scan,
        "issues": issues,
    }
    if write_report:
        write_json(OWNER_REVIEW_VALIDATION, result)
    return result


def _validate_claim_field_contract(sample_id: str, annotation: dict[str, Any], package: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    for claim_group in ("required_claims", "forbidden_claims"):
        claims = annotation.get(claim_group)
        if not isinstance(claims, list):
            issues.append({"code": "claim_collection_not_array", "sample_id": sample_id, "field": claim_group, "message": "Claim collection must be an array."})
            continue
        for index, claim in enumerate(claims):
            value = claim.get("text") if isinstance(claim, dict) else claim
            if _is_placeholder_claim(value):
                issues.append({"code": "claim_placeholder_value", "sample_id": sample_id, "field": f"{claim_group}[{index}]", "message": "Claim collection contains a null or placeholder element."})
        state = ((annotation.get("claim_field_states") or {}).get(claim_group) or {})
        if not claims:
            if state.get("field_state") == "unresolved_missing_annotation" and claim_group not in (package.get("review_required_fields") or []):
                issues.append({"code": "unresolved_field_excluded_from_review", "sample_id": sample_id, "field": claim_group, "message": "Unresolved scoring-relevant field is excluded from review."})
            elif state.get("field_state") == "unresolved_missing_annotation" and state.get("owner_input_required") is not True:
                issues.append({"code": "unresolved_field_not_marked_owner_input_required", "sample_id": sample_id, "field": claim_group, "message": "Unresolved scoring-relevant field must require owner input."})
            else:
                if state.get("field_state") not in {"empty_by_design", "not_applicable"} or not state.get("empty_reason") or not state.get("provenance_source"):
                    issues.append({"code": "empty_claim_field_metadata_missing", "sample_id": sample_id, "field": claim_group, "message": "Intentional empty claim field lacks explicit state, reason, or provenance."})
                if claim_group not in (package.get("sample_level_review_fields") or []):
                    issues.append({"code": "empty_claim_field_not_visible", "sample_id": sample_id, "field": claim_group, "message": "Intentional empty claim field is not visible for sample-level owner review."})


def _validate_bounded_evidence_contract(sample_id: str, annotation: dict[str, Any], package: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    locators = annotation.get("required_evidence") or []
    excerpts = package.get("bounded_source_evidence") or []
    by_locator: dict[str, list[dict[str, Any]]] = {}
    for excerpt in excerpts:
        locator_id = excerpt.get("evidence_locator_id")
        if locator_id:
            by_locator.setdefault(locator_id, []).append(excerpt)
        text = excerpt.get("bounded_evidence_text")
        if excerpt.get("resolution_status") == "resolved" and excerpt.get("excerpt_digest") != text_digest(str(text or "")):
            issues.append({"code": "bounded_excerpt_digest_mismatch", "sample_id": sample_id, "evidence_locator_id": locator_id, "message": "Bounded excerpt digest does not match bounded text."})
        if excerpt.get("resolution_status") == "needs_source_check" and locator_id in (package.get("review_required_fields") or []):
            issues.append({"code": "unresolved_locator_treated_reviewable", "sample_id": sample_id, "evidence_locator_id": locator_id, "message": "Unresolved locator cannot be treated as reviewable or approved."})
    for locator in locators:
        locator_id = locator.get("evidence_id")
        matches = by_locator.get(locator_id) or []
        if not matches:
            issues.append({"code": "gold_locator_missing_bounded_excerpt", "sample_id": sample_id, "evidence_locator_id": locator_id, "message": "Gold evidence locator lacks a matching bounded excerpt."})
        elif all(match.get("resolution_status") != "resolved" for match in matches):
            issues.append({"code": "gold_locator_needs_source_check", "sample_id": sample_id, "evidence_locator_id": locator_id, "message": "Gold evidence locator could not be resolved deterministically."})
    seen_text: dict[tuple[str, str], str] = {}
    for excerpt in excerpts:
        key = (str(excerpt.get("normalized_relative_source_identity")), str(excerpt.get("excerpt_digest")))
        locator_id = str(excerpt.get("evidence_locator_id"))
        previous = seen_text.get(key)
        if previous and previous != locator_id and not excerpt.get("identical_span_contract"):
            issues.append({"code": "bounded_excerpt_reused_without_contract", "sample_id": sample_id, "evidence_locator_id": locator_id, "message": "One bounded excerpt is reused for multiple locators without an identical-span contract."})
        seen_text[key] = locator_id


def audit_annotation_review_inventory() -> dict[str, Any]:
    annotations = read_jsonl(core_path("annotations"))
    field_records: list[dict[str, Any]] = []
    intentional_empty_fields: list[dict[str, Any]] = []
    all_counts = {category: 0 for category in PROVENANCE_CATEGORY_MAP.values()}
    review_counts = {category: 0 for category in PROVENANCE_CATEGORY_MAP.values()}
    type_counts = {
        "semantic_annotation_fields": 0,
        "evidence_locator_fields": 0,
        "intentional_empty_fields": 0,
        "review_required_items": 0,
        "non_review_required_deterministic_metadata": 0,
    }
    for annotation in annotations:
        for record in _annotation_field_records(annotation, _legacy_rows_by_sample()[annotation["sample_id"]]):
            field_records.append(record)
            all_counts[record["provenance_category"]] += 1
            if record["review_required"]:
                review_counts[record["provenance_category"]] += 1
                type_counts["review_required_items"] += 1
            if record["field_kind"] == "evidence_locator":
                type_counts["evidence_locator_fields"] += 1
            elif record["field_kind"] == "deterministic_metadata":
                type_counts["non_review_required_deterministic_metadata"] += 0 if record["review_required"] else 1
            else:
                type_counts["semantic_annotation_fields"] += 1
        for record in _intentional_empty_field_records(annotation):
            intentional_empty_fields.append(record)
            type_counts["intentional_empty_fields"] += 1
    return {
        "schema_version": "opk-rag.core-rag-annotation-inventory-audit.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "official_sample_count": len(annotations),
        "annotation_item_count": len(field_records),
        "review_required_item_count": sum(1 for record in field_records if record["review_required"]),
        "category_counts_all_annotation_items": all_counts,
        "category_counts_review_required_only": review_counts,
        "non_review_required_item_count": sum(1 for record in field_records if not record["review_required"]),
        "inventory_type_counts": type_counts,
        "intentional_empty_field_count": len(intentional_empty_fields),
        "intentional_empty_fields": intentional_empty_fields,
        "field_records": field_records,
    }


def _legacy_rows_by_sample() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(ROOT / "evaluation-data" / "answerability_dev.jsonl")
    rows.extend(read_jsonl(ROOT / "evaluation-data" / "answerability_test.jsonl"))
    return {row["id"]: row for row in rows}


def _owner_review_package_row(question: dict[str, Any], annotation: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    field_records = _annotation_field_records(annotation, legacy)
    required_field_paths = [record["field_path"] for record in field_records if record["review_required"]]
    evidence_text = _bounded_evidence_for_locators(annotation, legacy)
    empty_claim_fields = [
        state
        for state in (annotation.get("claim_field_states") or {}).values()
        if state.get("field_state") in {"empty_by_design", "not_applicable"}
    ]
    return {
        "schema_version": OWNER_REVIEW_PACKAGE_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "sample_id": question["sample_id"],
        "semantic_evaluation_role": question["dataset_role"],
        "question": question["question"],
        "current_answerability_label": annotation["answerability_label"],
        "expected_action": annotation["expected_action"],
        "question_type": annotation["question_type"],
        "stable_gold_document_identity": [
            item.get("identity") or {} for item in annotation.get("required_evidence") or []
        ],
        "stable_gold_evidence_locator": [
            {
                "evidence_id": item.get("evidence_id"),
                "granularity": item.get("granularity"),
                "identity": item.get("identity") or {},
                "legacy_source_reference": item.get("legacy_source_reference") or {},
            }
            for item in annotation.get("required_evidence") or []
        ],
        "bounded_source_evidence": evidence_text,
        "required_claims": annotation.get("required_claims") or [],
        "forbidden_claims": annotation.get("forbidden_claims") or [],
        "claim_field_states": annotation.get("claim_field_states") or {},
        "sample_level_review_fields": [state["field_path"] for state in empty_claim_fields],
        "intentional_empty_fields": empty_claim_fields,
        "unanswerable_reason": annotation.get("refusal_reason"),
        "partial_answer_scope": {
            "supported": legacy.get("supported_subquestions") or [],
            "unsupported": legacy.get("unsupported_subquestions") or [],
        }
        if question["answerability_label"] == "partially_answerable"
        else None,
        "field_provenance": field_records,
        "review_required_fields": required_field_paths,
        "unresolved_review_questions": [
            {
                "field_path": record["field_path"],
                "question": "Benchmark Owner must explicitly approve, edit, reject, or request source check for this semantic annotation field.",
            }
            for record in field_records
            if record["review_required"]
        ],
        "source_fingerprint": {
            "source_dataset": annotation.get("annotation_provenance", {}).get("source_dataset"),
            "source_record_digest": annotation.get("annotation_provenance", {}).get("source_record_digest"),
            "source_evidence_digest": stable_hash(legacy.get("evidence") or []),
        },
        "pre_review_annotation_digest": stable_hash(annotation),
        "benchmark_annotation_digest": file_digest(core_path("annotations")),
        "blind_review_policy": {
            "baseline_outcomes_excluded": True,
            "model_answers_excluded": True,
            "metric_effect_excluded": True,
        },
    }


def _bounded_evidence_for_locators(annotation: dict[str, Any], legacy: dict[str, Any]) -> list[dict[str, Any]]:
    by_excerpt_digest: dict[str, dict[str, Any]] = {}
    for item in legacy.get("evidence") or []:
        excerpt = str(item.get("excerpt") or "")
        by_excerpt_digest[text_digest(excerpt)] = item
        by_excerpt_digest.setdefault(text_digest(excerpt.strip()), item)
    bounded: list[dict[str, Any]] = []
    for locator in annotation.get("required_evidence") or []:
        legacy_ref = locator.get("legacy_source_reference") or {}
        excerpt_digest = legacy_ref.get("excerpt_digest")
        source = by_excerpt_digest.get(excerpt_digest) or _source_excerpt_for_locator(locator)
        if not source:
            bounded.append(
                {
                    "sample_id": annotation["sample_id"],
                    "evidence_locator_id": locator.get("evidence_id"),
                    "resolution_status": "needs_source_check",
                    "resolution_failure": "legacy excerpt digest did not match a bounded legacy evidence excerpt",
                    "expected_excerpt_digest": excerpt_digest,
                }
            )
            continue
        excerpt = str(source.get("excerpt") or "")
        if text_digest(excerpt) != excerpt_digest and text_digest(excerpt.strip()) == excerpt_digest:
            excerpt = excerpt.strip()
        heading = source.get("heading")
        source_file = _normalize_review_source_path(source.get("source_file"))
        bounded.append(
            {
                "sample_id": annotation["sample_id"],
                "evidence_locator_id": locator.get("evidence_id"),
                "resolution_status": "resolved",
                "normalized_relative_source_identity": source_file,
                "heading_or_source_span_identity": {
                    "heading_path": [heading] if heading else [],
                    "heading_digest": legacy_ref.get("heading_digest") or text_digest(str(heading or "")),
                },
                "source_fingerprint": {
                    "source_file_digest": legacy_ref.get("source_file_digest") or text_digest(str(source_file or "")),
                    "source_digest": (locator.get("identity") or {}).get("source_digest"),
                },
                "bounded_evidence_text": excerpt,
                "excerpt_digest": text_digest(excerpt),
            }
        )
    return bounded


def _source_excerpt_for_locator(locator: dict[str, Any]) -> dict[str, Any] | None:
    legacy_ref = locator.get("legacy_source_reference") or {}
    expected_source_digest = legacy_ref.get("source_file_digest")
    expected_excerpt_digest = legacy_ref.get("excerpt_digest")
    heading_digest = legacy_ref.get("heading_digest")
    for path in sorted((ROOT / "source-documents").rglob("*.md")):
        relative = path.relative_to(ROOT / "source-documents").as_posix()
        if expected_source_digest and text_digest(relative) != expected_source_digest:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            excerpt = line.strip()
            if excerpt and text_digest(excerpt) == expected_excerpt_digest:
                return {
                    "source_file": relative,
                    "heading": _heading_for_digest(heading_digest, path),
                    "excerpt": excerpt,
                }
    return None


def _heading_for_digest(heading_digest: str | None, path: Path) -> str | None:
    if not heading_digest:
        return None
    headings: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip()
        headings.append(heading)
        if stable_hash([heading]) == heading_digest:
            return heading
    return None


def _normalize_review_source_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("source-documents/"):
        text = text[len("source-documents/") :]
    while text.startswith("./"):
        text = text[2:]
    return text


def _pending_owner_decision(package_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": OWNER_REVIEW_DECISION_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "sample_id": package_row["sample_id"],
        "pre_review_annotation_digest": package_row["pre_review_annotation_digest"],
        "reviewed_fields": [],
        "decision": "pending",
        "explicit_edits": {},
        "rejected_fields": [],
        "review_role": "benchmark_owner",
        "review_note": "",
        "review_metadata": {
            "status": "pending_owner_input",
            "created_by": "workflow_template",
            "review_timestamp": None,
            "automatic_approval": False,
        },
    }


def _annotation_field_records(annotation: dict[str, Any], legacy: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.append(
        _field_record(
            annotation,
            legacy,
            field_path="annotation_provenance",
            current_value={
                "answerability_label": annotation.get("answerability_label"),
                "expected_action": annotation.get("expected_action"),
                "question_type": annotation.get("question_type"),
            },
            source_artifact=(annotation.get("annotation_provenance") or {}).get("source_dataset"),
            source_field="expected_answerability, expected_action, answerability_type",
            original_value={
                "expected_answerability": legacy.get("expected_answerability"),
                "expected_action": legacy.get("expected_action"),
                "answerability_type": legacy.get("answerability_type"),
            },
            provenance=annotation.get("annotation_provenance") or {},
            basis="Legacy answerability fields normalized into Core RAG benchmark names.",
            field_kind="semantic_annotation",
        )
    )
    for index, item in enumerate(annotation.get("required_evidence") or []):
        records.append(
            _field_record(
                annotation,
                legacy,
                field_path=f"required_evidence[{index}]",
                current_value=item,
                source_artifact=_gold_identity_source(annotation),
                source_field=f"samples[].required_evidence[{index}]",
                original_value=item.get("legacy_source_reference") or {},
                provenance=item.get("provenance") or {},
                deterministic_derivation_contract="legacy-source-heading-resolution resolves legacy source_file, heading, and excerpt digests into stable document/scope/source identity digests without semantic inference.",
                basis="Stable locator identity migrated from Phase 2 gold identity map.",
                field_kind="evidence_locator",
            )
        )
    for index, item in enumerate(annotation.get("required_claims") or []):
        records.append(
            _field_record(
                annotation,
                legacy,
                field_path=f"required_claims[{index}]",
                current_value=item.get("text"),
                source_artifact=(annotation.get("annotation_provenance") or {}).get("source_dataset"),
                source_field=f"required_facts[{index}]",
                original_value=(legacy.get("required_facts") or [None] * (index + 1))[index] if index < len(legacy.get("required_facts") or []) else None,
                provenance=item.get("provenance") or {},
                basis="Required claim text inferred from legacy required_facts and bounded evidence excerpts.",
                field_kind="semantic_annotation",
            )
        )
    for index, item in enumerate(annotation.get("forbidden_claims") or []):
        records.append(
            _field_record(
                annotation,
                legacy,
                field_path=f"forbidden_claims[{index}]",
                current_value=item.get("text"),
                source_artifact=(annotation.get("annotation_provenance") or {}).get("source_dataset"),
                source_field=f"unsupported_subquestions[{index}]",
                original_value=(legacy.get("unsupported_subquestions") or [None] * (index + 1))[index] if index < len(legacy.get("unsupported_subquestions") or []) else None,
                provenance=item.get("provenance") or {},
                basis="Forbidden claim text inferred from legacy unsupported_subquestions, or an unchanged null placeholder when none existed.",
                field_kind="semantic_annotation",
            )
        )
    if annotation.get("refusal_reason"):
        records.append(
            _field_record(
                annotation,
                legacy,
                field_path="refusal_reason",
                current_value=annotation.get("refusal_reason"),
                source_artifact=(annotation.get("annotation_provenance") or {}).get("source_dataset"),
                source_field="expected_reason, answerability_type",
                original_value={"expected_reason": legacy.get("expected_reason"), "answerability_type": legacy.get("answerability_type")},
                provenance=(annotation["refusal_reason"].get("provenance") or {}),
                basis="Refusal reason code inferred from legacy expected_reason and answerability_type.",
                field_kind="semantic_annotation",
            )
        )
    return records


def _intentional_empty_field_records(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for field_name, state in (annotation.get("claim_field_states") or {}).items():
        if state.get("field_state") in {"empty_by_design", "not_applicable"}:
            records.append(
                {
                    "sample_id": annotation.get("sample_id"),
                    "field_path": field_name,
                    "field_state": state.get("field_state"),
                    "empty_reason": state.get("empty_reason"),
                    "provenance_source": state.get("provenance_source"),
                    "semantic_meaning_changed": state.get("semantic_meaning_changed") is True,
                    "review_visibility": state.get("review_visibility"),
                    "owner_input_required": state.get("owner_input_required") is True,
                }
            )
    return records


def _field_record(
    annotation: dict[str, Any],
    legacy: dict[str, Any],
    *,
    field_path: str,
    current_value: Any,
    source_artifact: str | None,
    source_field: str,
    original_value: Any,
    provenance: dict[str, Any],
    basis: str,
    deterministic_derivation_contract: str | None = None,
    field_kind: str = "semantic_annotation",
) -> dict[str, Any]:
    raw_category = provenance.get("classification")
    category = PROVENANCE_CATEGORY_MAP.get(raw_category or "", "newly_inferred_semantic")
    return {
        "field_path": field_path,
        "field_kind": field_kind,
        "provenance_category": category,
        "raw_provenance_classification": raw_category,
        "review_required": provenance.get("review_status") == "owner_review_required",
        "source_artifact": source_artifact,
        "source_field": source_field,
        "original_source_value": original_value,
        "normalized_current_value": current_value,
        "semantic_meaning_changed": bool(provenance.get("semantic_mutation")),
        "deterministic_derivation_contract": deterministic_derivation_contract,
        "reasoning_basis": basis,
        "supporting_source_evidence": [
            {
                "source_file": item.get("source_file"),
                "heading_path": [item.get("heading")] if item.get("heading") else [],
                "excerpt_digest": text_digest(str(item.get("excerpt") or "")),
                "bounded_evidence_text": item.get("excerpt"),
            }
            for item in legacy.get("evidence") or []
        ],
    }


def _gold_identity_source(annotation: dict[str, Any]) -> str:
    return (
        "evaluation-data/diagnostics/phase2_development_gold_identity_v1.json"
        if annotation.get("dataset_role") == DEV_SPLIT
        else "evaluation-data/diagnostics/phase2_known_regression_gold_identity_v1.json"
    )


def _unique_rows_by_sample(rows: list[dict[str, Any]], label: str, issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            issues.append({"code": f"{label}_missing_sample_id", "message": "Row is missing sample_id."})
            continue
        if sample_id in seen:
            issues.append({"code": f"{label}_duplicate_sample_id", "sample_id": sample_id, "message": "Sample ID appears more than once."})
        seen.add(sample_id)
        by_id.setdefault(sample_id, row)
    return by_id


def _validate_no_leakage(value: Any, context: str, issues: list[dict[str, Any]]) -> None:
    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).lower()
                if key_text in OWNER_REVIEW_LEAKAGE_KEYS:
                    issues.append({"code": "baseline_outcome_leakage", "context": context, "field": path + "." + str(key), "message": "Review artifact contains a forbidden model outcome or metric field."})
                walk(child, path + "." + str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "$")


def _decision_is_generated_approval(decision: dict[str, Any]) -> bool:
    metadata = decision.get("review_metadata") or {}
    return metadata.get("automatic_approval") is True or metadata.get("created_by") == "workflow_template"


def _validate_annotation_edits(sample_id: str, edits: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    if not isinstance(edits, dict):
        issues.append({"code": "edit_values_invalid", "sample_id": sample_id, "message": "Explicit edits must be an object keyed by annotation field path."})
        return
    for field_path, value in edits.items():
        if not isinstance(field_path, str) or not field_path.strip():
            issues.append({"code": "edit_field_path_invalid", "sample_id": sample_id, "message": "Edit field paths must be non-empty strings."})
        if value is None:
            issues.append({"code": "edit_value_missing", "sample_id": sample_id, "field_path": field_path, "message": "Edited replacement value cannot be null."})


def _validate_reviewed_field_coverage(sample_id: str, package: dict[str, Any] | None, decision: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    required_fields = set((package or {}).get("review_required_fields") or [])
    reviewed_fields = set(decision.get("reviewed_fields") or [])
    missing = sorted(required_fields - reviewed_fields)
    if missing:
        issues.append({"code": "approve_with_unresolved_fields", "sample_id": sample_id, "fields": missing, "message": "Approved sample does not list every review-required field."})


def annotation_edit_update_policy() -> dict[str, Any]:
    return {
        "preserve_original_annotation": True,
        "preserve_owner_decision": True,
        "generate_annotation_change_record": True,
        "update_annotation_digest": True,
        "update_benchmark_binding_as_required": True,
        "invalidate_candidate_baselines_bound_to_old_annotation_digest": True,
        "fresh_candidate_baselines_required_before_publication": True,
        "version_change_policy": {
            "patch": "Typographical, formatting, provenance, or review-note changes that do not alter sample semantics, labels, evidence identity, expected action, or scoring interpretation.",
            "minor": "Development-split semantic annotation edits that preserve the benchmark family and do not change Known Regression semantics.",
            "major_or_new_benchmark_id": "Known Regression semantic changes, sample additions/removals, scoring contract changes, expected-action changes, answerability label changes, or incompatible evidence identity changes.",
        },
    }


def _content_checksum(paths: Iterable[Path]) -> str:
    payload = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        payload.append(
            {
                "path": relative_path(path),
                "sha256": file_digest(path),
                "sample_count": len(read_jsonl(path)) if path.suffix.lower() == ".jsonl" else None,
            }
        )
    return stable_hash(payload)


def _runtime_blockers() -> list[dict[str, Any]]:
    load_project_env()
    blockers: list[dict[str, Any]] = []
    private_manifest = ROOT / ".private" / "evaluation" / "phase2_corpus_manifest.json"
    if not private_manifest.exists():
        blockers.append({"component": "local_markdown_vault", "code": "private_corpus_manifest_missing", "path": ".private/evaluation/phase2_corpus_manifest.json"})
    required_env = {
        "local_markdown_vault": ["OPK_RAG_VAULT_PATH"],
        "remote_supabase": ["SUPABASE_URL"],
        "remote_deepseek": ["OPK_RAG_LLM_BASE_URL", "OPK_RAG_LLM_API_KEY", "OPK_RAG_LLM_MODEL", "OPK_RAG_LLM_ALLOW_REMOTE"],
    }
    for component, env_names in required_env.items():
        missing = [name for name in env_names if not os.environ.get(name)]
        if missing:
            if component == "remote_supabase" and os.environ.get("OPK_RAG_DATABASE_URL"):
                continue
            blockers.append({"component": component, "code": "missing_environment", "env_names": missing})
    allow_remote = os.environ.get("OPK_RAG_LLM_ALLOW_REMOTE")
    if allow_remote and allow_remote.lower() not in {"1", "true", "yes", "on"}:
        blockers.append({"component": "remote_deepseek", "code": "remote_llm_not_allowed", "env_names": ["OPK_RAG_LLM_ALLOW_REMOTE"]})
    base_url = os.environ.get("OPK_RAG_LLM_BASE_URL", "")
    if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
        blockers.append({"component": "remote_deepseek", "code": "local_llm_endpoint_not_official", "env_names": ["OPK_RAG_LLM_BASE_URL"]})
    return blockers


def _run_legacy_answerability_baseline(*, split: str, output_root: Path, run_id: str) -> dict[str, Any]:
    dataset = ROOT / "evaluation-data" / ("answerability_dev.jsonl" if split == DEV_SPLIT else "answerability_test.jsonl")
    legacy_dir = output_root / "legacy-answerability"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    results = legacy_dir / f"{split}_{run_id}_results.jsonl"
    report_json = legacy_dir / f"{split}_{run_id}_report.json"
    report_md = legacy_dir / f"{split}_{run_id}_report.md"
    command = [
        "uv",
        "run",
        "python",
        "scripts/evaluate_answerability_baseline.py",
        "--dataset",
        relative_path(dataset),
        "--results",
        repo_relative_or_absolute(results),
        "--report-json",
        repo_relative_or_absolute(report_json),
        "--markdown-report",
        repo_relative_or_absolute(report_md),
        "--vault-path",
        os.environ["OPK_RAG_VAULT_PATH"],
        "--baseline-mode",
        "current-configured",
        "--overwrite",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if not report_json.exists():
        write_json(
            report_json,
            {
                "status": "runner_failed_before_report",
                "run_summary": {"completed": 0, "failed": 0, "skipped": 0},
                "infrastructure_failure_count": None,
                "runner_stdout_tail": completed.stdout[-2000:],
                "runner_stderr_tail": completed.stderr[-2000:],
            },
        )
    return {
        "exit_code": completed.returncode,
        "results": results,
        "report_json": report_json,
        "markdown_report": report_md,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _core_sample_result(row: dict[str, Any], *, annotation: dict[str, Any], split: str, run_id: str) -> dict[str, Any]:
    infrastructure_failure = bool(row.get("infrastructure_failure")) or row.get("error_type") == "infrastructure_failure"
    expected_action = annotation["expected_action"]
    expected_behavior = "abstain" if expected_action == "abstain" else "answer"
    return {
        "schema_version": "opk-rag.core-rag-baseline-sample-result.v1",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "run_id": run_id,
        "sample_id": row.get("id"),
        "split": split,
        "answerability_label": annotation["answerability_label"],
        "question_type": annotation["question_type"],
        "expected_action": expected_action,
        "expected_behavior": expected_behavior,
        "execution_status": "infrastructure_failure" if infrastructure_failure else "completed",
        "infrastructure_failures": []
        if not infrastructure_failure
        else [
            {
                "component": "reference_runtime",
                "code": row.get("system_error_code") or row.get("infrastructure_error_category") or "sample_infrastructure_failure",
            }
        ],
        "final_action": row.get("final_action"),
        "actual_action": row.get("actual_action"),
        "error_type": row.get("error_type"),
        "failure_stage": row.get("failure_stage"),
        "decision_status": row.get("decision_status"),
        "generation_status": row.get("generation_status"),
        "retrieval_status": row.get("retrieval_status"),
        "citation_status": row.get("citation_status"),
        "grounding_status": row.get("grounding_status"),
        "grounding_valid": row.get("grounding_valid"),
        "has_required_evidence": bool(annotation.get("required_evidence")),
        "generation_overreach": _legacy_row_has_generation_overreach(row),
        "metrics": {
            "infrastructure_failure": infrastructure_failure,
            "answerability_match": row.get("actual_action") == expected_behavior,
        },
    }


def _legacy_row_has_generation_overreach(row: dict[str, Any]) -> bool:
    if row.get("generation_status") != "generated":
        return False
    unsupported = row.get("unsupported_claims")
    if isinstance(unsupported, list) and unsupported:
        return True
    model_decision = row.get("model_decision") if isinstance(row.get("model_decision"), dict) else {}
    contract_failures = model_decision.get("contract_failures")
    if isinstance(contract_failures, list) and contract_failures:
        return True
    grounding = row.get("grounding") if isinstance(row.get("grounding"), dict) else {}
    diagnostics = grounding.get("diagnostics") if isinstance(grounding.get("diagnostics"), dict) else {}
    return bool(diagnostics.get("unsupported_claims") or diagnostics.get("contract_failures"))


def _aggregate_metric(numerator: int, denominator: int, scope: str) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
        "scope": scope,
    }


def _compute_core_baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answering_actions = {"answer", "partial_answer", "correct_premise"}
    answerable = [row for row in rows if row["expected_action"] in answering_actions]
    unanswerable = [row for row in rows if row["expected_action"] == "abstain"]
    partial = [row for row in rows if row["expected_action"] == "partial_answer"]
    expected_positive = answerable
    decision_allowed_positive = [row for row in expected_positive if row.get("decision_status") == "allowed"]
    generated_positive = [row for row in expected_positive if row.get("generation_status") == "generated"]
    answered = [row for row in rows if row.get("actual_action") == "answer"]
    allowed_answered = [row for row in rows if row.get("actual_action") == "answer" and row.get("decision_status") == "allowed"]
    expected_with_sources = [row for row in expected_positive if row.get("has_required_evidence")]
    infrastructure_failures = [row for row in rows if row.get("error_type") == "infrastructure_failure"]
    schema_parse_failures = [
        row
        for row in rows
        if row.get("schema_parse_status") == "failed" or row.get("system_error_code") == "invalid_model_output"
    ]
    specs = {
        "answerability_accuracy": (sum(row.get("actual_action") == row.get("expected_behavior") for row in rows), len(rows), "all samples"),
        "false_answer_rate": (sum(row.get("actual_action") == "answer" for row in unanswerable), len(unanswerable), "expected abstain samples"),
        "over_refusal_rate": (sum(row.get("actual_action") == "abstain" for row in answerable), len(answerable), "expected answer / partial / correction samples"),
        "unanswerable_recall": (sum(row.get("actual_action") == "abstain" for row in unanswerable), len(unanswerable), "expected abstain samples"),
        "answerable_pass_rate": (sum(row.get("actual_action") == "answer" for row in answerable), len(answerable), "expected answer / partial / correction samples"),
        "partial_answer_handling_rate": (sum(row.get("actual_action") == "answer" and row.get("error_type") == "no_error" for row in partial), len(partial), "expected partial_answer samples"),
        "citation_validity_rate": (sum(row.get("citation_status") == "valid" for row in allowed_answered), len(allowed_answered), "allowed answered samples"),
        "grounding_pass_rate": (sum(row.get("grounding_valid") for row in answered), len(answered), "all actually answered samples"),
        "decision_over_refusal_rate": (sum(row.get("failure_stage") == "decision" for row in expected_positive), len(expected_positive), "expected answer / partial / correction samples"),
        "generation_abstention_rate": (sum(row.get("generation_status") == "abstained" for row in decision_allowed_positive), len(decision_allowed_positive), "decision-allowed positive samples"),
        "grounding_rejection_rate": (sum(row.get("failure_stage") == "grounding" for row in generated_positive), len(generated_positive), "positive samples with generated answers"),
        "generation_overreach_rate": (sum(row.get("generation_overreach") for row in rows), len(generated_positive), "positive samples with generated answers"),
        "retrieval_failure_rate": (sum(row.get("failure_stage") == "retrieval" for row in expected_with_sources), len(expected_with_sources), "positive samples with expected source files"),
        "infrastructure_failure_rate": (len(infrastructure_failures), len(rows), "all completed or failed samples"),
        "schema_parse_failure_rate": (len(schema_parse_failures), len(rows), "all completed or failed samples"),
        "false_premise_correction_rate": (
            sum(row.get("actual_action") == "answer" and row.get("error_type") == "no_error" for row in rows if row["expected_action"] == "correct_premise"),
            sum(row["expected_action"] == "correct_premise" for row in rows),
            "expected correct_premise samples",
        ),
    }
    return {name: _aggregate_metric(numerator, denominator, scope) for name, (numerator, denominator, scope) in specs.items()}


def _blockers_from_legacy_run(report: dict[str, Any], *, runner_exit_code: int | None) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if runner_exit_code not in (0, None):
        blockers.append({"component": "official_runner", "code": "legacy_runner_nonzero_exit", "exit_code": runner_exit_code})
    infrastructure_failure_count = report.get("infrastructure_failure_count")
    if isinstance(infrastructure_failure_count, int) and infrastructure_failure_count > 0:
        blockers.append({"component": "reference_runtime", "code": "sample_infrastructure_failures", "count": infrastructure_failure_count})
    run_summary = report.get("run_summary") or {}
    if isinstance(run_summary.get("failed"), int) and run_summary["failed"] > 0:
        blockers.append({"component": "reference_runtime", "code": "run_summary_failed_samples", "count": run_summary["failed"]})
    if report.get("status") == "runner_failed_before_report":
        blockers.append({"component": "official_runner", "code": "runner_failed_before_report"})
    return blockers


def _official_command(split: str, output_root: Path) -> str:
    return (
        "uv run python scripts/run_core_rag_benchmark.py "
        f"--benchmark-id {BENCHMARK_ID} --benchmark-version {BENCHMARK_VERSION} "
        f"--split {split} --reference-runtime v1 --output-root {relative_path(output_root.resolve()) if ROOT.resolve() in output_root.resolve().parents else output_root.as_posix()}"
    )


def _markdown_baseline_report(aggregate: dict[str, Any], runtime_manifest: dict[str, Any]) -> str:
    blockers = aggregate["infrastructure_failures"]
    blocker_lines = "\n".join(f"- `{item['component']}`: `{item['code']}`" for item in blockers) or "- None"
    return f"""# Core RAG Benchmark v1 {aggregate['split']} Baseline

- Benchmark: `{BENCHMARK_ID}` `{BENCHMARK_VERSION}`
- Run ID: `{aggregate['run_id']}`
- Status: `{aggregate['status']}`
- Reference runtime: `{runtime_manifest['runtime_id']}`
- Sample count: {aggregate['sample_count']}
- Infrastructure failures: {aggregate['infrastructure_failure_count']}
- Publication eligible: {str(aggregate['publication_eligible']).lower()}

## Blockers

{blocker_lines}

## Remaining Acceptance Command

```bash
{aggregate['remaining_acceptance_command']}
```
"""


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else True


def verify_candidate_run(run_dir: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    aggregates = sorted(run_dir.glob("*_baseline_aggregate.json"))
    results = sorted(run_dir.glob("*_baseline_results.jsonl"))
    reports = sorted(run_dir.glob("*_baseline_report.md"))
    runtimes = sorted(run_dir.glob("*_reference_runtime_v1.json"))
    for label, files in {"aggregate": aggregates, "results": results, "markdown_report": reports, "runtime": runtimes}.items():
        if len(files) != 1:
            issues.append({"code": f"{label}_file_count", "message": f"Expected exactly one {label} file.", "observed": len(files)})
    aggregate = read_json(aggregates[0]) if len(aggregates) == 1 else {}
    sample_rows = read_jsonl(results[0]) if len(results) == 1 else []
    required = set((read_json(core_path("report_contract")).get("required_identity_fields") or []))
    missing = sorted(field for field in required if field not in aggregate and field not in (aggregate.get("runtime_fingerprints") or {}))
    if missing:
        issues.append({"code": "missing_required_report_fields", "message": "Aggregate missing required report fields.", "fields": missing})
    if aggregate.get("sample_count") != len(sample_rows):
        issues.append({"code": "sample_count_mismatch", "message": "Aggregate sample_count does not match results rows."})
    if aggregate.get("completed_sample_count") != sum(1 for row in sample_rows if row.get("execution_status") == "completed"):
        issues.append({"code": "completed_count_mismatch", "message": "Aggregate completed count does not match results rows."})
    if aggregate.get("infrastructure_failure_count") != sum(1 for row in sample_rows if row.get("execution_status") == "infrastructure_failure"):
        issues.append({"code": "infrastructure_count_mismatch", "message": "Aggregate infrastructure count does not match results rows."})
    current_annotation_digest = file_digest(core_path("annotations")) if core_path("annotations").exists() else None
    aggregate_annotation_digest = (aggregate.get("runtime_fingerprints") or {}).get("annotation_digest") or aggregate.get("current_annotation_digest")
    if not aggregate_annotation_digest:
        issues.append({"code": "annotation_digest_binding_missing", "message": "Candidate aggregate is missing annotation digest binding."})
    elif aggregate_annotation_digest != current_annotation_digest:
        issues.append({"code": "annotation_digest_binding_stale", "message": "Candidate aggregate is bound to a stale annotation digest."})
    privacy_scan = scan_paths_for_privacy([run_dir])
    if privacy_scan["status"] != "pass":
        issues.extend(privacy_scan["findings"])
    publication_eligible = aggregate.get("publication_eligible") is True and not issues and aggregate.get("status") == "completed"
    return {
        "schema_version": "opk-rag.core-rag-candidate-verification.v1",
        "status": "valid" if not issues else "invalid",
        "publication_eligible": publication_eligible,
        "run_dir": run_dir.as_posix(),
        "issues": issues,
        "privacy_scan": privacy_scan,
    }


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Core RAG Benchmark v1 static integrity.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--owner-review", choices=["generate", "validate"], help="Generate or validate the Benchmark Owner annotation review workflow.")
    args = parser.parse_args(argv)
    if args.owner_review == "generate":
        result = build_owner_review_artifacts()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "valid" else 1
    if args.owner_review == "validate":
        result = validate_owner_review()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "valid" else 1
    result = verify_core_benchmark()
    if args.json:
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"core rag benchmark: {result.status}")
        for issue in result.issues:
            print(f"- {issue.severity}:{issue.code} [{issue.path}]: {issue.message}")
    return result.exit_code


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Core RAG Benchmark v1 official baseline.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--benchmark-version", required=True)
    parser.add_argument("--split", required=True, choices=sorted(VALID_SPLITS))
    parser.add_argument("--reference-runtime", required=True, choices=["v1"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    aggregate = run_official_baseline(
        benchmark_id=args.benchmark_id,
        benchmark_version=args.benchmark_version,
        split=args.split,
        reference_runtime=args.reference_runtime,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if aggregate["status"] == "acceptance_blocked" else 0


def verify_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Non-mutating verification for a Core RAG Benchmark candidate run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify_candidate_run(args.run_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"core rag candidate run: {result['status']}")
        for issue in result["issues"]:
            print(f"- {issue.get('code')}: {issue.get('message')}")
    return 0 if result["status"] == "valid" else 1


def build_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Core RAG Benchmark v1 artifacts.")
    parser.add_argument("--publication-state", default="implementation_complete", choices=sorted(PUBLICATION_STATES))
    args = parser.parse_args(argv)
    manifest = build_all_artifacts(publication_state=args.publication_state)
    print(json.dumps({"status": "built", "manifest_digest": stable_hash(manifest)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
