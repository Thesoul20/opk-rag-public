from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation.holdout_exposure_governance import (
    INCIDENT_ID,
    INCIDENT_PATH,
    iter_public_audit_files,
    validate_incident_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "evaluation-data" / "benchmark_registry.json"
BASELINE_CONTRACT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json"
PUBLIC_CORPUS_SNAPSHOT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
SCORING_CONTRACT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"
PUBLIC_HOLDOUT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_holdout_public.json"

SCHEMA_REGISTRY = "opk-rag.benchmark-registry.v1"
SCHEMA_BASELINE = "opk-rag.phase2-dogfooding-baseline.v1"
SCHEMA_CORPUS = "opk-rag.phase2-corpus-snapshot-public.v1"
SCHEMA_SCORING = "opk-rag.phase2-scoring-contract.v1"

VALID_ROLES = {
    "development",
    "known_regression",
    "sealed_holdout",
    "sealed_baseline_result",
    "stage_acceptance_decision",
    "development_diagnostic",
    "development_retrieval_baseline",
    "development_representation_experiment",
    "evaluation_infrastructure",
    "legacy_reference",
}
VALID_EXPOSURE = {"exposed", "sealed", "sealed_private", "private_exposed", "public_summary_only", "unknown"}
INCOMPLETE_CODES = {"holdout_missing", "corpus_private_manifest_missing", "indexed_corpus_unverified"}
NON_SAMPLE_ARTIFACT_TYPES = {"contract", "governance_decision", "diagnostic_result", "execution_contract"}
REQUIRED_EVIDENCE_DIAGNOSTIC_ID = "phase2-required-evidence-diagnostic-v1"

SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}",
        r"password\s*[:=]\s*['\"]?[^'\"\s]{8,}",
        r"authorization\s*:\s*bearer\s+[A-Za-z0-9_\-.]{12,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"postgres(?:ql)?://[^:\s]+:[^@\s]+@",
    )
]
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|data|mnt|Volumes|var|private)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
]


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    status: str
    exit_code: int
    issues: tuple[ValidationIssue, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def count_samples(path: Path) -> int | None:
    if path.suffix.lower() == ".jsonl":
        return len(read_jsonl(path))
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("samples", "questions", "results", "cases", "records", "benchmarks"):
                value = data.get(key)
                if isinstance(value, list):
                    return len(value)
            dataset_summary = data.get("dataset_summary")
            if isinstance(dataset_summary, dict) and isinstance(dataset_summary.get("total_samples"), int):
                return int(dataset_summary["total_samples"])
            if isinstance(data.get("total_samples"), int):
                return int(data["total_samples"])
            if isinstance(data.get("sample_count"), int):
                return int(data["sample_count"])
        return 1
    if path.suffix.lower() == ".csv":
        return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)
    return None


def content_checksum(paths: Iterable[Path]) -> str:
    payload: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        payload.append(
            {
                "path": relative_path(path),
                "sha256": file_checksum(path),
                "sample_count": count_samples(path),
            }
        )
    return stable_hash(payload)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path: str) -> Path:
    candidate = (ROOT / path).resolve()
    if ROOT.resolve() not in candidate.parents and candidate != ROOT.resolve():
        raise ValueError(f"path escapes repository root: {path}")
    return candidate


def classify_asset(path: Path) -> tuple[str, str, bool, str]:
    rel = relative_path(path)
    private = "/private/" in f"/{rel}"
    name = path.name.lower()
    if private:
        return "known_regression", "private_exposed", True, "private real Vault dogfooding artifact"
    if rel in {"evaluation-data/answerability_dev.jsonl"}:
        return "development", "exposed", True, "answerability development split"
    if rel in {"evaluation-data/answerability_test.jsonl"}:
        return "known_regression", "exposed", True, "original test split exposed by repeated task reports"
    if rel in {"evaluation-data/eval_dataset.jsonl", "evaluation-data/eval_dataset.md"}:
        return "legacy_reference", "exposed", True, "early QA evaluation asset"
    if rel.startswith("evaluation-data/results/") or "_report" in name or "_results" in name:
        return "known_regression", "exposed", private, "historical formal evaluation output"
    if rel.startswith("evaluation-data/dogfooding/"):
        return "known_regression", "public_summary_only", private, "dogfooding schema, manifest, or sanitized summary"
    if rel.startswith("evaluation-data/"):
        return "legacy_reference", "exposed", True, "checked-in evaluation asset"
    if rel.startswith("source-documents/"):
        return "legacy_reference", "exposed", True, "source corpus used by legacy and answerability evaluations"
    if rel.startswith(("docs/", "scripts/", "tests/", "tasks/")):
        return "legacy_reference", "exposed", False, "evaluation support artifact"
    return "legacy_reference", "unknown", private, "unclassified evaluation-adjacent asset"


def build_asset_inventory() -> list[dict[str, Any]]:
    roots = ["evaluation-data", "source-documents", "docs", "scripts", "tests", "tasks"]
    suffixes = {".jsonl", ".json", ".md", ".csv", ".py"}
    rows: list[dict[str, Any]] = []
    for path in iter_public_audit_files((ROOT / root_name for root_name in roots), suffixes=suffixes, root=ROOT):
        rel = relative_path(path)
        role, exposure, private, usage = classify_asset(path)
        rows.append(
            {
                "asset_id": hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16],
                "path": rel,
                "format": path.suffix.lower().lstrip(".") or "unknown",
                "sample_count": count_samples(path),
                "role": role,
                "exposure_status": exposure,
                "contains_private_data": private,
                "current_usage": usage,
                "checksum": file_checksum(path),
                "notes": "Inventory row generated from repository scan; private paths are excluded by the public audit iterator.",
            }
        )
    return rows


def benchmark_definitions() -> list[dict[str, Any]]:
    return [
        {
            "benchmark_id": "early-qa-eval-v1",
            "role": "legacy_reference",
            "source_files": ["evaluation-data/eval_dataset.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["historical comparison", "legacy regression context"],
            "forbidden_use": ["independent generalization claim", "sealed holdout claim"],
            "contains_private_data": True,
            "notes": "Early 30-case QA dataset; manifest marks candidate, private, and not approved for public release.",
        },
        {
            "benchmark_id": "answerability-canonical-v1",
            "role": "legacy_reference",
            "source_files": ["evaluation-data/answerability_dataset.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["split provenance", "historical aggregate accounting"],
            "forbidden_use": ["sealed holdout claim"],
            "contains_private_data": True,
            "notes": "Canonical 40-row Answerability dataset; current roles are assigned at split level.",
        },
        {
            "benchmark_id": "answerability-dev-v1",
            "role": "development",
            "source_files": ["evaluation-data/answerability_dev.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["diagnosis", "prompt and policy experiments", "development evaluation"],
            "forbidden_use": ["independent test claim", "sealed holdout claim"],
            "contains_private_data": True,
            "notes": "28-row development split.",
        },
        {
            "benchmark_id": "answerability-test-known-regression-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/answerability_test.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["known regression prevention", "public per-row diagnostics"],
            "forbidden_use": ["unseen test claim", "sealed holdout claim"],
            "contains_private_data": True,
            "notes": "12-row original test split; exposed by multiple TASK-0020 through TASK-0026 reports.",
        },
        {
            "benchmark_id": "conversation-eval-v1",
            "role": "legacy_reference",
            "source_files": ["evaluation-data/conversation_dataset.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["conversation behavior reference"],
            "forbidden_use": ["Phase 2 dogfooding primary benchmark"],
            "contains_private_data": True,
            "notes": "42 JSONL rows corresponding to 12 sessions and 30 follow-up turns in the baseline report.",
        },
        {
            "benchmark_id": "citation-grounding-mvp-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/citation_grounding_mvp_report.json"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["deterministic citation and grounding guard regression"],
            "forbidden_use": ["semantic answer quality claim"],
            "contains_private_data": False,
            "notes": "10 deterministic citation/grounding cases.",
        },
        {
            "benchmark_id": "retrieval-dev-fixture-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/results/task0027_retrieval_dev_fixture.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["retrieval fixture replay", "known retrieval regression"],
            "forbidden_use": ["unseen retrieval test claim"],
            "contains_private_data": True,
            "notes": "28-row frozen retrieval fixture from the exposed development split.",
        },
        {
            "benchmark_id": "generation-stability-dev-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/results/task0028_generation_stability_baseline.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["fixed evidence generation stability regression"],
            "forbidden_use": ["independent model capability claim"],
            "contains_private_data": True,
            "notes": "140 generated rows from repeated exposed development fixture runs.",
        },
        {
            "benchmark_id": "target-model-dev-baseline-v1",
            "role": "development",
            "source_files": ["evaluation-data/results/task0031_target_dev_baseline.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["target model development baseline"],
            "forbidden_use": ["sealed holdout claim"],
            "contains_private_data": True,
            "notes": "28-row target model baseline over exposed development data.",
        },
        {
            "benchmark_id": "residual-abstention-known-regression-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/results/task0032_residual_abstention_audit.jsonl"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["known residual abstention regression"],
            "forbidden_use": ["unseen failure discovery claim"],
            "contains_private_data": True,
            "notes": "13 exposed residual abstention samples and diagnostics.",
        },
        {
            "benchmark_id": "real-vault-dogfooding-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/dogfooding/private/real_vault_questions.jsonl"],
            "exposure_status": "private_exposed",
            "mutable": False,
            "intended_use": ["private known dogfooding regression", "sanitized aggregate reporting"],
            "forbidden_use": ["public per-row disclosure", "sealed holdout claim"],
            "contains_private_data": True,
            "notes": "36-row local private real Vault question set; sample IDs and reports have been used in TASK-0039 through TASK-0044.",
        },
        {
            "benchmark_id": "task0044-fixed-evidence-private-v1",
            "role": "known_regression",
            "source_files": ["evaluation-data/dogfooding/private/task0044/fixed_replay_inputs.jsonl"],
            "exposure_status": "private_exposed",
            "mutable": False,
            "intended_use": ["fixed evidence provider variance diagnosis"],
            "forbidden_use": ["sealed holdout claim", "public evidence disclosure"],
            "contains_private_data": True,
            "notes": "29 private fixed-evidence replay inputs reconstructed from exposed dogfooding runs.",
        },
        {
            "benchmark_id": "phase2-dogfooding-scoring-v1",
            "role": "known_regression",
            "artifact_type": "contract",
            "source_files": ["evaluation-data/dogfooding/phase2_scoring_contract.json"],
            "exposure_status": "public_summary_only",
            "mutable": False,
            "intended_use": ["Phase 2 scoring contract validation", "holdout annotation schema binding", "baseline quality gate evaluation"],
            "forbidden_use": ["dataset sample count claim", "sealed holdout sample source", "model quality claim by itself"],
            "contains_private_data": False,
            "notes": "Versioned scoring contract only; it contains no real holdout samples and no private Vault text.",
        },
        {
            "benchmark_id": "phase2-dogfooding-holdout-v1",
            "role": "sealed_holdout",
            "artifact_type": "sealed_holdout_manifest",
            "source_files": ["evaluation-data/dogfooding/phase2_holdout_public.json"],
            "exposure_status": "exposed",
            "mutable": False,
            "intended_use": ["historical Phase 2 sealed dogfooding baseline provenance only"],
            "forbidden_use": [
                "daily development",
                "prompt tuning",
                "retrieval tuning",
                "answerability tuning",
                "citation or grounding tuning",
                "new stage acceptance",
                "public per-row reporting",
                "soft target setting before formal baseline governance",
            ],
            "contains_private_data": True,
            "public_manifest_path": "evaluation-data/dogfooding/phase2_holdout_public.json",
            "holdout_id": "phase2-dogfooding-holdout-v1",
            "corpus_snapshot_id": "phase2-corpus-v1",
            "scoring_contract_id": "phase2-dogfooding-scoring-v1",
            "private_data": {
                "storage": "gitignored private evaluation assets",
                "exposure_status": "exposed",
                "per_sample_reporting_allowed": False,
                "contains_question_text_publicly": False,
                "contains_gold_claims_publicly": False,
                "contains_source_paths_publicly": False,
            },
            "downgrade_policy": "If per-sample sealed results drive system changes, downgrade this holdout version to known_regression and create a new holdout version.",
            "future_acceptance_eligible": False,
            "historical_sealed_baseline_validity": "retained_as_historical_run",
            "future_stage_acceptance_eligibility": "revoked_after_exposure",
            "replacement_holdout_required": True,
            "replacement_holdout_id": "pending",
            "replacement_holdout_creation_status": "pending_independent_process",
            "incident_id": INCIDENT_ID,
            "incident_artifact": "evaluation-data/governance/phase2_holdout_v1_exposure_incident.json",
            "notes": "Public manifest only. Holdout v1 had accidental content read during TASK-0059, was not used for implementation, parameter selection, or metrics, and is not eligible for future acceptance.",
        },
        {
            "benchmark_id": "phase2-sealed-baseline-v1",
            "role": "sealed_baseline_result",
            "artifact_type": "evaluation_result",
            "source_files": ["evaluation-data/results/phase2_sealed_baseline_v1_summary.json"],
            "exposure_status": "public_summary_only",
            "mutable": False,
            "intended_use": ["observed sealed Phase 2 baseline comparison", "aggregate regression comparison for future versions"],
            "forbidden_use": [
                "prompt tuning from per-sample holdout output",
                "retrieval tuning from per-sample holdout output",
                "public per-row reporting",
                "soft target threshold setting by copying holdout observations",
            ],
            "contains_private_data": False,
            "public_summary_path": "evaluation-data/results/phase2_sealed_baseline_v1_summary.json",
            "notes": "Sanitized aggregate result artifact only; private sample outputs remain under gitignored .private/evaluation.",
        },
    ]


def build_registry() -> dict[str, Any]:
    benchmarks = []
    for definition in benchmark_definitions():
        paths = [resolve_repo_path(path) for path in definition["source_files"]]
        sample_counts = [count_samples(path) for path in paths if path.exists()]
        countable = [count for count in sample_counts if count is not None]
        benchmark = dict(definition)
        benchmark["sample_count"] = None if benchmark.get("artifact_type") in NON_SAMPLE_ARTIFACT_TYPES else (countable[0] if countable else 0)
        benchmark["content_checksum"] = content_checksum(paths) if all(path.exists() for path in paths) else ""
        benchmarks.append(benchmark)

    return {
        "schema_version": SCHEMA_REGISTRY,
        "generated_at": utc_now(),
        "audit_scope": ["evaluation-data/", "source-documents/", "docs/", "scripts/", "tests/", "tasks/"],
        "governance_notes": {
            "holdout_v1_status": "exposed",
            "holdout_v1_future_acceptance_eligible": False,
            "answerability_dev_role": "development",
            "answerability_test_role": "known_regression",
            "early_qa_role": "legacy_reference",
            "holdout_exposure_incident_id": INCIDENT_ID,
        },
        "benchmarks": benchmarks,
        "asset_inventory": build_asset_inventory(),
    }


def build_runtime_contract() -> dict[str, Any]:
    embedding_contract = {
        "schema_version": "2026-07-17",
        "parser_version": "markdown-parser-v1",
        "chunking_version": "markdown-chunker-v1",
        "embedding_provider": "local_qwen",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "embedding_dimension": 1024,
        "normalize": True,
        "distance_metric": "cosine",
        "input_template_version": "embedding-input-v1",
        "max_input_tokens": 8192,
    }
    answer_contract = {
        "prompt_version": "answer-prompt-v4",
        "output_schema_version": "answer-response-v3",
        "evidence_context_version": "evidence-context-v1",
        "answerability_contract": "AnswerabilityConfig defaults",
        "citation_grounding_contract": "GroundingValidationConfig defaults + unsupported-claim validation",
    }
    retrieval_contract = {
        "mode": "vector",
        "top_k": 5,
        "candidate_k": 20,
        "rerank_enabled": False,
        "context_token_budget": 4096,
        "context_max_chunks": 5,
        "context_max_chunks_per_document": 2,
        "deduplicate": True,
        "overlap_deduplication_threshold": 0.8,
        "query_template_version": "query-input-v1",
    }
    runtime = {
        "vault": "local",
        "database": "remote-supabase",
        "database_engine": "PostgreSQL + pgvector",
        "supabase_connection_config_env": ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY"],
        "llm_provider": "deepseek",
        "llm_model_env": "OPK_RAG_LLM_MODEL",
        "llm_base_url_env": "OPK_RAG_LLM_BASE_URL",
        "llm_thinking_mode_env": "OPK_RAG_LLM_THINKING_MODE",
        "llm_response_format_env": "OPK_RAG_LLM_RESPONSE_FORMAT",
        "llm_secret_env": "OPK_RAG_LLM_API_KEY",
        "embedding_model_env": "OPK_RAG_EMBEDDING_MODEL",
        "embedding_dimension": 1024,
        "embedding_model": embedding_contract["embedding_model"],
        "embedding_model_revision": embedding_contract["model_revision"],
        "embedding_normalize": embedding_contract["normalize"],
        "embedding_distance_metric": embedding_contract["distance_metric"],
        "chunking_version_env": "OPK_RAG_CHUNKING_VERSION",
        "chunking_version": embedding_contract["chunking_version"],
        "parser_version": embedding_contract["parser_version"],
        "retrieval_env": [
            "OPK_RAG_SEARCH_MODE",
            "OPK_RAG_SEARCH_TOP_K",
            "OPK_RAG_SEARCH_CANDIDATE_K",
            "OPK_RAG_RERANK_ENABLED",
            "OPK_RAG_CONTEXT_MAX_CHUNKS",
            "OPK_RAG_CONTEXT_MAX_CHUNKS_PER_DOCUMENT",
        ],
        "generation_env": [
            "OPK_RAG_ANSWER_PROMPT_VERSION",
            "OPK_RAG_ANSWER_OUTPUT_SCHEMA_VERSION",
            "OPK_RAG_GROUNDING_VALIDATION_ENABLED",
        ],
        "config_source_priority": ["shell", "dotenv", "default", "compatibility_alias", "missing"],
        "embedding_contract_digest": stable_hash(embedding_contract),
        "retrieval_contract_digest": stable_hash(retrieval_contract),
        "generation_contract_digest": stable_hash(answer_contract),
        "embedding_contract": embedding_contract,
        "retrieval_contract": retrieval_contract,
        "generation_contract": answer_contract,
    }
    runtime["runtime_contract_digest"] = stable_hash(runtime)
    return runtime


def build_public_corpus_snapshot() -> dict[str, Any]:
    run_config_path = ROOT / "evaluation-data" / "dogfooding" / "private" / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    chunking_digest = stable_hash(
        {
            "parser_version": run_config.get("parser_version", "markdown-parser-v1"),
            "chunking_version": run_config.get("chunking_version", "markdown-chunker-v1"),
        }
    )
    embedding_digest = stable_hash(
        {
            "embedding_model": run_config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            "embedding_dimension": run_config.get("embedding_dimension", 1024),
            "normalize": True,
            "distance_metric": "cosine",
        }
    )
    return {
        "schema_version": SCHEMA_CORPUS,
        "snapshot_id": run_config.get("snapshot_id", "missing-private-snapshot"),
        "created_at": run_config.get("created_at", utc_now()),
        "status": "missing_private_manifest",
        "markdown_file_count": run_config.get("markdown_file_count"),
        "total_byte_count": None,
        "content_digest": run_config.get("content_hash_aggregate", ""),
        "chunking_contract_digest": chunking_digest,
        "embedding_contract_digest": embedding_digest,
        "private_manifest_present": False,
        "private_manifest_path": "evaluation-data/dogfooding/private/phase2_corpus_snapshot_manifest.json",
        "absolute_paths_redacted": True,
        "contains_private_filenames": False,
        "notes": "Public summary is derived from sanitized local run config only. A per-file private manifest is not present, so drift cannot be fully verified yet.",
    }


def build_baseline_contract(registry: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    benchmark_ids = {item["benchmark_id"]: item for item in registry["benchmarks"]}
    runtime = build_runtime_contract()
    contract = {
        "schema_version": SCHEMA_BASELINE,
        "baseline_id": "phase2-dogfooding-v1",
        "status": "incomplete",
        "created_at": utc_now(),
        "evaluation_roles": {
            "development": ["answerability-dev-v1", "target-model-dev-baseline-v1"],
            "known_regression": [
                "answerability-test-known-regression-v1",
                "citation-grounding-mvp-v1",
                "retrieval-dev-fixture-v1",
                "generation-stability-dev-v1",
                "residual-abstention-known-regression-v1",
                "real-vault-dogfooding-v1",
                "task0044-fixed-evidence-private-v1",
                "phase2-dogfooding-scoring-v1",
            ],
            "sealed_holdout": [],
            "legacy_reference": ["early-qa-eval-v1", "answerability-canonical-v1", "conversation-eval-v1"],
        },
        "corpus_snapshot": {
            "status": "missing_private_manifest",
            "public_manifest": relative_path(PUBLIC_CORPUS_SNAPSHOT_PATH),
            "private_manifest_required": True,
            "snapshot_id": corpus["snapshot_id"],
            "content_digest": corpus["content_digest"],
        },
        "runtime_contract": runtime,
        "holdout": {
            "status": "missing",
            "reason": "No truly unexposed sealed Phase 2 holdout is present in the audited repository or local private dogfooding assets.",
        },
        "scoring_contract": {
            "status": "frozen",
            "contract_id": "phase2-dogfooding-scoring-v1",
            "path": relative_path(SCORING_CONTRACT_PATH),
            "content_digest": file_checksum(SCORING_CONTRACT_PATH) if SCORING_CONTRACT_PATH.exists() else "",
            "schema_version": SCHEMA_SCORING,
            "informational_targets": [
                "end_to_end_success_rate",
                "fully_answerable_answer_rate",
                "partial_answer_success_rate",
                "over_abstention_rate",
                "required_evidence_coverage",
            ],
            "default_report_visibility": {
                "development": "per-row allowed",
                "known_regression": "per-row allowed unless private; private defaults to sanitized aggregates",
                "sealed_holdout": "aggregate only by default",
            },
        },
        "registry_digest": stable_hash(
            [
                {
                    "benchmark_id": item["benchmark_id"],
                    "role": item["role"],
                    "sample_count": item["sample_count"],
                    "content_checksum": item["content_checksum"],
                }
                for item in registry["benchmarks"]
            ]
        ),
        "notes": "Status remains incomplete because sealed_holdout is missing and the real Vault corpus lacks a private per-file manifest.",
    }
    contract["baseline_contract_digest"] = stable_hash(
        {k: v for k, v in contract.items() if k != "baseline_contract_digest"}
    )
    missing = [bid for role_ids in contract["evaluation_roles"].values() for bid in role_ids if bid not in benchmark_ids]
    if missing:
        contract["status"] = "blocked"
        contract["notes"] += f" Missing benchmark IDs: {', '.join(sorted(missing))}."
    return contract


def write_artifacts() -> None:
    registry = build_registry()
    corpus = build_public_corpus_snapshot()
    baseline = build_baseline_contract(registry, corpus)
    for path, payload in (
        (REGISTRY_PATH, registry),
        (PUBLIC_CORPUS_SNAPSHOT_PATH, corpus),
        (BASELINE_CONTRACT_PATH, baseline),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def contains_secret_like_text(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def contains_absolute_path_text(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS)


def is_git_tracked(path: Path) -> bool:
    rel = relative_path(path)
    result = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def validate_registry(registry: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if registry.get("schema_version") != SCHEMA_REGISTRY:
        issues.append(ValidationIssue("error", "registry_schema", "Invalid registry schema version.", relative_path(REGISTRY_PATH)))
    for benchmark in registry.get("benchmarks", []):
        bid = str(benchmark.get("benchmark_id", ""))
        role = benchmark.get("role")
        exposure = benchmark.get("exposure_status")
        if role not in VALID_ROLES:
            issues.append(ValidationIssue("error", "invalid_role", f"{bid} has invalid role {role!r}."))
        if exposure not in VALID_EXPOSURE:
            issues.append(ValidationIssue("error", "invalid_exposure", f"{bid} has invalid exposure status {exposure!r}."))
        documented_exposure = (
            bid == "phase2-dogfooding-holdout-v1"
            and exposure == "exposed"
            and benchmark.get("incident_id") == INCIDENT_ID
            and benchmark.get("future_acceptance_eligible") is False
        )
        if role == "sealed_holdout" and exposure in {"exposed", "private_exposed"} and not documented_exposure:
            issues.append(ValidationIssue("error", "exposed_holdout", f"{bid} marks exposed data as sealed holdout."))
        paths = [resolve_repo_path(path) for path in benchmark.get("source_files", [])]
        missing_paths = [path for path in paths if not path.exists()]
        if missing_paths:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_source_file",
                    f"{bid} references missing source files: {', '.join(relative_path(path) for path in missing_paths)}.",
                )
            )
            continue
        countable = [count_samples(path) for path in paths]
        counts = [count for count in countable if count is not None]
        if benchmark.get("artifact_type") in NON_SAMPLE_ARTIFACT_TYPES and benchmark.get("artifact_type") != "diagnostic_result":
            if benchmark.get("sample_count") not in {None, 0}:
                issues.append(
                    ValidationIssue(
                        "error",
                        "non_sample_artifact_sample_count_present",
                        f"{bid} is a non-sample artifact and must not register dataset sample_count.",
                    )
                )
        elif benchmark.get("artifact_type") == "sealed_holdout_manifest":
            public_path = paths[0] if paths else None
            if public_path and public_path.exists():
                public_manifest = read_json(public_path)
                if benchmark.get("sample_count") != public_manifest.get("sample_count"):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "sample_count_mismatch",
                            f"{bid} sample_count={benchmark.get('sample_count')} but manifest has {public_manifest.get('sample_count')}.",
                        )
                    )
        elif counts and benchmark.get("sample_count") != counts[0]:
            issues.append(
                ValidationIssue(
                    "error",
                    "sample_count_mismatch",
                    f"{bid} sample_count={benchmark.get('sample_count')} but computed {counts[0]}.",
                )
            )
        expected_checksum = content_checksum(paths)
        if benchmark.get("content_checksum") != expected_checksum:
            issues.append(ValidationIssue("error", "checksum_mismatch", f"{bid} checksum drift detected."))
        for path in paths:
            if "/private/" in f"/{relative_path(path)}" and is_git_tracked(path):
                issues.append(
                    ValidationIssue(
                        "error",
                        "private_manifest_tracked",
                        f"Private benchmark source is tracked by Git: {relative_path(path)}.",
                    )
                )
    return issues


def validate_baseline(contract: dict[str, Any], registry: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if contract.get("schema_version") != SCHEMA_BASELINE:
        issues.append(ValidationIssue("error", "baseline_schema", "Invalid baseline contract schema."))
    benchmark_ids = {item.get("benchmark_id") for item in registry.get("benchmarks", [])}
    for role, ids in contract.get("evaluation_roles", {}).items():
        if role not in VALID_ROLES:
            issues.append(ValidationIssue("error", "invalid_baseline_role", f"Invalid baseline role {role!r}."))
        for bid in ids:
            if bid not in benchmark_ids:
                issues.append(ValidationIssue("error", "missing_benchmark_reference", f"Baseline references unknown benchmark {bid}."))
    runtime = contract.get("runtime_contract")
    required_runtime = {
        "vault",
        "database",
        "database_engine",
        "llm_provider",
        "llm_model_env",
        "embedding_model_env",
        "embedding_dimension",
        "runtime_contract_digest",
        "retrieval_contract_digest",
        "generation_contract_digest",
    }
    if not isinstance(runtime, dict):
        issues.append(ValidationIssue("error", "runtime_contract_missing", "Runtime contract is missing."))
    else:
        missing = sorted(required_runtime - set(runtime))
        if missing:
            issues.append(ValidationIssue("error", "runtime_required_fields", f"Runtime contract missing: {', '.join(missing)}."))
    if contract.get("holdout", {}).get("status") == "missing":
        issues.append(ValidationIssue("incomplete", "holdout_missing", "No sealed holdout is currently registered."))
    elif contract.get("holdout", {}).get("status") == "exposed":
        holdout = contract.get("holdout", {})
        expected = {
            "holdout_v1_exposure_confirmed": True,
            "holdout_v1_used_for_implementation": False,
            "holdout_v1_used_for_parameter_selection": False,
            "holdout_v1_used_for_metric_calculation": False,
            "holdout_v1_future_acceptance_eligible": False,
            "replacement_holdout_required": True,
        }
        for key, expected_value in expected.items():
            if holdout.get(key) != expected_value:
                issues.append(ValidationIssue("error", f"holdout_exposure_{key}_mismatch", f"holdout.{key} must be {expected_value!r}."))
    elif contract.get("holdout", {}).get("status") == "sealed":
        try:
            from opk_rag.evaluation.phase2_holdout import PUBLIC_HOLDOUT_PATH as HOLDOUT_PUBLIC_PATH
            from opk_rag.evaluation.phase2_holdout import verify_holdout_seal

            holdout = contract.get("holdout", {})
            public_manifest_path = holdout.get("public_manifest")
            if public_manifest_path != relative_path(HOLDOUT_PUBLIC_PATH):
                issues.append(ValidationIssue("error", "holdout_public_manifest_mismatch", "Baseline references the wrong holdout public manifest."))
            result = verify_holdout_seal()
            if result.status != "sealed":
                issues.append(ValidationIssue("error", "holdout_not_sealed", "Registered holdout seal verification failed."))
            public_manifest = read_json(HOLDOUT_PUBLIC_PATH)
            if holdout.get("holdout_id") != public_manifest.get("holdout_id"):
                issues.append(ValidationIssue("error", "holdout_id_mismatch", "Baseline holdout id does not match public manifest."))
            if holdout.get("sample_count") != public_manifest.get("sample_count"):
                issues.append(ValidationIssue("error", "holdout_sample_count_mismatch", "Baseline holdout sample count does not match public manifest."))
            if holdout.get("corpus_snapshot_id") != public_manifest.get("corpus_snapshot_id"):
                issues.append(ValidationIssue("error", "holdout_corpus_snapshot_mismatch", "Baseline holdout corpus snapshot id does not match public manifest."))
            if holdout.get("scoring_contract_id") != public_manifest.get("scoring_contract_id"):
                issues.append(ValidationIssue("error", "holdout_scoring_contract_mismatch", "Baseline holdout scoring contract id does not match public manifest."))
            for issue in result.issues:
                if issue.severity == "error":
                    issues.append(ValidationIssue("error", issue.code, issue.message, issue.path))
        except Exception as exc:
            issues.append(ValidationIssue("error", "holdout_validation_error", f"Could not validate sealed holdout: {exc}"))
    scoring = contract.get("scoring_contract")
    if not isinstance(scoring, dict):
        issues.append(ValidationIssue("error", "scoring_contract_missing", "Baseline contract is missing scoring_contract."))
    else:
        if scoring.get("status") != "frozen":
            issues.append(
                ValidationIssue(
                    "error",
                    "scoring_contract_not_frozen",
                    f"Scoring contract status is {scoring.get('status')!r}; expected frozen.",
                )
            )
        if scoring.get("contract_id") != "phase2-dogfooding-scoring-v1":
            issues.append(ValidationIssue("error", "scoring_contract_id_mismatch", "Baseline references the wrong scoring contract id."))
        if scoring.get("schema_version") != SCHEMA_SCORING:
            issues.append(ValidationIssue("error", "scoring_contract_schema_mismatch", "Baseline references the wrong scoring contract schema."))
        scoring_path_value = scoring.get("path")
        if not isinstance(scoring_path_value, str):
            issues.append(ValidationIssue("error", "scoring_contract_path_missing", "Baseline scoring contract path is missing."))
        else:
            scoring_path = resolve_repo_path(scoring_path_value)
            if not scoring_path.exists():
                issues.append(ValidationIssue("error", "scoring_contract_file_missing", f"Scoring contract file is missing: {scoring_path_value}."))
            else:
                expected_digest = file_checksum(scoring_path)
                if scoring.get("content_digest") != expected_digest:
                    issues.append(ValidationIssue("error", "scoring_contract_digest_mismatch", "Scoring contract digest does not match referenced file."))
                try:
                    from opk_rag.evaluation.phase2_scoring import QUESTION_TYPES, validate_scoring_contract

                    scoring_contract = read_json(scoring_path)
                    scoring_result = validate_scoring_contract(scoring_contract, path=scoring_path_value)
                    for issue in scoring_result.issues:
                        if issue.severity == "error":
                            issues.append(ValidationIssue("error", issue.code, issue.message, issue.path))
                        elif issue.severity == "pending":
                            issues.append(ValidationIssue("pending", issue.code, issue.message, issue.path))
                    sample_schema = scoring_contract.get("sample_schema") if isinstance(scoring_contract, dict) else {}
                    if not sample_schema or not sample_schema.get("required_fields"):
                        issues.append(ValidationIssue("error", "holdout_schema_not_bound", "Holdout sample schema is not defined by scoring contract."))
                    if set((scoring_contract.get("question_types") or {}).keys()) != QUESTION_TYPES:
                        issues.append(ValidationIssue("error", "question_type_set_mismatch", "Scoring contract question type set is not exactly the Phase 2 v1 set."))
                except Exception as exc:
                    issues.append(ValidationIssue("error", "scoring_contract_validation_error", f"Could not validate scoring contract: {exc}"))
    if isinstance(contract.get("sealed_baseline"), dict):
        diagnostic = contract.get("required_evidence_diagnostic") or {}
        expected_diagnostic = {
            "diagnostic_artifact_id": REQUIRED_EVIDENCE_DIAGNOSTIC_ID,
            "root_cause_classification": "measurement_valid_retrieval_failure_confirmed",
            "required_evidence_metric_validity": "invalid_due_to_identity_mapping",
            "candidate_retrieval_failure": "confirmed",
            "evidence_bundle_loss": "present_secondary",
            "gold_annotation_compatibility": "verified_on_public_datasets",
            "sealed_baseline_rerun_required": "pending_governance_decision",
        }
        for key, expected in expected_diagnostic.items():
            if diagnostic.get(key) != expected:
                issues.append(ValidationIssue("error", f"required_evidence_diagnostic_{key}_mismatch", f"Baseline required_evidence_diagnostic.{key} must be {expected!r}."))
        override = (
            (contract.get("sealed_baseline") or {})
            .get("metric_validity_overrides", {})
            .get("required_evidence_coverage", {})
        )
        if override.get("historical_observation", {}).get("value") != 0.005555555555555556:
            issues.append(ValidationIssue("error", "required_evidence_observation_rewritten", "Historical required_evidence_coverage observation must remain 0.0056."))
        if override.get("quality_conclusion_validity") != "invalid":
            issues.append(ValidationIssue("error", "required_evidence_quality_validity_missing", "Required evidence observation must be marked invalid as a quality conclusion."))
    incident_path = INCIDENT_PATH
    if not incident_path.exists():
        issues.append(ValidationIssue("error", "holdout_exposure_incident_missing", "Holdout exposure incident artifact is missing.", relative_path(incident_path)))
    else:
        incident_issues = validate_incident_artifact(read_json(incident_path))
        for issue in incident_issues:
            issues.append(ValidationIssue("error", issue["code"], issue["message"], relative_path(incident_path)))
    corpus_status = contract.get("corpus_snapshot", {}).get("status")
    if corpus_status in {"missing_private_manifest", "missing", None}:
        issues.append(
            ValidationIssue(
                "incomplete",
                "corpus_private_manifest_missing",
                "Corpus snapshot is not fully bound to a private per-file manifest.",
            )
        )
    elif corpus_status == "frozen" and contract.get("corpus_snapshot", {}).get("indexed_corpus_status") in {
        "unchecked",
        "unavailable",
        "partial",
    }:
        issues.append(
            ValidationIssue(
                "incomplete",
                "indexed_corpus_unverified",
                "Source corpus is frozen, but indexed corpus consistency has not been fully verified.",
            )
        )
    return issues


def validate_public_files() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in (REGISTRY_PATH, BASELINE_CONTRACT_PATH, PUBLIC_CORPUS_SNAPSHOT_PATH, SCORING_CONTRACT_PATH):
        if not path.exists():
            issues.append(ValidationIssue("error", "public_file_missing", f"Missing public governance file {relative_path(path)}."))
            continue
        if contains_secret_like_text(path):
            issues.append(ValidationIssue("error", "secret_like_public_file", "Public governance file contains secret-like text.", relative_path(path)))
    if PUBLIC_CORPUS_SNAPSHOT_PATH.exists() and contains_absolute_path_text(PUBLIC_CORPUS_SNAPSHOT_PATH):
        issues.append(
            ValidationIssue(
                "error",
                "absolute_path_public_snapshot",
                "Public corpus snapshot contains an absolute path.",
                relative_path(PUBLIC_CORPUS_SNAPSHOT_PATH),
            )
        )
    if PUBLIC_HOLDOUT_PATH.exists():
        if contains_secret_like_text(PUBLIC_HOLDOUT_PATH):
            issues.append(ValidationIssue("error", "secret_like_public_file", "Public holdout manifest contains secret-like text.", relative_path(PUBLIC_HOLDOUT_PATH)))
        if contains_absolute_path_text(PUBLIC_HOLDOUT_PATH):
            issues.append(ValidationIssue("error", "absolute_path_public_holdout", "Public holdout manifest contains an absolute path.", relative_path(PUBLIC_HOLDOUT_PATH)))
    private_manifest = ROOT / ".private" / "evaluation" / "phase2_corpus_manifest.json"
    if private_manifest.exists() and is_git_tracked(private_manifest):
        issues.append(
            ValidationIssue(
                "error",
                "private_manifest_tracked",
                "Private corpus snapshot manifest must not be Git tracked.",
                relative_path(private_manifest),
            )
        )
    return issues


def validate_corpus_drift(*, check_index: bool = False) -> list[ValidationIssue]:
    from opk_rag.evaluation.corpus_snapshot import verify_snapshot

    result = verify_snapshot(check_index=check_index)
    issues: list[ValidationIssue] = []
    for issue in result.issues:
        code = "corpus_private_manifest_missing" if issue.code == "private_manifest_missing" else issue.code
        severity = "incomplete" if code in {"corpus_private_manifest_missing", "vault_unavailable", "index_unavailable"} else issue.severity
        issues.append(ValidationIssue(severity, code, issue.message, issue.path))
    return issues


def validate(check_corpus: bool = False, check_index: bool = False) -> ValidationResult:
    issues = validate_public_files()
    registry = read_json(REGISTRY_PATH) if REGISTRY_PATH.exists() else {}
    contract = read_json(BASELINE_CONTRACT_PATH) if BASELINE_CONTRACT_PATH.exists() else {}
    if registry:
        issues.extend(validate_registry(registry))
    if contract and registry:
        issues.extend(validate_baseline(contract, registry))
    if check_corpus:
        issues.extend(validate_corpus_drift(check_index=check_index))

    if any(issue.severity == "error" for issue in issues):
        return ValidationResult("invalid", 1, tuple(issues))
    incomplete = [issue for issue in issues if issue.severity == "incomplete"]
    if incomplete:
        return ValidationResult("incomplete", 2, tuple(issues))
    return ValidationResult("valid", 0, tuple(issues))


def print_text_result(result: ValidationResult) -> None:
    print(f"phase2 dogfooding baseline: {result.status}")
    for issue in result.issues:
        location = f" [{issue.path}]" if issue.path else ""
        print(f"- {issue.severity}:{issue.code}{location}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 2 dogfooding evaluation baseline governance.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation result.")
    parser.add_argument("--check-corpus", action="store_true", help="Check private corpus manifest drift when available.")
    parser.add_argument("--check-index", action="store_true", help="Opt in to read-only remote indexed corpus checks with --check-corpus.")
    parser.add_argument("--write-artifacts", action="store_true", help="Regenerate public governance artifacts.")
    args = parser.parse_args(argv)

    if args.write_artifacts:
        write_artifacts()

    result = validate(check_corpus=args.check_corpus, check_index=args.check_index)
    if args.json:
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
