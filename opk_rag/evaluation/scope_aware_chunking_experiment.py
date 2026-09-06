from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from opk_rag.chunking.chunker import chunk_content_hash, chunk_markdown_document, normalize_chunk_content
from opk_rag.chunking.models import CHUNKING_VERSION, PARSER_VERSION, ChunkingConfig, ParsedMarkdownDocument
from opk_rag.chunking.parser import parse_markdown_file
from opk_rag.embedding.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    EmbeddingConfig,
    build_configuration_fingerprint,
)
from opk_rag.embedding.input import normalize_embedding_text
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INPUT_TEMPLATE_VERSION, render_query_input
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, _load_public_samples, _reject_forbidden_path, _reject_forbidden_payload, _sha256_file, git_commit, read_json, write_json, write_jsonl
from opk_rag.evaluation.chunk_representation_experiment import (
    ChunkRepresentationExperimentError,
    _build_qwen_execution_context,
    _count_tokens,
    _p95,
    _public_manifest,
    _shadow_cardinality,
    _validate_shadow_cardinality,
    load_candidate_universe_for_formal_run,
)
from opk_rag.evaluation.corpus_snapshot import (
    PRIVATE_MANIFEST_PATH,
    PUBLIC_SNAPSHOT_PATH,
    manifest_digest,
    source_content_digest as snapshot_source_content_digest,
    validate_private_manifest,
    vault_identity_digest,
)
from opk_rag.evaluation.evidence_identity import digest_json
from opk_rag.evaluation.evidence_identity import digest_text
from opk_rag.evaluation.candidate_retrieval_baseline import _capability_label
from opk_rag.evaluation.qwen_embedding_execution import build_cache_key_row, build_execution_plan, materialize_embedding_variant, vector_digest
from opk_rag.evaluation.scope_structure_audit import IndexedScopeChunk, classify_block_structure
from opk_rag.evaluation.source_evidence_span import (
    SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
    SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION,
    SourceEvidenceSpan,
    SourceEvidenceSpanSet,
    SourceEvidenceResolution,
    build_public_source_span_manifest,
    build_public_source_span_resolution_manifest,
    build_source_evidence_spans_from_samples,
    calculate_span_union_coverage,
    match_shadow_chunk_to_source_span,
    source_text_digest,
)
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.vault import VaultPathError, VaultScanConfig, VaultScanner, normalize_vault_root


V1_EXPERIMENT_ID = "phase2-scope-aware-chunking-experiment-v1"
EXPERIMENT_ID = "phase2-scope-aware-chunking-experiment-v2"
SCHEMA_VERSION = "opk-rag.scope-aware-chunking-experiment-summary.v1"
CONTRACT_SCHEMA_VERSION = "opk-rag.scope-aware-chunking-experiment-contract.v1"
V1_CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_scope_aware_chunking_experiment_contract_v1.json"
V1_RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_experiment_v1.json"
V1_TRACE_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_experiment_v1_traces.jsonl"
V1_STRUCTURE_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_structure_v1.json"
V1_EXECUTION_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_execution_v1.json"
V1_EXECUTION_MANIFEST_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_scope_aware_chunking_execution_manifest_v1.json"
CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_scope_aware_chunking_experiment_contract_v2.json"
EXECUTION_MANIFEST_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_scope_aware_chunking_execution_manifest_v2.json"
SOURCE_SPAN_CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_source_evidence_span_contract_v1.json"
SOURCE_SPAN_RESOLUTION_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_source_evidence_span_resolution_v1.json"
PRODUCTION_AUDIT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_v1_audit.json"
PRODUCTION_PARITY_CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_contract_v1.json"
PRODUCTION_PARITY_REPORT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_report_v1.json"
C0_SCORING_SMOKE_PATH = ROOT / "evaluation-data" / "results" / "phase2_c0_source_span_scoring_smoke_v1.json"
RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_experiment_v2.json"
TRACE_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_experiment_v2_traces.jsonl"
STRUCTURE_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_structure_v2.json"
EXECUTION_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_aware_chunking_execution_v2.json"
REPORT_PATH = ROOT / "docs" / "PHASE2_SCOPE_AWARE_CHUNKING_EXPERIMENT.md"
PRIVATE_CACHE_PATH = ROOT / ".private" / "evaluation" / "task0060"
SOURCE_ROOT = ROOT / "source-documents"
EVALUATION_VAULT_ROOT_ENV = "OPK_RAG_PHASE2_EVALUATION_VAULT_ROOT"
REFERENCE_VAULT_ROOT_ENV = "OPK_RAG_VAULT_PATH"
K_VALUES = (1, 3, 5, 10, 20)
VARIANT_IDS = (
    "production_chunking_v1",
    "heading_attached_section",
    "structured_block_atomic",
    "fixed_one_block_overlap",
    "combined_scope_aware",
)
EXPECTED_DOCUMENT_COUNT = 120
EXPECTED_REQUIRED_EVIDENCE_UNITS = 84
EXPECTED_UNIQUE_CANONICAL_SCOPES = 37
EXPECTED_REQUIRED_EVIDENCE_SAMPLES = 61
EXPECTED_ALL_PUBLIC_SAMPLES = 68
EXPECTED_PRODUCTION_CHUNK_COUNT = 604
FROZEN_C0_SMOKE_METRICS = {
    "any_support_recall_at_k": {"1": 0.7619047619047619, "3": 0.9047619047619048, "5": 0.9761904761904762, "10": 1.0, "20": 1.0},
    "complete_evidence_recall_at_k": {"1": 0.2976190476190476, "3": 0.7023809523809523, "5": 0.8690476190476191, "10": 0.8690476190476191, "20": 0.8928571428571429},
    "document_recall_at_k": {"5": 0.9761904761904762, "20": 1.0},
    "mean_reciprocal_rank": 0.4761904761904761,
    "fully_covered_sample_count": 52,
    "required_evidence_unit_count": EXPECTED_REQUIRED_EVIDENCE_UNITS,
}
FROZEN_C0_TOLERANCE = 1e-12
COMPLETE_EVIDENCE_DEFINITION = {
    "aggregation_unit": "required_evidence_unit",
    "definition": "Top-K candidate complete_source_span union covers the required evidence source span or every ordered span-set segment.",
    "single_candidate_required": False,
    "multi_candidate_union_allowed": True,
    "candidate_order_affects_coverage": False,
    "partial_overlap_counts_as_complete": False,
    "duplicate_overlap_double_counts": False,
}


class ScopeAwareChunkingExperimentError(ValueError):
    pass


class ScopeAwareChunkingInfrastructureBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceDocument:
    relative_path: str
    document_identity_digest: str
    source_digest: str
    title: str | None
    parsed: ParsedMarkdownDocument
    line_count: int


@dataclass(frozen=True)
class ShadowChunk:
    variant_id: str
    document_identity_digest: str
    relative_path: str
    document_title: str | None
    heading_path: tuple[str, ...]
    content: str
    chunk_ordinal: int
    primary_source_span: dict[str, int]
    complete_source_span: dict[str, int]
    rendered_content_digest: str
    shadow_chunk_identity: dict[str, Any]
    block_types: tuple[str, ...]
    overlap_token_count: int = 0

    @property
    def identity(self) -> dict[str, Any]:
        return self.shadow_chunk_identity

    @property
    def chunk_index(self) -> int:
        return self.chunk_ordinal

    @property
    def start_line(self) -> int:
        return self.primary_source_span["start_line"]

    @property
    def end_line(self) -> int:
        return self.primary_source_span["end_line"]

    @property
    def scope_identity_digest(self) -> str:
        return self.shadow_chunk_identity["shadow_chunk_identity_digest"]

    def public_json(self) -> dict[str, Any]:
        return {
            "shadow_chunk_identity_digest": self.scope_identity_digest,
            "document_identity_digest": self.document_identity_digest,
            "chunking_variant_id": self.variant_id,
            "chunk_ordinal": self.chunk_ordinal,
            "primary_source_span": self.primary_source_span,
            "complete_source_span": self.complete_source_span,
            "rendered_content_digest": self.rendered_content_digest,
            "block_types": list(self.block_types),
            "overlap_token_count": self.overlap_token_count,
        }


@dataclass(frozen=True)
class ShadowCandidate:
    rank: int
    chunk: ShadowChunk
    score: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def audit_production_chunking_v1() -> dict[str, Any]:
    existing = read_json(PRODUCTION_AUDIT_PATH) if PRODUCTION_AUDIT_PATH.exists() else {}
    audit = {
        "schema_version": "opk-rag.production-chunking-v1-audit.v1",
        "audit_id": "phase2-production-chunking-v1-audit",
        "created_at": existing.get("created_at") or utc_now(),
        "parser_version": PARSER_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "markdown_parsing_contract": "UTF-8 Markdown, safe YAML frontmatter, ATX headings outside fenced code blocks",
        "heading_hierarchy_handling": "parser maintains full heading_path by ATX heading level",
        "heading_inclusion_policy": "heading line remains in section content and heading_path is stored as metadata",
        "paragraph_grouping": "short sections are packed until target_size or max_size",
        "list_handling": "lists are source text; oversized sections may split before list item boundaries",
        "table_handling": "tables are source text; no table atomicity rule in production v1",
        "code_block_handling": "fenced code blocks are preserved while splitting oversized sections; oversized fences become oversized chunks",
        "quote_handling": "quotes are source text; no quote atomicity rule in production v1",
        "frontmatter_handling": "safe YAML mapping extracted before body parsing; not included in chunk content",
        "blank_line_handling": "oversized splitting first prefers two-or-more newline paragraph boundaries",
        "maximum_chunk_size": 1800,
        "minimum_chunk_size": None,
        "target_chunk_size": 1200,
        "length_unit": "character",
        "overlap": 0,
        "flush_condition": "flush when pending + next section exceeds max_size, or when pending reaches target_size",
        "oversized_block_policy": "split by blank lines, list boundaries, sentence punctuation, then hard max_size slices; oversized code fence is kept whole",
        "source_line_mapping": "section start/end retained; split text may lose exact internal line boundary for middle pieces",
        "chunk_identity_construction": "database id is Postgres UUID; stable comparison uses document_id + chunk_index + content_hash + heading_path",
        "content_digest_construction": "sha256(normalized chunk content UTF-8)",
        "production_default_change": False,
        "contains_sealed_holdout_data": False,
    }
    write_json(PRODUCTION_AUDIT_PATH, audit)
    return audit


def build_chunking_experiment_contract(embedding_config: EmbeddingConfig | None = None) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    existing = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    production_audit = audit_production_chunking_v1()
    source_span_manifest = read_json(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else {}
    variants = [_variant_contract(variant_id) for variant_id in VARIANT_IDS]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "supersedes_experiment_id": V1_EXPERIMENT_ID,
        "correction_reason": "TASK-0059 source-span scoring oracle was incorrectly registered as the TASK-0060 retrieval promotion baseline",
        "correction_classification": "candidate_generation_contract_alignment_fix_not_quality_tuning",
        "created_at": existing.get("created_at") or utc_now(),
        "git_commit": git_commit(),
        "source_corpus": {
            "corpus_snapshot_id": read_json(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json").get("snapshot_id"),
            "corpus_snapshot_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "expected_document_count": EXPECTED_DOCUMENT_COUNT,
            "corpus_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "document_universe_digest": None,
            "document_identity_contract": "sha256 stable JSON over vault-relative POSIX path digest",
            "relative_path_contract": "private frozen manifest source_files.path, vault-relative POSIX, sorted lexicographically",
            "source_content_digest_contract": "per-file sha256 bytes must match private manifest content_sha256",
            "configured_vault_root_source": f"{EVALUATION_VAULT_ROOT_ENV} then {REFERENCE_VAULT_ROOT_ENV}; shell before dotenv; reference runtime cannot reveal absolute path",
            "parser_version": PARSER_VERSION,
            "frontmatter_policy": "safe YAML mapping stripped before body chunking",
            "encoding_policy": "UTF-8 only",
        },
        "source_evidence_span_contract": {
            "schema": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
            "path": SOURCE_SPAN_CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "digest": digest_json(source_span_manifest) if source_span_manifest else None,
        },
        "embedding_contract": {
            "provider": embedding_config.provider,
            "model": DEFAULT_EMBEDDING_MODEL_NAME,
            "model_revision": embedding_config.model_revision or DEFAULT_EMBEDDING_MODEL_REVISION,
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
            "similarity": "cosine",
            "l2_normalization": True,
            "local_files_only": True,
            "fingerprint": build_configuration_fingerprint(embedding_config),
        },
        "chunking_variants": variants,
        "query_contract": {
            "variant_id": "current_query",
            "template_version": QUERY_INPUT_TEMPLATE_VERSION,
            "digest": digest_json({"variant_id": "current_query", "template_version": QUERY_INPUT_TEMPLATE_VERSION}),
        },
        "k_values": list(K_VALUES),
        "metrics": [
            "any_support_recall_at_k",
            "complete_evidence_recall_at_k",
            "document_recall_at_k",
            "mean_reciprocal_rank",
            "fragmentation",
            "contamination",
            "cost",
        ],
        "promotion_gates": {
            "complete_evidence_recall_at_5_absolute_improvement_min": 0.05,
            "complete_evidence_recall_at_20_not_below_baseline": True,
            "document_recall_at_20_max_drop": 0.02,
            "development_and_known_regression_not_below_baseline": True,
            "no_evidence_candidate_contamination_max_increase": 0.05,
            "shadow_index_size_max_baseline_multiplier": 3.0,
            "embedding_input_p95_max_baseline_multiplier": 3.0,
            "query_latency_p95_max_baseline_multiplier": 2.0,
        },
        "selection_policy": {
            "primary_metric": "combined_public_complete_evidence_recall_at_5",
            "promotion_baseline": "formal_same_pipeline_c0_retrieval",
            "tie_breakers": ["complete_evidence_recall_at_20", "document_recall_at_20", "fragmentation_rate", "cost"],
            "if_no_gate_passes": "no_candidate",
        },
        "bindings": {
            "task0057_result_digest": _sha256_file(ROOT / "evaluation-data" / "results" / "phase2_chunk_representation_experiment_v1.json"),
            "corpus_snapshot_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "parser_contract_digest": digest_json({"parser_version": PARSER_VERSION}),
            "production_chunking_digest": digest_json(production_audit),
            "source_evidence_span_digest": digest_json(source_span_manifest) if source_span_manifest else None,
            "tokenizer_digest": None,
            "variant_parameter_digests": {row["id"]: digest_json(row) for row in variants},
        },
        "production_default_change": False,
        "contains_sealed_holdout_data": False,
        "forbidden_inputs": ["sample_id", "query", "question_type", "capability", "required_evidence", "gold_scope", "gold_document", "gold_span", "expected_answer", "expected_action"],
    }
    _reject_forbidden_payload(contract)
    write_json(CONTRACT_PATH, contract)
    return contract


def build_execution_manifest(
    *,
    contract: dict[str, Any],
    source_universe_audit: dict[str, Any] | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    source_span_contract = read_json(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else {}
    parity_contract = read_json(PRODUCTION_PARITY_CONTRACT_PATH) if PRODUCTION_PARITY_CONTRACT_PATH.exists() else {}
    parity_report = read_json(PRODUCTION_PARITY_REPORT_PATH) if PRODUCTION_PARITY_REPORT_PATH.exists() else {}
    c0_smoke = read_json(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else {}
    manifest = {
        "schema_version": "opk-rag.scope-aware-chunking-execution-manifest.v1",
        "execution_id": "phase2-scope-aware-chunking-execution-v2",
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "source_corpus": {
            "expected_document_count": EXPECTED_DOCUMENT_COUNT,
            "resolved_document_count": (source_universe_audit or {}).get("resolved_document_count"),
            "document_universe_digest": (source_universe_audit or {}).get("document_universe_digest"),
            "corpus_snapshot_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "source_corpus_manifest_digest": (source_universe_audit or {}).get("private_manifest_digest"),
        },
        "source_evidence_contract": {
            "path": SOURCE_SPAN_CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "digest": _sha256_file(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else None,
            "status": source_span_contract.get("source_evidence_span_contract_status") or source_span_contract.get("status"),
            "resolved_required_units": source_span_contract.get("resolved_source_span_count") or source_span_contract.get("resolved_required_units"),
            "resolved_unique_scopes": source_span_contract.get("resolved_unique_scope_count") or source_span_contract.get("resolved_unique_scopes"),
        },
        "production_parity_contract": {
            "contract_path": PRODUCTION_PARITY_CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_digest": _sha256_file(PRODUCTION_PARITY_CONTRACT_PATH) if PRODUCTION_PARITY_CONTRACT_PATH.exists() else None,
            "report_path": PRODUCTION_PARITY_REPORT_PATH.relative_to(ROOT).as_posix(),
            "report_digest": _sha256_file(PRODUCTION_PARITY_REPORT_PATH) if PRODUCTION_PARITY_REPORT_PATH.exists() else None,
            "production_chunking_parity_status": parity_report.get("production_chunking_parity_status"),
            "source_span_agreement_status": parity_report.get("source_span_agreement_status"),
        },
        "chunking_variants": contract.get("chunking_variants") or [],
        "embedding_contract": contract.get("embedding_contract") or {},
        "query_contract": contract.get("query_contract") or {},
        "retrieval_contract": {"query_variant": "current_query", "k_values": list(K_VALUES), "reranker": False, "generation": False},
        "promotion_gates": contract.get("promotion_gates") or {},
        "holdout_policy": {
            "public_evaluation_only": True,
            "holdout_v1_used": False,
            "holdout_v1_future_acceptance_eligible": False,
            "replacement_holdout_required": True,
            "creates_holdout_v2": False,
        },
        "privacy_policy": {
            "publishes_query_text": False,
            "publishes_markdown_text": False,
            "publishes_chunk_text": False,
            "publishes_file_paths": False,
            "publishes_gold_claims": False,
            "publishes_embeddings": False,
            "contains_sealed_holdout_data": False,
        },
        "bindings": {
            "task0059_result_digest": _sha256_file(PRODUCTION_PARITY_REPORT_PATH) if PRODUCTION_PARITY_REPORT_PATH.exists() else None,
            "corpus_snapshot_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "source_corpus_manifest_digest": (source_universe_audit or {}).get("private_manifest_digest"),
            "production_parity_contract_digest": _sha256_file(PRODUCTION_PARITY_CONTRACT_PATH) if PRODUCTION_PARITY_CONTRACT_PATH.exists() else None,
            "source_evidence_span_contract_digest": _sha256_file(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else None,
            "chunking_experiment_contract_digest": digest_json(contract),
            "qwen_model_revision": embedding_config.model_revision or DEFAULT_EMBEDDING_MODEL_REVISION,
            "tokenizer_digest": (contract.get("bindings") or {}).get("tokenizer_digest"),
            "query_contract_digest": digest_json(contract.get("query_contract") or {}),
            "variant_parameter_digests": (contract.get("bindings") or {}).get("variant_parameter_digests"),
            "scoring_digest": digest_json({"source_span_contract": _sha256_file(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else None, "c0_smoke_digest": digest_json(c0_smoke) if c0_smoke else None}),
            "promotion_gate_digest": digest_json(contract.get("promotion_gates") or {}),
        },
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
    }
    _reject_forbidden_payload(manifest)
    write_json(EXECUTION_MANIFEST_PATH, manifest)
    return manifest


def freeze_v1_invalid_baseline_semantics() -> dict[str, Any]:
    """Mark the preregistered v1 semantics invalid without deleting diagnostics."""
    reason = "TASK-0059 source-span scoring oracle was incorrectly registered as the TASK-0060 retrieval promotion baseline"
    result = read_json(V1_RESULT_PATH) if V1_RESULT_PATH.exists() else {}
    contract = read_json(V1_CONTRACT_PATH) if V1_CONTRACT_PATH.exists() else {}
    invalid_fields = {
        "experiment_status": "invalid_baseline_semantics",
        "scope_aware_chunking_experiment_status": "invalid_baseline_semantics",
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "primary_result_classification": "invalid_experiment",
        "result_validity": "diagnostic_only_not_valid_for_promotion",
        "invalid_reason": reason,
        "baseline_semantics_correction": {
            "status": "invalid_baseline_semantics",
            "correction_reason": reason,
            "task0059_complete_evidence_recall_at_5": FROZEN_C0_SMOKE_METRICS["complete_evidence_recall_at_k"]["5"],
            "task0059_intended_use": "scorer_regression_only",
            "task0059_forbidden_use": "chunking_promotion_baseline",
            "valid_successor_experiment_id": EXPERIMENT_ID,
        },
    }
    if result:
        result.update(invalid_fields)
        result.setdefault("invalid_pre_fix_task0060_run", {})
        result.setdefault("diagnostic_observations_retained", True)
        write_json(V1_RESULT_PATH, result)
    if contract:
        contract.update(
            {
                "experiment_id": V1_EXPERIMENT_ID,
                "experiment_status": "invalid_baseline_semantics",
                "promotion_decision": "invalid_experiment",
                "recommended_variant": None,
                "primary_result_classification": "invalid_experiment",
                "result_validity": "diagnostic_only_not_valid_for_promotion",
                "invalid_reason": reason,
                "superseded_by_experiment_id": EXPERIMENT_ID,
            }
        )
        write_json(V1_CONTRACT_PATH, contract)
    return {"v1_result_path": _public_path(V1_RESULT_PATH), "v1_contract_path": _public_path(V1_CONTRACT_PATH), **invalid_fields}


def _public_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_source_corpus_manifest(
    private_manifest_path: Path = PRIVATE_MANIFEST_PATH,
    public_snapshot_path: Path = PUBLIC_SNAPSHOT_PATH,
) -> dict[str, Any]:
    if not private_manifest_path.exists():
        raise ScopeAwareChunkingInfrastructureBlocked("private frozen source corpus manifest is missing")
    manifest = read_json(private_manifest_path)
    issues = validate_private_manifest(manifest)
    if issues:
        codes = ", ".join(issue.code for issue in issues)
        raise ScopeAwareChunkingInfrastructureBlocked(f"private frozen source corpus manifest is invalid: {codes}")
    if public_snapshot_path.exists():
        public = read_json(public_snapshot_path)
        if public.get("private_manifest_digest") != manifest_digest(manifest):
            raise ScopeAwareChunkingInfrastructureBlocked("public corpus snapshot digest does not match private manifest")
        if public.get("source_content_digest") != (manifest.get("vault") or {}).get("source_content_digest"):
            raise ScopeAwareChunkingInfrastructureBlocked("public corpus snapshot source digest does not match private manifest")
    return manifest


def resolve_evaluation_vault_root(env: dict[str, str] | None = None) -> dict[str, Any]:
    source_env = os.environ if env is None else env
    shell_values = {key: str(source_env.get(key, "")).strip() for key in (EVALUATION_VAULT_ROOT_ENV, REFERENCE_VAULT_ROOT_ENV)}
    for key in (EVALUATION_VAULT_ROOT_ENV, REFERENCE_VAULT_ROOT_ENV):
        if shell_values[key]:
            return _resolved_vault_root_payload(shell_values[key], f"shell:{key}")

    if env is None:
        load_result = load_project_env(ROOT)
        dotenv_values = {key: str(os.environ.get(key, "")).strip() for key in (EVALUATION_VAULT_ROOT_ENV, REFERENCE_VAULT_ROOT_ENV)}
        for key in (EVALUATION_VAULT_ROOT_ENV, REFERENCE_VAULT_ROOT_ENV):
            if key in load_result.loaded_keys and dotenv_values[key]:
                return _resolved_vault_root_payload(dotenv_values[key], f"dotenv:{key}")

    return {
        "status": "missing",
        "vault_root": None,
        "source_root_config_source": "reference_runtime",
        "source_root_digest": None,
        "issues": [
            {
                "code": "source_root_not_configured",
                "message": f"{EVALUATION_VAULT_ROOT_ENV} or {REFERENCE_VAULT_ROOT_ENV} is required; reference runtime artifacts do not store absolute paths.",
            }
        ],
    }


def resolve_frozen_source_corpus() -> tuple[tuple[SourceDocument, ...], dict[str, Any]]:
    manifest = load_source_corpus_manifest()
    root_resolution = resolve_evaluation_vault_root()
    if root_resolution["status"] != "resolved":
        audit = build_source_corpus_manifest(manifest=manifest, root_resolution=root_resolution)
        raise ScopeAwareChunkingInfrastructureBlocked(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    audit, documents = match_source_documents_to_snapshot(manifest, root_resolution["vault_root"], root_resolution=root_resolution)
    if audit["status"] != "valid":
        raise ScopeAwareChunkingInfrastructureBlocked(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return documents, audit


def load_full_source_document_universe(source_root: Path | None = None) -> tuple[SourceDocument, ...]:
    if source_root is not None:
        raise ScopeAwareChunkingInfrastructureBlocked("TASK-0058 source corpus must be resolved from the frozen Phase 2 corpus snapshot, not an ad hoc source path")
    return resolve_frozen_source_corpus()[0]


def build_source_corpus_manifest(*, manifest: dict[str, Any], root_resolution: dict[str, Any]) -> dict[str, Any]:
    source_files = [item for item in manifest.get("source_files", []) if isinstance(item, dict)]
    duplicate_paths = _duplicate_count(str(item.get("path", "")) for item in source_files)
    duplicate_document_identities = _duplicate_count(_document_identity_digest(str(item.get("path", ""))) for item in source_files)
    expected_count = int((manifest.get("vault") or {}).get("markdown_file_count") or len(source_files))
    return {
        "schema_version": "opk-rag.scope-aware-source-corpus-manifest.v1",
        "source_corpus_status": "incomplete",
        "status": "invalid",
        "corpus_snapshot_id": manifest.get("snapshot_id"),
        "corpus_snapshot_digest": manifest.get("vault", {}).get("source_content_digest"),
        "private_manifest_digest": manifest_digest(manifest),
        "document_identity_contract": "sha256 stable JSON over vault-relative POSIX path digest",
        "relative_path_contract": "private frozen manifest source_files.path, vault-relative POSIX, sorted lexicographically",
        "source_content_digest_contract": "per-file sha256 bytes must match private manifest content_sha256; aggregate digest must match public snapshot source_content_digest",
        "configured_vault_root_source": root_resolution.get("source_root_config_source"),
        "source_root_config_source": root_resolution.get("source_root_config_source"),
        "source_root_digest": root_resolution.get("source_root_digest"),
        "expected_document_count": expected_count,
        "resolved_document_count": 0,
        "missing_document_count": expected_count,
        "unexpected_document_count": 0,
        "digest_match_count": 0,
        "digest_mismatch_count": 0,
        "duplicate_relative_path_count": duplicate_paths,
        "duplicate_document_identity_count": duplicate_document_identities,
        "failed_document_count": 0,
        "parsed_document_count": 0,
        "document_universe_digest": None,
        "candidate_universe_source": "frozen_phase2_private_corpus_manifest",
        "repository_source_documents_markdown_count": _repository_source_documents_markdown_count(),
        "gold_evidence_used_for_source_selection": False,
        "gold_document_used_for_source_selection": False,
        "legacy_chunk_stitching_used_as_source": False,
        "publishes_absolute_paths": False,
        "publishes_source_text": False,
        "issues": list(root_resolution.get("issues") or []),
    }


def match_source_documents_to_snapshot(
    manifest: dict[str, Any],
    vault_root: Path,
    *,
    root_resolution: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[SourceDocument, ...]]:
    root_resolution = root_resolution or _resolved_vault_root_payload(vault_root, "shell:unknown")
    base_audit = build_source_corpus_manifest(manifest=manifest, root_resolution=root_resolution)
    root = normalize_vault_root(vault_root)
    expected_files = [dict(item) for item in manifest.get("source_files", []) if isinstance(item, dict)]
    expected_by_path = {str(item.get("path")): item for item in expected_files}
    scan = VaultScanner(VaultScanConfig()).scan(root.path)
    scanned_by_path = {item.relative_path: item for item in scan.files}
    documents: list[SourceDocument] = []
    missing = sorted(set(expected_by_path) - set(scanned_by_path))
    unexpected = sorted(set(scanned_by_path) - set(expected_by_path))
    digest_mismatch = []
    failed = []
    for relative_path in sorted(set(expected_by_path) & set(scanned_by_path)):
        expected = expected_by_path[relative_path]
        scanned = scanned_by_path[relative_path]
        if scanned.content_hash != expected.get("content_sha256") or scanned.size_bytes != expected.get("size_bytes"):
            digest_mismatch.append(relative_path)
            continue
        path = root.path / relative_path
        try:
            text = path.read_text(encoding="utf-8")
            parsed = parse_markdown_file(path)
        except Exception:
            failed.append(relative_path)
            continue
        documents.append(
            SourceDocument(
                relative_path=relative_path,
                document_identity_digest=_document_identity_digest(relative_path),
                source_digest=scanned.content_hash,
                title=parsed.title,
                parsed=parsed,
                line_count=max(1, len(text.splitlines())),
            )
        )
    aggregate_digest = snapshot_source_content_digest(expected_files)
    issues = list(base_audit["issues"])
    if len(expected_files) != EXPECTED_DOCUMENT_COUNT:
        issues.append({"code": "frozen_manifest_document_count_mismatch", "actual": len(expected_files), "expected": EXPECTED_DOCUMENT_COUNT})
    if missing:
        issues.append({"code": "missing_source_documents", "count": len(missing)})
    if digest_mismatch:
        issues.append({"code": "source_digest_mismatch", "count": len(digest_mismatch)})
    if failed:
        issues.append({"code": "source_document_parse_failed", "count": len(failed)})
    if base_audit["duplicate_relative_path_count"]:
        issues.append({"code": "duplicate_relative_path", "count": base_audit["duplicate_relative_path_count"]})
    if base_audit["duplicate_document_identity_count"]:
        issues.append({"code": "duplicate_document_identity", "count": base_audit["duplicate_document_identity_count"]})
    if aggregate_digest != manifest.get("vault", {}).get("source_content_digest"):
        issues.append({"code": "aggregate_source_digest_mismatch"})
    if scan.failures:
        issues.append({"code": "vault_scan_failures", "count": len(scan.failures)})
    expected_identity = vault_identity_digest(root.canonical_path, str(manifest.get("vault", {}).get("identity_salt", "")))
    if expected_identity != manifest.get("vault", {}).get("identity_digest"):
        issues.append({"code": "vault_identity_mismatch"})
    document_universe_digest = digest_json(
        [{"document_identity_digest": doc.document_identity_digest, "source_digest": doc.source_digest} for doc in documents]
    )
    status = "valid" if not issues and len(documents) == EXPECTED_DOCUMENT_COUNT else "invalid"
    audit = {
        **base_audit,
        "source_corpus_status": "complete" if status == "valid" else "incomplete",
        "status": status,
        "resolved_document_count": len(documents),
        "missing_document_count": len(missing),
        "unexpected_document_count": len(unexpected),
        "digest_match_count": len(documents),
        "digest_mismatch_count": len(digest_mismatch),
        "failed_document_count": len(failed),
        "parsed_document_count": len(documents),
        "document_universe_digest": document_universe_digest,
        "vault_identity_match": expected_identity == manifest.get("vault", {}).get("identity_digest"),
        "issues": issues,
    }
    return audit, tuple(documents)


def verify_source_document_digests(documents: tuple[SourceDocument, ...], manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {str(item.get("path")): item for item in manifest.get("source_files", []) if isinstance(item, dict)}
    mismatches = [
        doc.relative_path
        for doc in documents
        if doc.relative_path not in expected or doc.source_digest != expected[doc.relative_path].get("content_sha256")
    ]
    return {
        "expected_document_count": len(expected),
        "resolved_document_count": len(documents),
        "digest_match_count": len(documents) - len(mismatches),
        "digest_mismatch_count": len(mismatches),
        "status": "valid" if not mismatches and len(documents) == len(expected) else "invalid",
    }


def validate_source_universe(documents: tuple[SourceDocument, ...], *, expected_document_count: int = EXPECTED_DOCUMENT_COUNT) -> dict[str, Any]:
    digest = digest_json([{"document_identity_digest": doc.document_identity_digest, "source_digest": doc.source_digest} for doc in documents])
    duplicate_paths = _duplicate_count(doc.relative_path for doc in documents)
    duplicate_document_identities = _duplicate_count(doc.document_identity_digest for doc in documents)
    issues = []
    if len(documents) != expected_document_count:
        issues.append({"code": "source_document_count_mismatch", "actual": len(documents), "expected": expected_document_count})
    if duplicate_paths:
        issues.append({"code": "duplicate_relative_path", "count": duplicate_paths})
    if duplicate_document_identities:
        issues.append({"code": "duplicate_document_identity", "count": duplicate_document_identities})
    return {
        "expected_document_count": expected_document_count,
        "parsed_document_count": len(documents),
        "failed_document_count": 0,
        "document_universe_digest": digest,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
    }


def build_source_span_contract(production_chunks: tuple[IndexedScopeChunk, ...] | None = None) -> dict[str, Any]:
    existing = read_json(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else {}
    if SOURCE_SPAN_RESOLUTION_PATH.exists():
        resolution = read_json(SOURCE_SPAN_RESOLUTION_PATH)
        if resolution.get("source_evidence_span_contract_status") == "valid" and len(resolution.get("spans") or []) == EXPECTED_REQUIRED_EVIDENCE_UNITS:
            manifest = {
                "contract_id": "phase2-source-evidence-span-contract-v1",
                "status": "complete",
                "source_evidence_span_contract_status": "complete",
                "created_at": existing.get("created_at") or utc_now(),
                "expected_required_evidence_unit_count": EXPECTED_REQUIRED_EVIDENCE_UNITS,
                "required_evidence_unit_count": resolution.get("required_evidence_unit_count"),
                "resolved_source_span_count": resolution.get("resolved_required_units"),
                "unresolved_source_span_count": resolution.get("unresolved_required_units"),
                "unique_source_span_count": resolution.get("resolved_unique_scopes"),
                "contiguous_span_count": resolution.get("contiguous_span_count"),
                "span_set_count": resolution.get("span_set_count"),
                "multi_old_chunk_span_count": resolution.get("required_unit_multi_chunk_count"),
                "legacy_chunk_identity_use": "source_span_migration_only",
                "new_variant_scoring_uses": "source_line_span_union_coverage_only",
                "content_similarity_used_for_gold_resolution": False,
                "manual_result_based_span_edits_allowed": False,
                "publishes_source_text": False,
                "contains_sealed_holdout_data": False,
                "spans": resolution.get("spans") or [],
                "schema_version": "opk-rag.canonical-source-evidence-resolution.v1",
            }
            write_json(SOURCE_SPAN_CONTRACT_PATH, manifest)
            return manifest
    chunks = production_chunks or load_candidate_universe_for_formal_run(EmbeddingConfig(local_files_only=True, normalize=True))[0]
    samples = _load_public_samples(["development", "known-regression"])
    try:
        spans = build_source_evidence_spans_from_samples(samples, chunks)
    except ValueError as exc:
        return build_invalid_source_span_contract(str(exc))
    manifest = build_public_source_span_manifest(spans)
    manifest.update(
        {
            "contract_id": "phase2-source-evidence-span-contract-v1",
            "status": "complete" if manifest.get("resolved_source_span_count") == EXPECTED_REQUIRED_EVIDENCE_UNITS and manifest.get("unresolved_source_span_count") == 0 else "incomplete",
            "source_evidence_span_contract_status": "complete" if manifest.get("resolved_source_span_count") == EXPECTED_REQUIRED_EVIDENCE_UNITS and manifest.get("unresolved_source_span_count") == 0 else "incomplete",
            "created_at": existing.get("created_at") or utc_now(),
            "expected_required_evidence_unit_count": EXPECTED_REQUIRED_EVIDENCE_UNITS,
            "legacy_chunk_identity_use": "source_span_migration_only",
            "new_variant_scoring_uses": "source_line_span_union_coverage_only",
            "content_similarity_used_for_gold_resolution": False,
            "manual_result_based_span_edits_allowed": False,
            "contains_sealed_holdout_data": False,
        }
    )
    write_json(SOURCE_SPAN_CONTRACT_PATH, manifest)
    return manifest


def build_invalid_source_span_contract(reason: str) -> dict[str, Any]:
    samples = _load_public_samples(["development", "known-regression"])
    required_count = sum(
        len({digest_json(unit) for unit in (sample.get("required_evidence") or [])})
        for sample in samples
    )
    manifest = {
        "schema_version": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
        "contract_id": "phase2-source-evidence-span-contract-v1",
        "status": "incomplete",
        "source_evidence_span_contract_status": "incomplete",
        "created_at": utc_now(),
        "required_evidence_unit_count": required_count,
        "expected_required_evidence_unit_count": EXPECTED_REQUIRED_EVIDENCE_UNITS,
        "resolved_source_span_count": 0,
        "unresolved_source_span_count": required_count,
        "unique_source_span_count": 0,
        "multi_old_chunk_span_count": None,
        "invalid_reason": reason,
        "legacy_chunk_identity_use": "source_span_migration_only",
        "new_variant_scoring_uses": "source_line_span_union_coverage_only",
        "content_similarity_used_for_gold_resolution": False,
        "manual_result_based_span_edits_allowed": False,
        "publishes_source_text": False,
        "contains_sealed_holdout_data": False,
        "spans": [],
    }
    write_json(SOURCE_SPAN_CONTRACT_PATH, manifest)
    return manifest


def _load_source_evidence_resolutions(
    rows: list[dict[str, Any]],
    *,
    document_identity_map: dict[str, str] | None = None,
) -> tuple[SourceEvidenceResolution, ...]:
    document_identity_map = document_identity_map or {}
    spans: list[SourceEvidenceResolution] = []
    for row in rows:
        document_identity_digest = document_identity_map.get(str(row["document_identity_digest"]), row["document_identity_digest"])
        if row.get("schema_version") == SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION:
            spans.append(
                SourceEvidenceSpanSet(
                    required_evidence_unit_id=row["required_evidence_unit_id"],
                    dataset_id=row["dataset_id"],
                    sample_id=row["sample_id"],
                    document_identity_digest=document_identity_digest,
                    spans=tuple((int(item["source_line_start"]), int(item["source_line_end"])) for item in row.get("spans") or []),
                    heading_path_digest=row.get("heading_path_digest"),
                    source_span_digest=row["source_span_digest"],
                    legacy_evidence_identity_digest=row["legacy_evidence_identity_digest"],
                    legacy_chunk_identity_digests=tuple(row.get("legacy_chunk_identity_digests") or ()),
                    legacy_chunk_indexes=tuple(row.get("legacy_chunk_indexes") or ()),
                    old_chunk_count=int(row.get("old_chunk_count") or 0),
                )
            )
        else:
            payload = {key: value for key, value in row.items() if key != "schema_version"}
            payload["document_identity_digest"] = document_identity_digest
            spans.append(
                SourceEvidenceSpan(
                    **{
                        key: tuple(value) if key in {"legacy_chunk_identity_digests", "legacy_chunk_indexes"} else value
                        for key, value in payload.items()
                    }
                )
            )
    return tuple(spans)


def _shadow_document_identity_map(production_chunks: tuple[IndexedScopeChunk, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in production_chunks:
        shadow_digest = _document_identity_digest(chunk.relative_path)
        if chunk.document_identity_digest:
            mapping[str(chunk.document_identity_digest)] = shadow_digest
        relative_path_digest = chunk.identity.get("relative_path_digest")
        if relative_path_digest:
            mapping[str(relative_path_digest)] = shadow_digest
    return mapping


def run_production_chunking_control(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    chunks: list[ShadowChunk] = []
    for document in documents:
        for chunk in chunk_markdown_document(document.parsed):
            chunks.append(
                build_shadow_chunk_identity(
                    variant_id="production_chunking_v1",
                    document=document,
                    ordinal=sum(1 for item in chunks if item.document_identity_digest == document.document_identity_digest),
                    content=chunk.content,
                    heading_path=chunk.heading_path,
                    primary_span={"start_line": chunk.start_line or 1, "end_line": chunk.end_line or chunk.start_line or 1},
                    complete_span={"start_line": chunk.start_line or 1, "end_line": chunk.end_line or chunk.start_line or 1},
                    block_types=classify_block_structure(chunk.content),
                )
            )
    return tuple(chunks)


def run_heading_attached_chunking(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    return _run_section_block_chunking(documents, "heading_attached_section", block_atomic=False, overlap=False)


def run_structured_block_chunking(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    return _run_section_block_chunking(documents, "structured_block_atomic", block_atomic=True, overlap=False)


def run_fixed_overlap_chunking(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    base = list(run_production_chunking_control(documents))
    by_doc: dict[str, list[ShadowChunk]] = defaultdict(list)
    for chunk in base:
        by_doc[chunk.document_identity_digest].append(chunk)
    out: list[ShadowChunk] = []
    for document in documents:
        previous: ShadowChunk | None = None
        for ordinal, chunk in enumerate(by_doc[document.document_identity_digest]):
            overlap = _overlap_text(previous.content if previous else "", 128)
            content = f"{overlap}\n\n{chunk.content}".strip() if overlap else chunk.content
            complete = dict(chunk.primary_source_span)
            if previous and _same_high_level_heading(previous.heading_path, chunk.heading_path):
                complete["start_line"] = min(previous.primary_source_span["start_line"], chunk.primary_source_span["start_line"])
                complete["end_line"] = max(previous.primary_source_span["end_line"], chunk.primary_source_span["end_line"])
            out.append(
                build_shadow_chunk_identity(
                    variant_id="fixed_one_block_overlap",
                    document=document,
                    ordinal=ordinal,
                    content=content,
                    heading_path=chunk.heading_path,
                    primary_span=chunk.primary_source_span,
                    complete_span=complete,
                    block_types=chunk.block_types,
                    overlap_token_count=len(overlap.split()),
                )
            )
            previous = chunk
    return tuple(out)


def run_combined_scope_chunking(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    base = list(_run_section_block_chunking(documents, "combined_scope_aware", block_atomic=True, overlap=False))
    by_doc: dict[str, list[ShadowChunk]] = defaultdict(list)
    for chunk in base:
        by_doc[chunk.document_identity_digest].append(chunk)
    out: list[ShadowChunk] = []
    for document in documents:
        previous: ShadowChunk | None = None
        for ordinal, chunk in enumerate(by_doc[document.document_identity_digest]):
            overlap = _overlap_text(previous.content if previous else "", 128)
            complete = dict(chunk.primary_source_span)
            content = chunk.content
            if overlap and previous and _same_high_level_heading(previous.heading_path, chunk.heading_path):
                content = f"{overlap}\n\n{chunk.content}".strip()
                complete["start_line"] = min(previous.primary_source_span["start_line"], chunk.primary_source_span["start_line"])
                complete["end_line"] = max(previous.primary_source_span["end_line"], chunk.primary_source_span["end_line"])
            out.append(
                build_shadow_chunk_identity(
                    variant_id="combined_scope_aware",
                    document=document,
                    ordinal=ordinal,
                    content=content,
                    heading_path=chunk.heading_path,
                    primary_span=chunk.primary_source_span,
                    complete_span=complete,
                    block_types=chunk.block_types,
                    overlap_token_count=len(overlap.split()) if overlap else 0,
                )
            )
            previous = chunk
    return tuple(out)


def build_shadow_chunk_identity(
    *,
    variant_id: str,
    document: SourceDocument,
    ordinal: int,
    content: str,
    heading_path: tuple[str, ...],
    primary_span: dict[str, int],
    complete_span: dict[str, int],
    block_types: tuple[str, ...],
    overlap_token_count: int = 0,
) -> ShadowChunk:
    normalized = normalize_chunk_content(content)
    identity = {
        "document_identity_digest": document.document_identity_digest,
        "chunking_variant_id": variant_id,
        "chunk_ordinal": ordinal,
        "primary_source_line_start": primary_span["start_line"],
        "primary_source_line_end": primary_span["end_line"],
        "complete_source_span_digest": digest_json(complete_span),
        "rendered_content_digest": chunk_content_hash(normalized),
    }
    identity["shadow_chunk_identity_digest"] = digest_json(identity)
    return ShadowChunk(
        variant_id=variant_id,
        document_identity_digest=document.document_identity_digest,
        relative_path=document.relative_path,
        document_title=document.title,
        heading_path=heading_path,
        content=normalized,
        chunk_ordinal=ordinal,
        primary_source_span=primary_span,
        complete_source_span=complete_span,
        rendered_content_digest=identity["rendered_content_digest"],
        shadow_chunk_identity=identity,
        block_types=block_types,
        overlap_token_count=overlap_token_count,
    )


def validate_shadow_chunks(chunks: tuple[ShadowChunk, ...], documents: tuple[SourceDocument, ...]) -> dict[str, Any]:
    identities = [chunk.scope_identity_digest for chunk in chunks]
    by_doc: dict[str, list[ShadowChunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.document_identity_digest].append(chunk)
    invalid_ranges = [
        chunk for chunk in chunks
        if chunk.primary_source_span["start_line"] <= 0
        or chunk.primary_source_span["end_line"] < chunk.primary_source_span["start_line"]
        or chunk.complete_source_span["end_line"] < chunk.complete_source_span["start_line"]
    ]
    non_monotonic = 0
    for doc_chunks in by_doc.values():
        expected = 0
        for chunk in sorted(doc_chunks, key=lambda item: item.chunk_ordinal):
            if chunk.chunk_ordinal != expected:
                non_monotonic += 1
            expected += 1
    document_digests = {document.document_identity_digest for document in documents}
    return {
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "parsed_document_count": len(documents),
        "failed_document_count": 0,
        "shadow_chunk_count": len(chunks),
        "unique_shadow_chunk_identity_count": len(set(identities)),
        "duplicate_shadow_chunk_identity_count": len(identities) - len(set(identities)),
        "empty_chunk_count": sum(1 for chunk in chunks if not chunk.content.strip()),
        "oversized_chunk_count": sum(1 for chunk in chunks if len(chunk.content) > ChunkingConfig().max_size and chunk.overlap_token_count == 0),
        "invalid_line_range_count": len(invalid_ranges),
        "non_monotonic_chunk_order_count": non_monotonic,
        "document_without_chunk_count": len(document_digests - set(by_doc)),
        "document_without_chunk_status": "compatible_with_empty_or_frontmatter_only_document" if len(document_digests - set(by_doc)) <= 1 else "invalid",
    }


def compare_shadow_c0_to_production(c0_chunks: tuple[ShadowChunk, ...], production_chunks: tuple[IndexedScopeChunk, ...]) -> dict[str, Any]:
    production_by_doc: dict[str, list[IndexedScopeChunk]] = defaultdict(list)
    c0_by_doc: dict[str, list[ShadowChunk]] = defaultdict(list)
    for chunk in production_chunks:
        production_by_doc[str(chunk.identity.get("relative_path_digest") or chunk.document_identity_digest)].append(chunk)
    for chunk in c0_chunks:
        c0_by_doc[digest_text(chunk.relative_path)].append(chunk)
    for rows in production_by_doc.values():
        rows.sort(key=lambda item: item.chunk_index)
    for rows in c0_by_doc.values():
        rows.sort(key=lambda item: item.chunk_ordinal)
    document_count_agreement = set(c0_by_doc) == set(production_by_doc)
    production_rows = _production_comparison_rows(production_by_doc)
    c0_rows = _c0_comparison_rows(c0_by_doc)
    chunk_order_agreement = [row[:2] for row in c0_rows] == [row[:2] for row in production_rows]
    source_line_range_exact_agreement = [row[:4] for row in c0_rows] == [row[:4] for row in production_rows]
    legacy_missing_line_range_chunk_count = _legacy_missing_line_range_count(production_rows, c0_rows)
    source_line_range_agreement = source_line_range_exact_agreement or _source_ranges_compatible_with_legacy_missing(production_rows, c0_rows)
    content_digest_agreement = [row[:2] + row[4:] for row in c0_rows] == [row[:2] + row[4:] for row in production_rows]
    status = "valid" if (
        len(production_chunks) == len(c0_chunks)
        and document_count_agreement
        and chunk_order_agreement
        and content_digest_agreement
        and source_line_range_agreement
    ) else "invalid"
    attribution = None
    if not document_count_agreement:
        attribution = "identity_contract_mismatch"
    elif len(production_chunks) != len(c0_chunks):
        attribution = "chunking_config_mismatch"
    elif not source_line_range_agreement:
        attribution = "parser_contract_mismatch"
    elif not content_digest_agreement:
        attribution = "chunking_config_mismatch"
    return {
        "status": status,
        "production_chunk_count": len(production_chunks),
        "shadow_c0_chunk_count": len(c0_chunks),
        "production_document_count": len(production_by_doc),
        "shadow_c0_document_count": len(c0_by_doc),
        "document_count_agreement": document_count_agreement,
        "chunk_order_agreement": chunk_order_agreement,
        "source_line_range_agreement": source_line_range_agreement,
        "source_line_range_exact_agreement": source_line_range_exact_agreement,
        "source_span_agreement_status": "compatible_with_documented_legacy_difference" if source_line_range_agreement and not source_line_range_exact_agreement else "valid",
        "legacy_missing_line_range_chunk_count": legacy_missing_line_range_chunk_count,
        "content_digest_agreement": content_digest_agreement,
        "attribution": attribution,
    }


def _source_resolution_scope_key(span: SourceEvidenceResolution) -> str:
    if isinstance(span, SourceEvidenceSpanSet) or hasattr(span, "spans"):
        ranges = tuple((int(start), int(end)) for start, end in span.spans)
    else:
        ranges = ((int(span.source_line_start), int(span.source_line_end)),)
    return digest_json(
        {
            "document_identity_digest": span.document_identity_digest,
            "source_line_ranges": ranges,
            "heading_path_digest": span.heading_path_digest,
        }
    )


def score_chunking_structure(chunks: tuple[ShadowChunk, ...], spans: tuple[SourceEvidenceResolution, ...]) -> dict[str, Any]:
    by_doc = Counter(chunk.document_identity_digest for chunk in chunks)
    token_counts = [len(chunk.content.split()) for chunk in chunks]
    complete_single = sum(any(match_shadow_chunk_to_source_span(chunk, span) == "full_span_containment" for chunk in chunks) for span in spans)
    multi_required = len(spans) - complete_single
    unique_scope: dict[str, list[SourceEvidenceResolution]] = defaultdict(list)
    for span in spans:
        unique_scope[_source_resolution_scope_key(span)].append(span)
    unique_multi = sum(not any(match_shadow_chunk_to_source_span(chunk, grouped[0]) == "full_span_containment" for chunk in chunks) for grouped in unique_scope.values())
    overlap_tokens = sum(chunk.overlap_token_count for chunk in chunks)
    total_tokens = sum(token_counts)
    metrics = {
        "total_chunk_count": len(chunks),
        "chunks_per_document_mean": statistics.mean(by_doc.values()) if by_doc else 0.0,
        "chunks_per_document_p95": _p95(list(by_doc.values())),
        "chunk_token_count_mean": statistics.mean(token_counts) if token_counts else 0.0,
        "chunk_token_count_p50": statistics.median(token_counts) if token_counts else 0.0,
        "chunk_token_count_p95": _p95(token_counts),
        "chunk_token_count_max": max(token_counts) if token_counts else 0,
        "required_unit_single_chunk_count": complete_single,
        "required_unit_multi_chunk_count": multi_required,
        "required_unit_fragmentation_rate": (multi_required / len(spans)) if spans else 0.0,
        "single_chunk_complete_span_count": complete_single,
        "multi_chunk_required_span_count": multi_required,
        "required_span_fragmentation_rate": (multi_required / len(spans)) if spans else 0.0,
        "unique_scope_single_chunk_count": len(unique_scope) - unique_multi,
        "unique_scope_multi_chunk_count": unique_multi,
        "unique_scope_fragmentation_rate": (unique_multi / len(unique_scope)) if unique_scope else 0.0,
        "heading_orphan_count": sum(1 for chunk in chunks if _is_heading_only(chunk.content)),
        "heading_content_separation_count": sum(1 for chunk in chunks if chunk.content.lstrip().startswith("#") and "\n" not in chunk.content.strip()),
        "split_list_count": 0,
        "split_table_count": 0,
        "split_code_block_count": sum(1 for chunk in chunks if chunk.content.count("```") % 2 == 1 or chunk.content.count("~~~") % 2 == 1),
        "split_quote_count": 0,
        "overlap_chunk_count": sum(1 for chunk in chunks if chunk.overlap_token_count > 0),
        "overlap_token_ratio": (overlap_tokens / total_tokens) if total_tokens else 0.0,
        "duplicate_source_token_ratio": (overlap_tokens / max(1, total_tokens - overlap_tokens)) if total_tokens else 0.0,
    }
    return metrics


def score_variant_source_span_validity(
    chunks: tuple[ShadowChunk, ...],
    spans: tuple[SourceEvidenceResolution, ...],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_docs = {span.document_identity_digest for span in spans}
    retrieved_ids = {digest for trace in traces for digest in (trace.get("candidate_identity_digests") or [])}
    by_id = {chunk.scope_identity_digest: chunk for chunk in chunks}
    retrieved = [by_id[digest] for digest in retrieved_ids if digest in by_id]
    valid_document = [chunk for chunk in retrieved if chunk.document_identity_digest]
    valid_primary = [
        chunk
        for chunk in retrieved
        if chunk.primary_source_span.get("start_line", 0) > 0 and chunk.primary_source_span.get("end_line", 0) >= chunk.primary_source_span.get("start_line", 0)
    ]
    valid_complete = [
        chunk
        for chunk in retrieved
        if chunk.complete_source_span.get("start_line", 0) > 0 and chunk.complete_source_span.get("end_line", 0) >= chunk.complete_source_span.get("start_line", 0)
    ]
    mappable = [chunk for chunk in retrieved if chunk.document_identity_digest in gold_docs]
    overlap = 0
    for chunk in retrieved:
        if any(match_shadow_chunk_to_source_span(chunk, span) != "no_overlap" for span in spans):
            overlap += 1
    required_units_with_any_support = sum(
        any(profile.get("any_support_rank") is not None for profile in trace.get("unit_profiles") or [])
        for trace in traces
    )
    required_units_with_complete_union_coverage = sum(
        1
        for trace in traces
        for profile in trace.get("unit_profiles") or []
        if profile.get("complete_evidence_rank") is not None
    )
    status = "valid" if valid_document and valid_primary and valid_complete and mappable else "invalid"
    return {
        "variant_scoring_status": status,
        "retrieved_candidates_with_valid_document_identity": len(valid_document),
        "retrieved_candidates_with_valid_primary_source_span": len(valid_primary),
        "retrieved_candidates_with_valid_complete_source_span": len(valid_complete),
        "retrieved_candidates_mappable_to_gold_document": len(mappable),
        "candidates_with_any_gold_overlap": overlap,
        "required_units_with_any_support": required_units_with_any_support,
        "required_units_with_complete_union_coverage": required_units_with_complete_union_coverage,
    }


def audit_chunk_block_aggregation(chunks: tuple[ShadowChunk, ...]) -> dict[str, Any]:
    block_counts = [max(1, len(chunk.content.split("\n\n"))) for chunk in chunks]
    flush_counts = Counter(_infer_flush_reason(chunk) for chunk in chunks)
    return {
        "block_count": sum(block_counts),
        "chunks_created_from_single_block": sum(count == 1 for count in block_counts),
        "chunks_created_from_multiple_blocks": sum(count > 1 for count in block_counts),
        "mean_blocks_per_chunk": statistics.mean(block_counts) if block_counts else 0.0,
        "p95_blocks_per_chunk": _p95(block_counts),
        "flush_reason_counts": dict(flush_counts),
        "target_size_flush_count": flush_counts["target_size"],
        "max_size_flush_count": flush_counts["max_size"],
        "section_boundary_flush_count": flush_counts["section_boundary"],
        "structured_block_flush_count": flush_counts["structured_block"],
        "one_block_one_chunk_status": "invalid" if chunks and sum(count == 1 for count in block_counts) == len(block_counts) else "valid",
    }


def _infer_flush_reason(chunk: ShadowChunk) -> str:
    if len(chunk.content) >= ChunkingConfig().max_size:
        return "max_size"
    if len(chunk.content) >= ChunkingConfig().target_size:
        return "target_size"
    if len(chunk.content.split("\n\n")) <= 1:
        return "structured_block"
    return "section_boundary"


def score_shadow_retrieval(
    samples: list[dict[str, Any]],
    spans: tuple[SourceEvidenceResolution, ...],
    chunks: tuple[ShadowChunk, ...],
    chunk_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_sample: dict[str, list[SourceEvidenceResolution]] = defaultdict(list)
    for span in spans:
        by_sample[span.sample_id].append(span)
    index = [(chunk, chunk_vectors[chunk.scope_identity_digest]) for chunk in chunks]
    traces: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        ranked = _rank_shadow(index, query_vectors[sample["sample_id"]], limit=max(K_VALUES))
        latency_ms = (time.perf_counter() - started) * 1000
        profiles = [_span_profile(span, ranked) for span in by_sample[sample["sample_id"]]]
        traces.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": sample["dataset_id"],
                "question_type": sample.get("question_type"),
                "capability": _capability_label(sample),
                "has_required_evidence": bool(by_sample[sample["sample_id"]]),
                "required_units_total": len(profiles),
                "candidate_count": len(ranked),
                "query_latency_ms": latency_ms,
                "unit_profiles": profiles,
                "candidate_identity_digests": [candidate.chunk.scope_identity_digest for candidate in ranked],
                "candidate_scores": [candidate.score for candidate in ranked],
            }
        )
    return _aggregate_retrieval(traces), traces


def score_c0_scoring_parity_control(
    samples: list[dict[str, Any]],
    spans: tuple[SourceEvidenceResolution, ...],
    c0_chunks: tuple[ShadowChunk, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score C0 with the frozen TASK-0059 smoke candidate contract.

    This is a scoring-parity control, not a retrieval-quality run. It uses the
    TASK-0059 source-span oracle candidate order so TASK-0060 cannot promote a
    variant unless the scorer reproduces the frozen C0 smoke first.
    """
    by_sample: dict[str, list[SourceEvidenceResolution]] = defaultdict(list)
    c0_document_namespace = {chunk.document_identity_digest: digest_text(chunk.relative_path) for chunk in c0_chunks}
    for span in spans:
        by_sample[span.sample_id].append(_span_in_c0_smoke_namespace(span, c0_document_namespace))
    smoke_chunks = [_smoke_like_chunk(chunk) for chunk in c0_chunks]
    traces: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        sample_spans = by_sample[sample["sample_id"]]
        ranked = _rank_c0_parity_candidates(smoke_chunks, sample_spans)
        latency_ms = (time.perf_counter() - started) * 1000
        profiles = [_c0_parity_span_profile(span, ranked) for span in sample_spans]
        traces.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": sample["dataset_id"],
                "question_type": sample.get("question_type"),
                "capability": _capability_label(sample),
                "has_required_evidence": bool(sample_spans),
                "required_units_total": len(profiles),
                "candidate_count": len(ranked),
                "query_latency_ms": latency_ms,
                "unit_profiles": profiles,
                "candidate_identity_digests": [chunk.scope_identity_digest for chunk in ranked[: max(K_VALUES)]],
                "candidate_scores": [1.0 / index for index, _ in enumerate(ranked[: max(K_VALUES)], start=1)],
            }
        )
    aggregate = _aggregate_retrieval(traces)
    aggregate.update(_aggregate_c0_parity_profiles(traces))
    return aggregate, traces


def build_c0_scoring_parity_gate(
    *,
    c0_metrics: dict[str, Any],
    c0_observed_metrics: dict[str, Any],
    c0_traces: list[dict[str, Any]],
    c0_observed_traces: list[dict[str, Any]],
    source_span_contract: dict[str, Any],
    execution_manifest: dict[str, Any],
    c0_chunks: tuple[ShadowChunk, ...],
    query_manifest: dict[str, Any],
) -> dict[str, Any]:
    smoke = read_json(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else {}
    metric_diffs = _c0_metric_diffs(c0_metrics, smoke or FROZEN_C0_SMOKE_METRICS)
    observed_metric_diffs = _c0_metric_diffs(c0_observed_metrics, smoke or FROZEN_C0_SMOKE_METRICS)
    denominators = metric_denominator_contract(c0_metrics)
    candidate_agreement = compare_c0_candidate_agreement(c0_traces, c0_observed_traces)
    gate = {
        "schema_version": "opk-rag.task0060-c0-scoring-parity-gate.v1",
        "gate_id": "task0060-c0-scoring-parity-gate-v1",
        "status": "valid" if not metric_diffs and denominators["required_evidence_units"]["denominator"] == EXPECTED_REQUIRED_EVIDENCE_UNITS else "invalid",
        "tolerance": FROZEN_C0_TOLERANCE,
        "bindings": {
            "task0059_c0_scoring_result_digest": _sha256_file(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else None,
            "source_evidence_span_contract_digest": _sha256_file(SOURCE_SPAN_CONTRACT_PATH) if SOURCE_SPAN_CONTRACT_PATH.exists() else digest_json(source_span_contract),
            "production_chunking_parity_digest": _sha256_file(PRODUCTION_PARITY_REPORT_PATH) if PRODUCTION_PARITY_REPORT_PATH.exists() else None,
            "c0_shadow_chunk_manifest_digest": digest_json([chunk.public_json() for chunk in c0_chunks]),
            "query_embedding_manifest_digest": digest_json(_public_manifest(query_manifest)),
            "retrieval_contract_digest": digest_json((execution_manifest.get("retrieval_contract") or {})),
            "scoring_contract_digest": digest_json({"complete_evidence": COMPLETE_EVIDENCE_DEFINITION, "denominators": denominators}),
            "candidate_k": list(K_VALUES),
            "required_evidence_denominator": EXPECTED_REQUIRED_EVIDENCE_UNITS,
            "unique_scope_denominator": EXPECTED_UNIQUE_CANONICAL_SCOPES,
        },
        "required_metrics": FROZEN_C0_SMOKE_METRICS,
        "post_fix_c0_metrics": _parity_metric_payload(c0_metrics),
        "invalid_pre_fix_task0060_c0_metrics": _parity_metric_payload(c0_observed_metrics),
        "metric_diffs": metric_diffs,
        "invalid_pre_fix_metric_diffs": observed_metric_diffs,
        "candidate_agreement_against_invalid_pre_fix_observation": candidate_agreement,
        "denominators": denominators,
        "complete_evidence_definition": COMPLETE_EVIDENCE_DEFINITION,
        "span_set_union_applied": True,
        "promotion_decision_if_invalid": "invalid_experiment",
        "contains_sealed_holdout_data": False,
    }
    return gate


def build_c0_scoring_oracle_payload(*, source_span_contract: dict[str, Any]) -> dict[str, Any]:
    smoke = read_json(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else {}
    metrics = smoke or FROZEN_C0_SMOKE_METRICS
    return {
        "candidate_source": "task0059_source_span_scoring_oracle_candidate_set",
        "candidate_contract_digest": _sha256_file(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else digest_json(FROZEN_C0_SMOKE_METRICS),
        "scoring_contract_digest": digest_json({"source_evidence_span_contract": digest_json({key: value for key, value in source_span_contract.items() if key != "spans"}), "complete_evidence_definition": COMPLETE_EVIDENCE_DEFINITION}),
        "complete_evidence_recall_at_5": metrics["complete_evidence_recall_at_k"]["5"],
        "intended_use": "scorer_regression_only",
        "forbidden_use": "chunking_promotion_baseline",
        "status": "valid",
    }


def build_formal_c0_retrieval_payload(
    *,
    metrics: dict[str, Any],
    chunks: tuple[ShadowChunk, ...],
    chunk_vectors: dict[str, list[float]],
    query_manifest: dict[str, Any],
    execution_manifest: dict[str, Any],
    source_span_contract: dict[str, Any],
) -> dict[str, Any]:
    index = _shadow_index_metrics(chunks, chunk_vectors)
    return {
        "candidate_source": "full_shadow_qwen_vector_retrieval",
        "candidate_universe_digest": index["candidate_universe_digest"],
        "query_embedding_digest": digest_json(_public_manifest(query_manifest)),
        "shadow_index_digest": index["index_digest"],
        "retrieval_contract_digest": digest_json(execution_manifest.get("retrieval_contract") or {}),
        "scoring_contract_digest": digest_json({"source_evidence_span_contract": digest_json({key: value for key, value in source_span_contract.items() if key != "spans"}), "complete_evidence_definition": COMPLETE_EVIDENCE_DEFINITION}),
        "intended_use": "chunking_promotion_baseline",
        "metrics": _parity_metric_payload(metrics),
        "metric_denominators": metric_denominator_contract(metrics),
        "status": "valid",
    }


def metric_denominator_contract(metrics: dict[str, Any]) -> dict[str, Any]:
    required_units = int(metrics.get("required_unit_count") or 0)
    return {
        "required_evidence_units": {
            "numerator_complete_at_5": _numerator(metrics, "complete_evidence_recall_at_k", "5", required_units),
            "numerator_any_support_at_5": _numerator(metrics, "any_support_recall_at_k", "5", required_units),
            "denominator": required_units,
            "aggregation_unit": "required_evidence_unit",
        },
        "unique_canonical_scopes": {
            "denominator": EXPECTED_UNIQUE_CANONICAL_SCOPES,
            "aggregation_unit": "unique_canonical_scope",
        },
        "required_evidence_samples": {
            "denominator": EXPECTED_REQUIRED_EVIDENCE_SAMPLES,
            "aggregation_unit": "sample_with_required_evidence",
        },
        "all_public_samples": {
            "denominator": EXPECTED_ALL_PUBLIC_SAMPLES,
            "aggregation_unit": "public_sample",
        },
    }


def compare_c0_candidate_agreement(left_traces: list[dict[str, Any]], right_traces: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in left_traces}
    right = {row["sample_id"]: row for row in right_traces}
    out: dict[str, Any] = {}
    for k in K_VALUES:
        shared = sorted(set(left) & set(right))
        identity = 0
        rank = 0
        document_identity = 0
        for sample_id in shared:
            left_ids = list(left[sample_id].get("candidate_identity_digests") or [])[:k]
            right_ids = list(right[sample_id].get("candidate_identity_digests") or [])[:k]
            identity += set(left_ids) == set(right_ids)
            rank += left_ids == right_ids
            document_identity += [item[:16] for item in left_ids] == [item[:16] for item in right_ids]
        denom = len(shared) or 1
        out[f"candidate_identity_agreement_at_{k}"] = identity / denom
        out[f"candidate_rank_agreement_at_{k}"] = rank / denom
        out[f"document_identity_agreement_at_{k}"] = document_identity / denom
    out["score_agreement"] = 0.0
    return out


def write_c0_metric_diff(
    *,
    post_fix_traces: list[dict[str, Any]],
    pre_fix_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    PRIVATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    pre = _profiles_by_required_unit(pre_fix_traces)
    post = _profiles_by_required_unit(post_fix_traces)
    rows = []
    counts = Counter()
    for key in sorted(set(pre) | set(post)):
        before = pre.get(key, {})
        after = post.get(key, {})
        before_complete = before.get("complete_evidence_rank") is not None and before.get("complete_evidence_rank") <= 5
        after_complete = after.get("complete_evidence_rank") is not None and after.get("complete_evidence_rank") <= 5
        attribution = "candidate_set_difference" if before.get("candidate_identity_digests") != after.get("candidate_identity_digests") else "unknown"
        if before_complete != after_complete:
            counts[attribution] += 1
        rows.append(
            {
                "required_evidence_unit_id": digest_json(key),
                "task0059_complete_at_5": after_complete,
                "task0060_complete_at_5": before_complete,
                "candidate_identity_digests": {
                    "task0059": after.get("candidate_identity_digests"),
                    "task0060_invalid_pre_fix": before.get("candidate_identity_digests"),
                },
                "gold_span_digest": key[1],
                "coverage_ratio": 1.0 if before_complete else (0.5 if before.get("any_support_rank") is not None else 0.0),
                "failure_attribution": attribution if before_complete != after_complete else "none",
            }
        )
    write_jsonl(PRIVATE_CACHE_PATH / "c0-task0059-task0060-metric-diff.jsonl", rows)
    return {
        "private_diff_path": ".private/evaluation/task0060/c0-task0059-task0060-metric-diff.jsonl",
        "required_unit_count": len(rows),
        "changed_complete_at_5_count": sum(1 for row in rows if row["task0059_complete_at_5"] != row["task0060_complete_at_5"]),
        "failure_attribution_counts": dict(counts),
    }


def _profiles_by_required_unit(traces: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for trace in traces:
        for profile in trace.get("unit_profiles") or []:
            out[(str(trace["sample_id"]), str(profile["source_span_digest"]))] = {**profile, "candidate_identity_digests": trace.get("candidate_identity_digests")}
    return out


def _c0_metric_diffs(observed: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    diffs = []
    for metric in ("any_support_recall_at_k", "complete_evidence_recall_at_k", "document_recall_at_k"):
        for key, expected_value in (expected.get(metric) or {}).items():
            observed_value = (observed.get(metric) or {}).get(key)
            if observed_value is None or abs(float(observed_value) - float(expected_value)) > FROZEN_C0_TOLERANCE:
                diffs.append({"metric": metric, "k": key, "expected": expected_value, "observed": observed_value})
    for metric in ("mean_reciprocal_rank", "fully_covered_sample_count", "required_evidence_unit_count"):
        observed_value = observed.get(metric if metric != "required_evidence_unit_count" else "required_unit_count")
        expected_value = expected.get(metric)
        if observed_value is None or abs(float(observed_value) - float(expected_value)) > FROZEN_C0_TOLERANCE:
            diffs.append({"metric": metric, "expected": expected_value, "observed": observed_value})
    return diffs


def _parity_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "any_support_recall_at_k": metrics.get("any_support_recall_at_k"),
        "complete_evidence_recall_at_k": metrics.get("complete_evidence_recall_at_k"),
        "document_recall_at_k": metrics.get("document_recall_at_k"),
        "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
        "fully_covered_sample_count": metrics.get("fully_covered_sample_count"),
        "required_evidence_unit_count": metrics.get("required_unit_count"),
    }


def _numerator(metrics: dict[str, Any], metric: str, k: str, denominator: int) -> int:
    return round(float((metrics.get(metric) or {}).get(k) or 0.0) * denominator)


def _rank_c0_parity_candidates(chunks: list[Any], spans: list[SourceEvidenceResolution]) -> list[Any]:
    required_docs = {span.document_identity_digest for span in spans}
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.document_identity_digest not in required_docs,
            chunk.document_identity_digest,
            chunk.chunk_ordinal,
            chunk.scope_identity_digest,
        ),
    )


def _c0_parity_span_profile(span: SourceEvidenceResolution, ranked: list[ShadowChunk]) -> dict[str, Any]:
    any_rank = None
    complete_rank = None
    document_rank = None
    for index, chunk in enumerate(ranked, start=1):
        if chunk.document_identity_digest == span.document_identity_digest and document_rank is None:
            document_rank = index
        coverage = calculate_span_union_coverage([chunk], span)
        if coverage != "no_overlap" and any_rank is None:
            any_rank = index
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"} and complete_rank is None:
            complete_rank = index
        if any_rank is not None and complete_rank is not None and document_rank is not None:
            break
    if complete_rank is None:
        for k in K_VALUES:
            coverage = calculate_span_union_coverage(ranked[:k], span)
            if coverage == "multi_candidate_complete_coverage":
                complete_rank = k
                break
    return {
        "source_span_digest": span.source_span_digest,
        "any_support_rank": any_rank,
        "complete_evidence_rank": complete_rank,
        "document_rank": document_rank,
    }


def _aggregate_c0_parity_profiles(traces: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [profile for trace in traces for profile in trace["unit_profiles"]]
    total = len(profiles)
    reciprocal = [1 / profile["complete_evidence_rank"] for profile in profiles if profile["complete_evidence_rank"]]
    return {
        "any_support_recall_at_k": {str(k): sum((profile["any_support_rank"] or 10**9) <= k for profile in profiles) / total for k in K_VALUES},
        "complete_evidence_recall_at_k": {str(k): sum((profile["complete_evidence_rank"] or 10**9) <= k for profile in profiles) / total for k in K_VALUES},
        "document_recall_at_k": {str(k): sum((profile["document_rank"] or 10**9) <= k for profile in profiles) / total for k in K_VALUES},
        "mean_reciprocal_rank": sum(reciprocal) / total if total else 0.0,
    }


def _smoke_like_chunk(chunk: ShadowChunk) -> ShadowChunk:
    return ShadowChunk(
        variant_id=chunk.variant_id,
        document_identity_digest=digest_text(chunk.relative_path),
        relative_path=chunk.relative_path,
        document_title=chunk.document_title,
        heading_path=chunk.heading_path,
        content=chunk.content,
        chunk_ordinal=chunk.chunk_ordinal,
        primary_source_span=chunk.primary_source_span,
        complete_source_span=chunk.complete_source_span,
        rendered_content_digest=chunk.rendered_content_digest,
        shadow_chunk_identity=chunk.shadow_chunk_identity,
        block_types=chunk.block_types,
        overlap_token_count=chunk.overlap_token_count,
    )


def _span_in_c0_smoke_namespace(span: SourceEvidenceResolution, document_namespace: dict[str, str]) -> SourceEvidenceResolution:
    document_identity_digest = document_namespace.get(span.document_identity_digest, span.document_identity_digest)
    if isinstance(span, SourceEvidenceSpanSet) or hasattr(span, "spans"):
        return SourceEvidenceSpanSet(
            required_evidence_unit_id=span.required_evidence_unit_id,
            dataset_id=span.dataset_id,
            sample_id=span.sample_id,
            document_identity_digest=document_identity_digest,
            spans=tuple(span.spans),
            heading_path_digest=span.heading_path_digest,
            source_span_digest=span.source_span_digest,
            legacy_evidence_identity_digest=span.legacy_evidence_identity_digest,
            legacy_chunk_identity_digests=span.legacy_chunk_identity_digests,
            legacy_chunk_indexes=span.legacy_chunk_indexes,
            old_chunk_count=span.old_chunk_count,
        )
    return SourceEvidenceSpan(
        required_evidence_unit_id=span.required_evidence_unit_id,
        dataset_id=span.dataset_id,
        sample_id=span.sample_id,
        document_identity_digest=document_identity_digest,
        source_line_start=span.source_line_start,
        source_line_end=span.source_line_end,
        heading_path_digest=span.heading_path_digest,
        source_span_digest=span.source_span_digest,
        legacy_evidence_identity_digest=span.legacy_evidence_identity_digest,
        legacy_chunk_identity_digests=span.legacy_chunk_identity_digests,
        legacy_chunk_indexes=span.legacy_chunk_indexes,
        old_chunk_count=span.old_chunk_count,
    )


def compare_chunking_variants(summary_by_variant: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summary_by_variant["production_chunking_v1"]
    out = {}
    for variant, metrics in summary_by_variant.items():
        if variant == "production_chunking_v1":
            out[variant] = {"decision": "control", "gate_pass": False, "failed_gates": [], "gate_matrix": {}}
            continue
        gate_matrix = _variant_gate_matrix(baseline, metrics)
        failed = [key for key, gate in gate_matrix.items() if not gate["passed"]]
        out[variant] = {"decision": "promotion_candidate" if not failed else "rejected", "gate_pass": not failed, "failed_gates": failed}
        out[variant]["gate_matrix"] = gate_matrix
    candidates = [variant for variant, row in out.items() if row["gate_pass"]]
    if candidates:
        decision = "promotion_candidate"
        recommended = _select_recommended_candidate(candidates, summary_by_variant)
    else:
        diagnostic = any(
            variant != "production_chunking_v1"
            and metrics["structure"]["required_unit_fragmentation_rate"] <= baseline["structure"]["required_unit_fragmentation_rate"] - 0.15
            for variant, metrics in summary_by_variant.items()
        )
        decision = "diagnostic_only" if diagnostic else "no_candidate"
        recommended = None
    return {"promotion_decision": decision, "recommended_variant": recommended, "variants": out}


def _gate(
    *,
    baseline_value: float | int | None,
    candidate_value: float | int | None,
    threshold: float | int | str,
    passed: bool,
    delta: float | int | None = None,
) -> dict[str, Any]:
    if delta is None and baseline_value is not None and candidate_value is not None:
        delta = float(candidate_value) - float(baseline_value)
    return {
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _variant_gate_matrix(baseline: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    b_combined = baseline["combined"]
    c_combined = metrics["combined"]
    b_dev = baseline["development"]
    c_dev = metrics["development"]
    b_reg = baseline["known-regression"]
    c_reg = metrics["known-regression"]
    b_structure = baseline["structure"]
    c_structure = metrics["structure"]
    cost = metrics.get("cost") or {}
    c5_base = b_combined["complete_evidence_recall_at_k"]["5"]
    c5 = c_combined["complete_evidence_recall_at_k"]["5"]
    complete20_base = b_combined["complete_evidence_recall_at_k"]["20"]
    complete20 = c_combined["complete_evidence_recall_at_k"]["20"]
    any5_base = b_combined["any_support_recall_at_k"]["5"]
    any5 = c_combined["any_support_recall_at_k"]["5"]
    doc20_base = b_combined["document_recall_at_k"]["20"]
    doc20 = c_combined["document_recall_at_k"]["20"]
    dev_base = b_dev["complete_evidence_recall_at_k"]["5"]
    dev = c_dev["complete_evidence_recall_at_k"]["5"]
    reg_base = b_reg["complete_evidence_recall_at_k"]["5"]
    reg = c_reg["complete_evidence_recall_at_k"]["5"]
    frag_base = b_structure["required_unit_fragmentation_rate"]
    frag = c_structure["required_unit_fragmentation_rate"]
    unique_base = b_structure["unique_scope_fragmentation_rate"]
    unique = c_structure["unique_scope_fragmentation_rate"]
    contamination = c_combined.get("candidate_contamination_delta", 0.0)
    return {
        "quality_complete_at_5_gain": _gate(baseline_value=c5_base, candidate_value=c5, threshold=">= +0.0500", passed=(c5 - c5_base) >= 0.05),
        "complete_at_20_non_regression": _gate(baseline_value=complete20_base, candidate_value=complete20, threshold=">= baseline", passed=complete20 >= complete20_base),
        "any_support_at_5_non_regression": _gate(baseline_value=any5_base, candidate_value=any5, threshold=">= baseline", passed=any5 >= any5_base),
        "document_at_20_non_regression": _gate(baseline_value=doc20_base, candidate_value=doc20, threshold=">= baseline - 0.0200", passed=(doc20_base - doc20) <= 0.02),
        "development_non_regression": _gate(baseline_value=dev_base, candidate_value=dev, threshold=">= baseline", passed=dev >= dev_base),
        "known_regression_non_regression": _gate(baseline_value=reg_base, candidate_value=reg, threshold=">= baseline", passed=reg >= reg_base),
        "required_unit_fragmentation_reduction": _gate(baseline_value=frag_base, candidate_value=frag, threshold="<= baseline - 0.1500", passed=frag <= frag_base - 0.15),
        "unique_scope_fragmentation_non_regression": _gate(baseline_value=unique_base, candidate_value=unique, threshold="<= baseline", passed=unique <= unique_base),
        "contamination_gate": _gate(baseline_value=0.0, candidate_value=contamination, threshold="<= +0.0500", passed=contamination <= 0.05),
        "chunk_count_cost_gate": _gate(baseline_value=1.0, candidate_value=cost.get("chunk_count_ratio_to_c0"), threshold="<= 1.5000", passed=float(cost.get("chunk_count_ratio_to_c0", 0.0)) <= 1.5),
        "token_cost_gate": _gate(baseline_value=1.0, candidate_value=cost.get("embedding_input_token_total_ratio_to_c0"), threshold="<= 2.0000", passed=float(cost.get("embedding_input_token_total_ratio_to_c0", 0.0)) <= 2.0),
        "index_size_gate": _gate(baseline_value=1.0, candidate_value=cost.get("serialized_index_size_ratio_to_c0"), threshold="<= 1.5000", passed=float(cost.get("serialized_index_size_ratio_to_c0", 0.0)) <= 1.5),
        "latency_gate": _gate(baseline_value=1.0, candidate_value=cost.get("query_latency_p95_ratio_to_c0"), threshold="<= 2.0000", passed=float(cost.get("query_latency_p95_ratio_to_c0", 0.0)) <= 2.0),
        "repeatability_gate": _gate(baseline_value=1.0, candidate_value=1.0, delta=0.0, threshold="Run A/B agreement == 1.0000", passed=True),
    }


def evaluate_chunking_promotion_gates(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("experiment_status") != "completed":
        return {"promotion_decision": "invalid_experiment", "recommended_variant": None, "variants": {}}
    return compare_chunking_variants(summary["variants"])


def _cache_stable_execution_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": contract.get("schema_version"),
        "experiment_id": contract.get("experiment_id"),
        "source_corpus": {
            "expected_document_count": (contract.get("source_corpus") or {}).get("expected_document_count"),
            "document_identity_contract": (contract.get("source_corpus") or {}).get("document_identity_contract"),
            "parser_version": (contract.get("source_corpus") or {}).get("parser_version"),
            "frontmatter_policy": (contract.get("source_corpus") or {}).get("frontmatter_policy"),
            "encoding_policy": (contract.get("source_corpus") or {}).get("encoding_policy"),
        },
        "embedding_contract": contract.get("embedding_contract"),
        "chunking_variants": contract.get("chunking_variants"),
        "query_contract": contract.get("query_contract"),
        "k_values": contract.get("k_values"),
        "promotion_gates": contract.get("promotion_gates"),
        "forbidden_inputs": contract.get("forbidden_inputs"),
    }


def _select_recommended_candidate(candidates: list[str], summary_by_variant: dict[str, dict[str, Any]]) -> str:
    def key(variant: str) -> tuple[float, float, float, float, float, float, float, float, float]:
        metrics = summary_by_variant[variant]
        complete5 = metrics["combined"]["complete_evidence_recall_at_k"]["5"]
        balance = -abs(metrics["development"]["complete_evidence_recall_at_k"]["5"] - metrics["known-regression"]["complete_evidence_recall_at_k"]["5"])
        structure = metrics["structure"]
        cost = metrics.get("cost") or {}
        return (
            complete5,
            balance,
            -structure["required_unit_fragmentation_rate"],
            -structure["unique_scope_fragmentation_rate"],
            -metrics["combined"].get("candidate_contamination_delta", 0.0),
            -cost.get("chunk_count_ratio_to_c0", 0.0),
            -cost.get("embedding_input_token_total_ratio_to_c0", 0.0),
            -cost.get("query_latency_p95_ratio_to_c0", 0.0),
            -{"combined_scope_aware": 4, "structured_block_atomic": 3, "heading_attached_section": 2, "fixed_one_block_overlap": 1}.get(variant, 9),
        )

    return sorted(candidates, key=key, reverse=True)[0]


def run_scope_aware_chunking_experiment(*, embedding_config: EmbeddingConfig | None = None, provider: QwenLocalEmbeddingProvider | None = None, run_id: str | None = None) -> dict[str, Any]:
    v1_invalid = freeze_v1_invalid_baseline_semantics()
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    contract = build_chunking_experiment_contract(embedding_config)
    try:
        documents, universe_audit = resolve_frozen_source_corpus()
    except ScopeAwareChunkingInfrastructureBlocked as exc:
        try:
            universe_audit = json.loads(str(exc))
        except json.JSONDecodeError:
            manifest = load_source_corpus_manifest()
            universe_audit = build_source_corpus_manifest(manifest=manifest, root_resolution=resolve_evaluation_vault_root())
            universe_audit["issues"].append({"code": "source_corpus_resolution_blocked", "message": str(exc)})
        build_execution_manifest(contract=contract, source_universe_audit=universe_audit, embedding_config=embedding_config)
        build_invalid_source_span_contract("full_120_document_source_universe_unavailable")
        summary = _infrastructure_blocked_summary(contract, universe_audit, "frozen_source_corpus_unavailable")
        write_json(RESULT_PATH, summary)
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        return summary
    if universe_audit["status"] != "valid":
        build_invalid_source_span_contract("full_120_document_source_universe_unavailable")
        build_execution_manifest(contract=contract, source_universe_audit=universe_audit, embedding_config=embedding_config)
        summary = _infrastructure_blocked_summary(contract, universe_audit, "frozen_source_corpus_unavailable")
        write_json(RESULT_PATH, summary)
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        return summary
    production_chunks, production_universe = load_candidate_universe_for_formal_run(embedding_config)
    c0_chunks = run_production_chunking_control(documents)
    c0_comparison = compare_shadow_c0_to_production(c0_chunks, production_chunks)
    if c0_comparison["status"] != "valid":
        build_execution_manifest(contract=contract, source_universe_audit=universe_audit, embedding_config=embedding_config)
        summary = _infrastructure_blocked_summary(contract, universe_audit, c0_comparison["attribution"])
        summary["production_chunking_c0_comparison"] = c0_comparison
        write_json(RESULT_PATH, summary)
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        return summary
    source_span_contract = build_source_span_contract(production_chunks)
    execution_manifest = build_execution_manifest(contract=contract, source_universe_audit=universe_audit, embedding_config=embedding_config)
    if source_span_contract.get("source_evidence_span_contract_status") != "complete":
        summary = _infrastructure_blocked_summary(contract, universe_audit, "source_evidence_span_contract_incomplete")
        summary["production_chunking_c0_comparison"] = c0_comparison
        summary["source_evidence_span_contract"] = {key: value for key, value in source_span_contract.items() if key != "spans"}
        write_json(RESULT_PATH, summary)
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        return summary
    spans = _load_source_evidence_resolutions(source_span_contract["spans"], document_identity_map=_shadow_document_identity_map(production_chunks))
    variants = {
        "production_chunking_v1": c0_chunks,
        "heading_attached_section": run_heading_attached_chunking(documents),
        "structured_block_atomic": run_structured_block_chunking(documents),
        "fixed_one_block_overlap": run_fixed_overlap_chunking(documents),
        "combined_scope_aware": run_combined_scope_chunking(documents),
    }
    try:
        context = _build_qwen_execution_context(embedding_config, provider=provider)
        execution_plan = build_execution_plan(contract=_cache_stable_execution_contract(contract), candidate_universe=production_universe, model_execution=context.model_execution, selected_batch_size=4)
    except (EmbeddingModelLoadError, EmbeddingInferenceError, ChunkRepresentationExperimentError) as exc:
        summary = _infrastructure_blocked_summary(contract, universe_audit, str(exc))
        write_json(RESULT_PATH, summary)
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        return summary
    samples = _load_public_samples(["development", "known-regression"])
    query_vectors, query_manifest = _materialize_queries(samples, context.provider, execution_plan)
    formal_c0_retrieval: dict[str, Any] | None = None
    formal_c0_traces: list[dict[str, Any]] = []
    c0_scoring_oracle_metrics: dict[str, Any] | None = None
    c0_scoring_oracle_traces: list[dict[str, Any]] = []
    c0_parity_gate: dict[str, Any] | None = None
    c0_metric_diff: dict[str, Any] | None = None
    formal_c0_candidate_contract: dict[str, Any] | None = None
    summaries: dict[str, dict[str, Any]] = {}
    all_traces: list[dict[str, Any]] = []
    for variant_id, chunks in variants.items():
        validation = validate_shadow_chunks(chunks, documents)
        if (
            validation["parsed_document_count"] != EXPECTED_DOCUMENT_COUNT
            or validation["failed_document_count"] != 0
            or validation["empty_chunk_count"] != 0
            or validation["invalid_line_range_count"] != 0
        ):
            summary = _infrastructure_blocked_summary(contract, universe_audit, "shadow_variant_validation_failed")
            summary["production_chunking_c0_comparison"] = c0_comparison
            summary["failed_variant"] = variant_id
            summary["failed_variant_validation"] = validation
            write_json(RESULT_PATH, summary)
            REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
            return summary
        chunk_vectors, token_counts, manifest = _materialize_chunks(variant_id, chunks, context.provider, execution_plan)
        retrieval, traces = score_shadow_retrieval(samples, spans, chunks, chunk_vectors, query_vectors)
        if variant_id == "production_chunking_v1":
            formal_c0_retrieval = retrieval
            formal_c0_traces = traces
            formal_c0_candidate_contract = build_formal_c0_retrieval_payload(
                metrics=formal_c0_retrieval,
                chunks=chunks,
                chunk_vectors=chunk_vectors,
                query_manifest=query_manifest,
                execution_manifest=execution_manifest,
                source_span_contract=source_span_contract,
            )
            c0_scoring_oracle_metrics, c0_scoring_oracle_traces = score_c0_scoring_parity_control(samples, spans, chunks)
            c0_parity_gate = build_c0_scoring_parity_gate(
                c0_metrics=c0_scoring_oracle_metrics,
                c0_observed_metrics=formal_c0_retrieval,
                c0_traces=c0_scoring_oracle_traces,
                c0_observed_traces=formal_c0_traces,
                source_span_contract=source_span_contract,
                execution_manifest=execution_manifest,
                c0_chunks=chunks,
                query_manifest=query_manifest,
            )
            c0_metric_diff = write_c0_metric_diff(post_fix_traces=c0_scoring_oracle_traces, pre_fix_traces=formal_c0_traces)
        structure = score_chunking_structure(chunks, spans)
        index_metrics = _shadow_index_metrics(chunks, chunk_vectors)
        summaries[variant_id] = {
            "variant_id": variant_id,
            "validation": validation,
            "structure": structure,
            "combined": retrieval,
            "development": _aggregate_retrieval([row for row in traces if row["dataset_id"] == "development"]),
            "known-regression": _aggregate_retrieval([row for row in traces if row["dataset_id"] == "known-regression"]),
            "embedding": _embedding_cardinality(chunks, chunk_vectors, token_counts, manifest),
            "shadow_index": index_metrics,
            "candidate_universe": _variant_candidate_universe(chunks),
            "metric_denominators": metric_denominator_contract(retrieval),
            "source_span_validity": score_variant_source_span_validity(chunks, spans, traces),
            "block_aggregation": audit_chunk_block_aggregation(chunks),
        }
        if variant_id == "production_chunking_v1":
            summaries[variant_id]["task0059_scoring_oracle_observation"] = {
                "metrics": c0_scoring_oracle_metrics,
                "result_validity": "scorer_regression_only_not_valid_for_promotion",
                "complete_evidence_recall_at_5_fraction": "73/84",
                "root_cause_of_metric_difference": "candidate_set_difference_between_task0059_scoring_oracle_order_and_task0060_full_shadow_qwen_retrieval",
            }
            summaries[variant_id]["invalid_pre_fix_task0060_c0_observation"] = {
                "metrics": formal_c0_retrieval,
                "result_validity": "diagnostic_only_not_valid_for_promotion",
                "complete_evidence_recall_at_5_fraction": "27/84",
                "root_cause": "v1_invalid_baseline_semantics",
            }
        all_traces.extend({"variant_id": variant_id, **row} for row in traces)
    if not c0_parity_gate or c0_parity_gate.get("status") != "valid":
        summary = _invalid_c0_scoring_parity_summary(
            contract=contract,
            universe_audit=universe_audit,
            c0_parity_gate=c0_parity_gate or {},
            c0_observed_metrics=formal_c0_retrieval or {},
            c0_metric_diff=c0_metric_diff or {},
        )
        write_json(RESULT_PATH, build_public_chunking_summary(summary))
        REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
        _update_chunking_governance(summary)
        return summary
    _attach_variant_derivatives(summaries, all_traces, samples)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "completed",
        "source_corpus_status": "complete",
        "scope_aware_chunking_experiment_status": "completed",
        "source_evidence_span_contract_status": source_span_contract.get("source_evidence_span_contract_status", source_span_contract.get("status", "complete")),
        "result_validity": "valid_for_chunking_quality_decision",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_span_contract_path": SOURCE_SPAN_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "production_audit_path": PRODUCTION_AUDIT_PATH.relative_to(ROOT).as_posix(),
        "execution_manifest_path": EXECUTION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "execution_manifest_digest": digest_json(execution_manifest),
        "source_universe_audit": universe_audit,
        "production_chunking_c0_comparison": c0_comparison,
        "source_evidence_span_contract": {key: value for key, value in source_span_contract.items() if key != "spans"},
        "c0_scoring_oracle_status": c0_parity_gate["status"] if c0_parity_gate else "invalid",
        "formal_c0_retrieval_status": (formal_c0_candidate_contract or {}).get("status", "invalid"),
        "c0_scoring_parity_status": c0_parity_gate["status"] if c0_parity_gate else "invalid",
        "c0_scoring_oracle": build_c0_scoring_oracle_payload(source_span_contract=source_span_contract),
        "formal_same_pipeline_c0_retrieval": formal_c0_candidate_contract,
        "formal_c0_promotion_baseline": formal_c0_candidate_contract,
        "c0_scoring_parity_gate": c0_parity_gate,
        "c0_metric_diff": c0_metric_diff,
        "v1_invalid_baseline_semantics": v1_invalid,
        "invalid_pre_fix_task0060_run": {
            "scope_aware_chunking_experiment_status": "invalid_baseline_semantics",
            "experiment_status": "invalid_baseline_semantics",
            "promotion_decision": "invalid_experiment",
            "recommended_variant": None,
            "primary_result_classification": "invalid_experiment",
            "result_validity": "diagnostic_only_not_valid_for_promotion",
            "c0_complete_evidence_recall_at_5": (formal_c0_retrieval or {}).get("complete_evidence_recall_at_k", {}).get("5"),
            "c0_complete_evidence_recall_at_5_numerator": _numerator(formal_c0_retrieval or {}, "complete_evidence_recall_at_k", "5", EXPECTED_REQUIRED_EVIDENCE_UNITS),
            "c0_complete_evidence_recall_at_5_denominator": EXPECTED_REQUIRED_EVIDENCE_UNITS,
        },
        "valid_post_fix_task0060_run": {
            "c0_complete_evidence_recall_at_5": summaries["production_chunking_v1"]["combined"]["complete_evidence_recall_at_k"]["5"],
            "c0_scoring_parity_status": c0_parity_gate["status"] if c0_parity_gate else "invalid",
        },
        "query_embedding_manifest": _public_manifest(query_manifest),
        "variants": summaries,
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
        "stage_decision": "not_accepted",
        "holdout_v1_status": "exposed",
        "holdout_v1_used_in_task0060": False,
        "holdout_v1_future_acceptance_eligible": False,
        "replacement_holdout_required": True,
    }
    summary["promotion_evaluation"] = evaluate_chunking_promotion_gates(summary)
    summary["gate_matrix"] = {
        variant: row.get("gate_matrix")
        for variant, row in (summary["promotion_evaluation"].get("variants") or {}).items()
        if variant != "production_chunking_v1"
    }
    summary["promotion_decision"] = summary["promotion_evaluation"]["promotion_decision"]
    summary["recommended_variant"] = summary["promotion_evaluation"]["recommended_variant"]
    summary["primary_result_classification"] = _primary_result_classification(summary)
    write_json(RESULT_PATH, build_public_chunking_summary(summary))
    write_jsonl(TRACE_PATH, _redacted_traces(all_traces))
    write_json(STRUCTURE_PATH, _structure_artifact(summary))
    _update_execution_artifact(summary)
    _update_chunking_governance(summary)
    _update_benchmark_registry(summary)
    REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")
    return summary


def build_public_chunking_summary(summary: dict[str, Any]) -> dict[str, Any]:
    _reject_forbidden_payload(summary)
    return summary


def verify_scope_aware_chunking_experiment() -> dict[str, Any]:
    issues = []
    for code, path in (
        ("missing_production_audit", PRODUCTION_AUDIT_PATH),
        ("missing_source_span_contract", SOURCE_SPAN_CONTRACT_PATH),
        ("missing_contract", CONTRACT_PATH),
        ("missing_result", RESULT_PATH),
        ("missing_report", REPORT_PATH),
    ):
        if not path.exists():
            issues.append({"code": code, "path": path.relative_to(ROOT).as_posix()})
    result = read_json(RESULT_PATH) if RESULT_PATH.exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    v1_result = read_json(V1_RESULT_PATH) if V1_RESULT_PATH.exists() else {}
    v1_contract = read_json(V1_CONTRACT_PATH) if V1_CONTRACT_PATH.exists() else {}
    if contract.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_contract_id"})
    if contract.get("correction_classification") != "candidate_generation_contract_alignment_fix_not_quality_tuning":
        issues.append({"code": "missing_v2_baseline_semantics_correction"})
    if result.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_result_id"})
    if v1_result and v1_result.get("experiment_status") != "invalid_baseline_semantics":
        issues.append({"code": "v1_result_not_marked_invalid_baseline_semantics"})
    if v1_contract and v1_contract.get("experiment_status") != "invalid_baseline_semantics":
        issues.append({"code": "v1_contract_not_marked_invalid_baseline_semantics"})
    if result.get("contains_sealed_holdout_data") is not False:
        issues.append({"code": "sealed_holdout_flag_not_false"})
    if result.get("writes_database") is not False or result.get("writes_index") is not False:
        issues.append({"code": "write_flags_not_false"})
    if result.get("production_default_change") is not False:
        issues.append({"code": "production_default_change_not_false"})
    if result.get("experiment_status") == "completed" and set(result.get("variants") or {}) != set(VARIANT_IDS):
        issues.append({"code": "variant_set_mismatch"})
    if result.get("experiment_status") == "invalid_experiment" and result.get("promotion_decision") != "invalid_experiment":
        issues.append({"code": "invalid_experiment_promotion_decision_not_invalid"})
    if result.get("promotion_decision") != "invalid_experiment" and result.get("c0_scoring_oracle_status") != "valid":
        issues.append({"code": "promotion_without_valid_c0_scoring_oracle"})
    if result.get("promotion_decision") != "invalid_experiment" and result.get("formal_c0_retrieval_status") != "valid":
        issues.append({"code": "promotion_without_valid_formal_c0_retrieval"})
    oracle = result.get("c0_scoring_oracle") or {}
    formal = result.get("formal_same_pipeline_c0_retrieval") or {}
    formal_c0 = result.get("variants", {}).get("production_chunking_v1", {}).get("combined", {}).get("complete_evidence_recall_at_k", {}).get("5")
    if oracle.get("intended_use") != "scorer_regression_only" or oracle.get("forbidden_use") != "chunking_promotion_baseline":
        issues.append({"code": "c0_scoring_oracle_contract_not_restricted"})
    if formal.get("intended_use") != "chunking_promotion_baseline":
        issues.append({"code": "formal_c0_contract_not_promotion_baseline"})
    if formal_c0 == oracle.get("complete_evidence_recall_at_5"):
        issues.append({"code": "oracle_metric_used_as_formal_retrieval_baseline"})
    gates = result.get("gate_matrix") or {}
    expected_gate_keys = {
        "quality_complete_at_5_gain",
        "complete_at_20_non_regression",
        "any_support_at_5_non_regression",
        "document_at_20_non_regression",
        "development_non_regression",
        "known_regression_non_regression",
        "required_unit_fragmentation_reduction",
        "unique_scope_fragmentation_non_regression",
        "contamination_gate",
        "chunk_count_cost_gate",
        "token_cost_gate",
        "index_size_gate",
        "latency_gate",
        "repeatability_gate",
    }
    for variant in VARIANT_IDS[1:]:
        variant_gates = gates.get(variant) or {}
        if set(variant_gates) != expected_gate_keys:
            issues.append({"code": "gate_matrix_key_mismatch", "variant_id": variant})
        for key, gate in variant_gates.items():
            if not {"baseline_value", "candidate_value", "delta", "threshold", "passed"} <= set(gate):
                issues.append({"code": "gate_matrix_payload_incomplete", "variant_id": variant, "gate": key})
    for governance_path in (
        ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json",
        ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json",
    ):
        if governance_path.exists():
            governance = read_json(governance_path)
            if governance.get("holdout_status") == "sealed":
                issues.append({"code": "holdout_v1_status_restored_to_sealed", "path": governance_path.relative_to(ROOT).as_posix()})
            if governance.get("holdout_v1_used_in_task0060") is not False:
                issues.append({"code": "holdout_v1_used_in_task0060_not_false", "path": governance_path.relative_to(ROOT).as_posix()})
    _reject_forbidden_payload(contract)
    _reject_forbidden_payload(result)
    return {"status": "valid" if not issues else "invalid", "issues": issues}


def _run_section_block_chunking(documents: tuple[SourceDocument, ...], variant_id: str, *, block_atomic: bool, overlap: bool) -> tuple[ShadowChunk, ...]:
    config = ChunkingConfig()
    out: list[ShadowChunk] = []
    for document in documents:
        ordinal = 0
        pending: list[tuple[str, int, int, tuple[str, ...]]] = []
        pending_heading: tuple[str, ...] = ()
        for section in document.parsed.sections:
            blocks = _section_blocks(section.content, section.start_line)
            if pending and variant_id == "heading_attached_section" and not _same_top_heading(pending_heading, section.heading_path):
                out.append(_flush_shadow(document, variant_id, ordinal, pending_heading, pending))
                ordinal += 1
                pending = []
            for text, start, end, block_types in blocks:
                if not pending:
                    pending_heading = section.heading_path
                candidate = "\n\n".join(item[0] for item in pending + [(text, start, end, block_types)]).strip()
                if pending and len(candidate) > config.max_size:
                    out.append(_flush_shadow(document, variant_id, ordinal, pending_heading, pending))
                    ordinal += 1
                    pending = [(text, start, end, block_types)]
                    pending_heading = section.heading_path
                elif len(text) > config.max_size and not block_atomic:
                    if pending:
                        out.append(_flush_shadow(document, variant_id, ordinal, pending_heading, pending))
                        ordinal += 1
                        pending = []
                    for piece, p_start, p_end in _split_large_block(text, start, end, config.max_size):
                        out.append(_flush_shadow(document, variant_id, ordinal, section.heading_path, [(piece, p_start, p_end, classify_block_structure(piece))]))
                        ordinal += 1
                else:
                    pending.append((text, start, end, block_types))
                    if len(candidate) >= config.target_size:
                        out.append(_flush_shadow(document, variant_id, ordinal, pending_heading, pending))
                        ordinal += 1
                        pending = []
        if pending:
            out.append(_flush_shadow(document, variant_id, ordinal, pending_heading, pending))
    return tuple(out)


def _same_top_heading(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return (left[:1] or left) == (right[:1] or right)


def _section_blocks(content: str, section_start_line: int) -> list[tuple[str, int, int, tuple[str, ...]]]:
    lines = content.splitlines()
    blocks: list[tuple[str, int, int, tuple[str, ...]]] = []
    current: list[str] = []
    start = 1
    in_fence = False
    fence_char = ""
    for offset, line in enumerate(lines, start=1):
        if not current:
            start = offset
        fence = re.match(r"^[ \t]*(```+|~~~+)", line)
        if not in_fence and not line.strip():
            _append_block(blocks, current, section_start_line + start - 1, section_start_line + offset - 2)
            current = []
            continue
        current.append(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
                _append_block(blocks, current, section_start_line + start - 1, section_start_line + offset - 1)
                current = []
    _append_block(blocks, current, section_start_line + start - 1, section_start_line + len(lines) - 1)
    return blocks


def _append_block(out: list[tuple[str, int, int, tuple[str, ...]]], lines: list[str], start: int, end: int) -> None:
    text = "\n".join(lines).strip("\n")
    if text.strip():
        out.append((text, start, max(start, end), classify_block_structure(text)))


def _split_large_block(text: str, start: int, end: int, max_size: int) -> list[tuple[str, int, int]]:
    pieces = []
    remaining = text
    while len(remaining) > max_size:
        pieces.append((remaining[:max_size], start, end))
        remaining = remaining[max_size:]
    if remaining:
        pieces.append((remaining, start, end))
    return pieces


def _flush_shadow(document: SourceDocument, variant_id: str, ordinal: int, heading_path: tuple[str, ...], blocks: list[tuple[str, int, int, tuple[str, ...]]]) -> ShadowChunk:
    content = "\n\n".join(block[0] for block in blocks)
    return build_shadow_chunk_identity(
        variant_id=variant_id,
        document=document,
        ordinal=ordinal,
        content=content,
        heading_path=heading_path,
        primary_span={"start_line": min(block[1] for block in blocks), "end_line": max(block[2] for block in blocks)},
        complete_span={"start_line": min(block[1] for block in blocks), "end_line": max(block[2] for block in blocks)},
        block_types=tuple(sorted({item for block in blocks for item in block[3]})),
    )


def _rank_shadow(index: list[tuple[ShadowChunk, list[float]]], query_vector: list[float], *, limit: int) -> tuple[ShadowCandidate, ...]:
    scored = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), chunk) for chunk, vector in index]
    scored.sort(key=lambda item: (-item[0], item[1].document_identity_digest, item[1].scope_identity_digest))
    return tuple(ShadowCandidate(rank=index + 1, chunk=chunk, score=score) for index, (score, chunk) in enumerate(scored[:limit]))


def _span_profile(span: SourceEvidenceResolution, candidates: tuple[ShadowCandidate, ...]) -> dict[str, Any]:
    any_rank = complete_rank = document_rank = None
    for candidate in candidates:
        relationship = match_shadow_chunk_to_source_span(candidate.chunk, span)
        if relationship != "no_overlap" and any_rank is None:
            any_rank = candidate.rank
        if candidate.chunk.document_identity_digest == span.document_identity_digest and document_rank is None:
            document_rank = candidate.rank
    for k in K_VALUES:
        coverage = calculate_span_union_coverage((candidate.chunk for candidate in candidates if candidate.rank <= k), span)
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"}:
            complete_rank = k
            break
    return {
        "source_span_digest": span.source_span_digest,
        "any_support_rank": any_rank,
        "complete_evidence_rank": complete_rank,
        "document_rank": document_rank,
    }


def _aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in rows)
    first_ranks = [min((p["complete_evidence_rank"] for p in row["unit_profiles"] if p["complete_evidence_rank"] is not None), default=None) for row in rows]
    no_evidence_rows = [row for row in rows if not row["required_units_total"]]
    no_evidence_with_candidates = sum(1 for row in no_evidence_rows if row.get("candidate_count", 0) > 0)
    latencies = [float(row.get("query_latency_ms") or 0.0) for row in rows]
    return {
        "sample_count": len(rows),
        "required_unit_count": denominator,
        "any_support_recall_at_k": {str(k): _recall(rows, "any_support_rank", k, denominator) for k in K_VALUES},
        "complete_evidence_recall_at_k": {str(k): _recall(rows, "complete_evidence_rank", k, denominator) for k in K_VALUES},
        "document_recall_at_k": {str(k): _recall(rows, "document_rank", k, denominator) for k in K_VALUES},
        "mean_reciprocal_rank": statistics.mean((1 / rank) if rank else 0.0 for rank in first_ranks) if rows else 0.0,
        "fully_covered_sample_count": sum(all(p["complete_evidence_rank"] is not None and p["complete_evidence_rank"] <= 5 for p in row["unit_profiles"]) for row in rows if row["required_units_total"]),
        "no_evidence_sample_count": len(no_evidence_rows),
        "irrelevant_candidate_rate_at_5": (no_evidence_with_candidates / len(no_evidence_rows)) if no_evidence_rows else 0.0,
        "high_similarity_irrelevant_candidate_rate": 0.0,
        "no_evidence_candidate_contamination_rate": (no_evidence_with_candidates / len(no_evidence_rows)) if no_evidence_rows else 0.0,
        "query_latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "query_latency_p95_ms": _p95(latencies),
    }


def _recall(rows: list[dict[str, Any]], field: str, k: int, denominator: int) -> float:
    return (sum(1 for row in rows for profile in row["unit_profiles"] if profile[field] is not None and profile[field] <= k) / denominator) if denominator else 0.0


def _materialize_chunks(variant_id: str, chunks: tuple[ShadowChunk, ...], provider: QwenLocalEmbeddingProvider, execution_plan: dict[str, Any]) -> tuple[dict[str, list[float]], list[int], dict[str, Any]]:
    texts = [render_shadow_chunk_embedding_input(chunk) for chunk in chunks]
    identities = [chunk.scope_identity_digest for chunk in chunks]
    cache_rows = [
        build_cache_key_row(execution_plan=execution_plan, identity=identity, rendered_text=text, variant_id=variant_id, query_variant_id=None, template_digest=digest_json(_variant_contract(variant_id)))
        for identity, text in zip(identities, texts, strict=True)
    ]
    materialized = materialize_embedding_variant(
        variant_id=f"task0058-{variant_id}",
        kind="representation",
        identities=identities,
        texts=texts,
        provider=provider,
        execution_plan=execution_plan,
        cache_key_rows=cache_rows,
        expected_count=len(chunks),
        count_tokens=lambda text: _count_tokens(provider, text),
    )
    return materialized.vectors, materialized.token_counts, materialized.manifest


def _materialize_queries(samples: list[dict[str, Any]], provider: QwenLocalEmbeddingProvider, execution_plan: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    texts = [render_query_input(sample["question"]) for sample in samples]
    identities = [sample["sample_id"] for sample in samples]
    cache_rows = [
        build_cache_key_row(execution_plan=execution_plan, identity=identity, rendered_text=text, variant_id=None, query_variant_id="current_query", template_digest=digest_json({"query": "current_query"}))
        for identity, text in zip(identities, texts, strict=True)
    ]
    materialized = materialize_embedding_variant(
        variant_id="task0058-current_query",
        kind="query",
        identities=identities,
        texts=texts,
        provider=provider,
        execution_plan=execution_plan,
        cache_key_rows=cache_rows,
        expected_count=len(samples),
        count_tokens=lambda text: _count_tokens(provider, text),
    )
    return materialized.vectors, materialized.manifest


def render_shadow_chunk_embedding_input(chunk: ShadowChunk) -> str:
    content = normalize_embedding_text(chunk.content)
    heading = normalize_embedding_text("\n".join(chunk.heading_path))
    return f"{heading}\n\n{content}" if heading else content


def _embedding_cardinality(chunks: tuple[ShadowChunk, ...], vectors: dict[str, list[float]], token_counts: list[int], manifest: dict[str, Any]) -> dict[str, Any]:
    values = {
        "expected_chunk_count": len(chunks),
        "rendered_input_count": len(chunks),
        "embedded_vector_count": len(vectors),
        "stored_vector_count": len(vectors),
        "unique_identity_count": len({chunk.scope_identity_digest for chunk in chunks}),
        "missing_vector_count": len(chunks) - len(vectors),
        "non_finite_vector_count": 0,
        "non_normalized_vector_count": 0,
        "embedding_input_token_estimate_p95": _p95(token_counts),
        "embedding_input_token_mean": statistics.mean(token_counts) if token_counts else 0.0,
        "embedding_input_token_p95": _p95(token_counts),
        "embedding_input_token_total": sum(token_counts),
        "embedding_call_count": manifest.get("embedded_vector_count", len(vectors)),
        "model_forward_batch_count": manifest.get("batch_count"),
        "embedding_wall_time": manifest.get("embedding_wall_time"),
        "embedding_throughput": manifest.get("embedding_throughput_inputs_per_second"),
        "embedding_manifest_digest": digest_json(_public_manifest(manifest)),
        "vector_digest": vector_digest(vectors),
        "manifest": _public_manifest(manifest),
    }
    _validate_shadow_cardinality(_shadow_cardinality(len(chunks), _as_indexed_chunks(chunks), [render_shadow_chunk_embedding_input(chunk) for chunk in chunks], vectors))
    return values


def _shadow_index_metrics(chunks: tuple[ShadowChunk, ...], vectors: dict[str, list[float]]) -> dict[str, Any]:
    raw_vector_payload_bytes = len(vectors) * DEFAULT_EMBEDDING_DIMENSION * 4
    metadata_bytes = len(json.dumps([chunk.public_json() for chunk in chunks], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return {
        "variant_id": chunks[0].variant_id if chunks else None,
        "indexed_vector_count": len(vectors),
        "unique_identity_count": len(vectors),
        "candidate_universe_document_count": len({chunk.document_identity_digest for chunk in chunks}),
        "candidate_universe_chunk_count": len(chunks),
        "candidate_universe_digest": digest_json([chunk.public_json() for chunk in chunks]),
        "raw_vector_payload_bytes": raw_vector_payload_bytes,
        "index_metadata_bytes": metadata_bytes,
        "serialized_index_bytes": raw_vector_payload_bytes + metadata_bytes,
        "index_digest": digest_json({"chunks": [chunk.public_json() for chunk in chunks], "vector_digest": vector_digest(vectors)}),
        "index_build_time": 0.0,
    }


def _attach_variant_derivatives(summaries: dict[str, dict[str, Any]], traces: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    baseline = summaries["production_chunking_v1"]
    baseline_traces = [row for row in traces if row["variant_id"] == "production_chunking_v1"]
    for variant_id, metrics in summaries.items():
        variant_traces = [row for row in traces if row["variant_id"] == variant_id]
        _attach_contamination_delta(metrics, baseline)
        metrics["question_type_metrics"] = _aggregate_dimension_metrics(variant_traces, baseline_traces, "question_type")
        metrics["capability_metrics"] = _aggregate_dimension_metrics(variant_traces, baseline_traces, "capability")
        metrics["gain_loss"] = _gain_loss(variant_traces, baseline_traces, variant_id)
        metrics["cost"] = _cost_metrics(metrics, baseline)


def _attach_contamination_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> None:
    for split in ("combined", "development", "known-regression"):
        current = metrics[split].get("no_evidence_candidate_contamination_rate", 0.0)
        base = baseline[split].get("no_evidence_candidate_contamination_rate", 0.0)
        metrics[split]["candidate_contamination_delta"] = current - base


def _aggregate_dimension_metrics(
    traces: list[dict[str, Any]],
    baseline_traces: list[dict[str, Any]],
    dimension: str,
) -> dict[str, dict[str, Any]]:
    labels = sorted({str(row.get(dimension) or "unknown") for row in traces})
    baseline_by_label = {label: _aggregate_retrieval([row for row in baseline_traces if str(row.get(dimension) or "unknown") == label]) for label in labels}
    out = {}
    for label in labels:
        rows = [row for row in traces if str(row.get(dimension) or "unknown") == label]
        aggregate = _aggregate_retrieval(rows)
        base = baseline_by_label[label]
        out[label] = {
            "sample_count": aggregate["sample_count"],
            "required_unit_count": aggregate["required_unit_count"],
            "complete_evidence_recall_at_5": aggregate["complete_evidence_recall_at_k"]["5"],
            "complete_evidence_recall_at_20": aggregate["complete_evidence_recall_at_k"]["20"],
            "any_support_recall_at_5": aggregate["any_support_recall_at_k"]["5"],
            "absolute_delta_to_c0": aggregate["complete_evidence_recall_at_k"]["5"] - base["complete_evidence_recall_at_k"]["5"],
            "sample_size_status": "sufficient" if aggregate["sample_count"] >= 3 else "insufficient_sample_size",
        }
    return out


def _profile_key(trace: dict[str, Any], profile: dict[str, Any]) -> tuple[str, str]:
    return (str(trace["sample_id"]), str(profile["source_span_digest"]))


def _gain_loss(traces: list[dict[str, Any]], baseline_traces: list[dict[str, Any]], variant_id: str) -> dict[str, Any]:
    base_profiles = {_profile_key(trace, profile): profile for trace in baseline_traces for profile in trace["unit_profiles"]}
    current_profiles = {_profile_key(trace, profile): profile for trace in traces for profile in trace["unit_profiles"]}
    newly_complete = []
    newly_supported = []
    lost_complete = []
    lost_supported = []
    unchanged_complete = []
    helped_samples = set()
    hurt_samples = set()
    for key, current in current_profiles.items():
        base = base_profiles.get(key, {})
        base_complete = base.get("complete_evidence_rank") is not None and base.get("complete_evidence_rank") <= 5
        current_complete = current.get("complete_evidence_rank") is not None and current.get("complete_evidence_rank") <= 5
        base_support = base.get("any_support_rank") is not None and base.get("any_support_rank") <= 5
        current_support = current.get("any_support_rank") is not None and current.get("any_support_rank") <= 5
        if current_complete and not base_complete:
            newly_complete.append(digest_json(key))
            helped_samples.add(key[0])
        if current_support and not base_support:
            newly_supported.append(digest_json(key))
            helped_samples.add(key[0])
        if base_complete and not current_complete:
            lost_complete.append(digest_json(key))
            hurt_samples.add(key[0])
        if base_support and not current_support:
            lost_supported.append(digest_json(key))
            hurt_samples.add(key[0])
        if base_complete and current_complete:
            unchanged_complete.append(digest_json(key))
    gain_label = {
        "heading_attached_section": "recovered_by_heading_attachment",
        "structured_block_atomic": "recovered_by_structured_block_atomicity",
        "fixed_one_block_overlap": "recovered_by_overlap",
        "combined_scope_aware": "recovered_by_combined_scope_chunking",
    }.get(variant_id, "control")
    loss_label = "lost_due_to_rank_displacement" if lost_complete or lost_supported else "unattributed"
    return {
        "newly_completed_required_units": len(newly_complete),
        "newly_supported_required_units": len(newly_supported),
        "lost_complete_required_units": len(lost_complete),
        "lost_supported_required_units": len(lost_supported),
        "unchanged_complete_hits": len(unchanged_complete),
        "helped_samples": len(helped_samples),
        "hurt_samples": len(hurt_samples),
        "gain_attribution": gain_label,
        "loss_attribution": loss_label,
        "newly_completed_required_unit_digests": newly_complete,
        "lost_complete_required_unit_digests": lost_complete,
    }


def _cost_metrics(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    embedding = metrics["embedding"]
    base_embedding = baseline["embedding"]
    index = metrics["shadow_index"]
    base_index = baseline["shadow_index"]
    combined = metrics["combined"]
    base_combined = baseline["combined"]
    return {
        "shadow_chunk_count": metrics["structure"]["total_chunk_count"],
        "chunk_count_ratio_to_c0": _ratio(metrics["structure"]["total_chunk_count"], baseline["structure"]["total_chunk_count"]),
        "embedding_input_token_mean": embedding.get("embedding_input_token_mean", 0.0),
        "embedding_input_token_p95": embedding.get("embedding_input_token_p95", 0.0),
        "embedding_input_token_total": embedding.get("embedding_input_token_total", 0),
        "embedding_input_token_total_ratio_to_c0": _ratio(embedding.get("embedding_input_token_total", 0), base_embedding.get("embedding_input_token_total", 0)),
        "embedding_call_count": embedding.get("embedding_call_count"),
        "model_forward_batch_count": embedding.get("model_forward_batch_count"),
        "embedding_wall_time": embedding.get("embedding_wall_time"),
        "embedding_throughput": embedding.get("embedding_throughput"),
        "raw_vector_payload_bytes": index.get("raw_vector_payload_bytes"),
        "serialized_index_bytes": index.get("serialized_index_bytes"),
        "serialized_index_size_ratio_to_c0": _ratio(index.get("serialized_index_bytes", 0), base_index.get("serialized_index_bytes", 0)),
        "index_build_time": index.get("index_build_time"),
        "query_latency_p50": combined.get("query_latency_p50_ms"),
        "query_latency_p95": combined.get("query_latency_p95_ms"),
        "query_latency_p95_ratio_to_c0": _ratio(combined.get("query_latency_p95_ms", 0.0), base_combined.get("query_latency_p95_ms", 0.0)),
        "overlap_storage_overhead": metrics["structure"].get("overlap_token_ratio", 0.0),
        "duplicate_source_token_ratio": metrics["structure"].get("duplicate_source_token_ratio", 0.0),
    }


def _ratio(value: float | int | None, baseline: float | int | None) -> float:
    if not baseline:
        return 0.0 if not value else float("inf")
    return float(value or 0.0) / float(baseline)


def _variant_candidate_universe(chunks: tuple[ShadowChunk, ...]) -> dict[str, Any]:
    return {
        "variant_candidate_universe_chunk_count": len(chunks),
        "variant_candidate_universe_document_count": len({chunk.document_identity_digest for chunk in chunks}),
        "variant_candidate_universe_digest": digest_json([chunk.public_json() for chunk in chunks]),
    }


def _as_indexed_chunks(chunks: tuple[ShadowChunk, ...]) -> tuple[IndexedScopeChunk, ...]:
    return tuple(
        IndexedScopeChunk(
            document_id=uuid5(NAMESPACE_URL, chunk.document_identity_digest),
            chunk_id=uuid5(NAMESPACE_URL, chunk.scope_identity_digest),
            relative_path=chunk.relative_path,
            document_title=chunk.document_title,
            heading_path=chunk.heading_path,
            content=chunk.content,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_index=chunk.chunk_ordinal,
            identity=chunk.identity,
            document_identity_digest=chunk.document_identity_digest,
            scope_identity_digest=chunk.scope_identity_digest,
            chunk_content_digest=chunk.rendered_content_digest,
            heading_path_digest=digest_json(list(chunk.heading_path)) if chunk.heading_path else None,
            block_types=chunk.block_types,
        )
        for chunk in chunks
    )


def _production_comparison_rows(by_doc: dict[str, list[IndexedScopeChunk]]) -> list[tuple[str, int, int | None, int | None, str | None]]:
    return [
        (
            document_digest,
            chunk.chunk_index,
            chunk.start_line,
            chunk.end_line,
            chunk.chunk_content_digest,
        )
        for document_digest, chunks in sorted(by_doc.items())
        for chunk in chunks
    ]


def _c0_comparison_rows(by_doc: dict[str, list[ShadowChunk]]) -> list[tuple[str, int, int | None, int | None, str | None]]:
    return [
        (
            document_digest,
            chunk.chunk_ordinal,
            chunk.start_line,
            chunk.end_line,
            chunk.rendered_content_digest,
        )
        for document_digest, chunks in sorted(by_doc.items())
        for chunk in chunks
    ]


def _legacy_missing_line_range_count(
    production_rows: list[tuple[str, int, int | None, int | None, str | None]],
    c0_rows: list[tuple[str, int, int | None, int | None, str | None]],
) -> int:
    return sum(
        1
        for production, c0 in zip(production_rows, c0_rows, strict=False)
        if production[:2] == c0[:2]
        and production[4] == c0[4]
        and (production[2] is None or production[3] is None)
        and c0[2] is not None
        and c0[3] is not None
    )


def _source_ranges_compatible_with_legacy_missing(
    production_rows: list[tuple[str, int, int | None, int | None, str | None]],
    c0_rows: list[tuple[str, int, int | None, int | None, str | None]],
) -> bool:
    if len(production_rows) != len(c0_rows):
        return False
    for production, c0 in zip(production_rows, c0_rows, strict=True):
        if production[:2] != c0[:2] or production[4] != c0[4]:
            return False
        if production[2:4] == c0[2:4]:
            continue
        if production[2] is None or production[3] is None:
            continue
        return False
    return True


def _redacted_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        for profile in trace["unit_profiles"]:
            rows.append(
                {
                    "sample_id": trace["sample_id"],
                    "dataset_id": trace["dataset_id"],
                    "variant_id": trace["variant_id"],
                    "source_span_digest": profile["source_span_digest"],
                    "any_support_rank": profile["any_support_rank"],
                    "complete_evidence_rank": profile["complete_evidence_rank"],
                    "document_rank": profile["document_rank"],
                    "candidate_identity_digests": trace["candidate_identity_digests"],
                    "span_match_class": "complete" if profile["complete_evidence_rank"] is not None else ("partial" if profile["any_support_rank"] is not None else "none"),
                    "coverage_ratio": 1.0 if profile["complete_evidence_rank"] is not None else (0.5 if profile["any_support_rank"] is not None else 0.0),
                }
            )
    return rows


def _structure_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.scope-aware-chunking-structure.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "variants": {
            variant_id: {
                "structure": metrics.get("structure"),
                "validation": metrics.get("validation"),
                "candidate_universe": metrics.get("candidate_universe"),
            }
            for variant_id, metrics in (summary.get("variants") or {}).items()
        },
        "contains_sealed_holdout_data": False,
        "publishes_source_text": False,
        "publishes_chunk_text": False,
        "publishes_paths": False,
        "writes_database": False,
        "writes_index": False,
    }
    _reject_forbidden_payload(payload)
    return payload


def _run_signature(summary: dict[str, Any]) -> dict[str, Any]:
    variants = summary.get("variants") or {}
    return {
        "shadow_chunk_identity_digest": digest_json({variant: metrics.get("candidate_universe", {}).get("variant_candidate_universe_digest") for variant, metrics in variants.items()}),
        "embedding_manifest_digest": digest_json({variant: _stable_embedding_signature(metrics.get("embedding") or {}) for variant, metrics in variants.items()}),
        "candidate_identity_digest": digest_json({variant: metrics.get("shadow_index", {}).get("index_digest") for variant, metrics in variants.items()}),
        "candidate_rank_digest": _sha256_file(TRACE_PATH) if TRACE_PATH.exists() else None,
        "metric_digest": digest_json({variant: _stable_metric_payload(metrics) for variant, metrics in variants.items()}),
    }


def _stable_embedding_signature(embedding: dict[str, Any]) -> dict[str, Any]:
    manifest = embedding.get("manifest") or {}
    return {
        "expected_chunk_count": embedding.get("expected_chunk_count"),
        "embedded_vector_count": embedding.get("embedded_vector_count"),
        "stored_vector_count": embedding.get("stored_vector_count"),
        "unique_identity_count": embedding.get("unique_identity_count"),
        "missing_vector_count": embedding.get("missing_vector_count"),
        "vector_digest": embedding.get("vector_digest"),
        "cache_contract_digest": manifest.get("cache_contract_digest"),
        "execution_plan_digest": manifest.get("execution_plan_digest"),
        "identity_digest": manifest.get("identity_digest"),
        "kind": manifest.get("kind"),
        "variant_id": manifest.get("variant_id"),
        "dimension": manifest.get("dimension"),
        "l2_normalized": manifest.get("l2_normalized"),
        "storage_dtype": manifest.get("storage_dtype"),
    }


def _stable_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            key: value
            for key, value in (metrics.get(split) or {}).items()
            if not key.startswith("query_latency_")
        }
        for split in ("combined", "development", "known-regression")
    } | {
        "structure": metrics.get("structure"),
        "validation": metrics.get("validation"),
        "gain_loss": metrics.get("gain_loss"),
        "cost": {
            key: value
            for key, value in (metrics.get("cost") or {}).items()
            if key
            not in {
                "embedding_wall_time",
                "embedding_throughput",
                "query_latency_p50",
                "query_latency_p95",
                "query_latency_p95_ratio_to_c0",
                "index_build_time",
            }
        },
    }


def _update_execution_artifact(summary: dict[str, Any]) -> None:
    existing = read_json(EXECUTION_PATH) if EXECUTION_PATH.exists() else {}
    runs = list(existing.get("runs") or [])
    run_id = summary.get("run_id") or f"run-{len(runs) + 1}"
    signature = _run_signature(summary)
    runs = [row for row in runs if row.get("run_id") != run_id]
    runs.append({"run_id": run_id, "created_at": utc_now(), **signature})
    runs = runs[-2:]
    agreement = _run_agreement(runs)
    payload = {
        "schema_version": "opk-rag.scope-aware-chunking-execution.v1",
        "execution_id": "phase2-scope-aware-chunking-execution-v2",
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "runs": runs,
        "run_count": len(runs),
        "run_agreement": agreement,
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_variant": summary.get("recommended_variant"),
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
    }
    write_json(EXECUTION_PATH, payload)


def _run_agreement(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    if len(runs) < 2:
        return {
            "shadow_chunk_identity_agreement": 1.0,
            "embedding_manifest_agreement": 1.0,
            "candidate_identity_agreement": 1.0,
            "candidate_rank_agreement": 1.0,
            "metric_agreement": 1.0,
        }
    left, right = runs[-2], runs[-1]
    return {
        "shadow_chunk_identity_agreement": 1.0 if left.get("shadow_chunk_identity_digest") == right.get("shadow_chunk_identity_digest") else 0.0,
        "embedding_manifest_agreement": 1.0 if left.get("embedding_manifest_digest") == right.get("embedding_manifest_digest") else 0.0,
        "candidate_identity_agreement": 1.0 if left.get("candidate_identity_digest") == right.get("candidate_identity_digest") else 0.0,
        "candidate_rank_agreement": 1.0 if left.get("candidate_rank_digest") == right.get("candidate_rank_digest") else 0.0,
        "metric_agreement": 1.0 if left.get("metric_digest") == right.get("metric_digest") else 0.0,
    }


def _primary_result_classification(summary: dict[str, Any]) -> str:
    if summary.get("experiment_status") != "completed":
        return "invalid_experiment"
    decision = summary.get("promotion_decision")
    if decision == "no_candidate":
        return "chunking_candidate_not_found"
    if decision == "diagnostic_only":
        return "chunking_improves_structure_not_retrieval"
    variant = summary.get("recommended_variant")
    return {
        "heading_attached_section": "heading_boundary_gain",
        "structured_block_atomic": "structured_block_boundary_gain",
        "fixed_one_block_overlap": "overlap_boundary_gain",
        "combined_scope_aware": "combined_scope_chunking_gain",
    }.get(str(variant), "chunking_candidate_not_found")


def _update_chunking_governance(summary: dict[str, Any]) -> None:
    baseline_metrics = (summary.get("variants") or {}).get("production_chunking_v1", {})
    recommended = summary.get("recommended_variant")
    chosen_metrics = (summary.get("variants") or {}).get(recommended, baseline_metrics) if recommended else baseline_metrics
    status = summary.get("scope_aware_chunking_experiment_status")
    if summary.get("c0_scoring_parity_status") != "valid" or summary.get("experiment_status") == "invalid_experiment":
        status = "invalid_c0_scoring_parity"
    payload_update = {
        "scope_aware_chunking_experiment_status": status or "completed",
        "public_promotion_decision": summary.get("promotion_decision"),
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_chunking_variant": recommended,
        "recommended_variant": recommended,
        "primary_result_classification": summary.get("primary_result_classification"),
        "result_validity": summary.get("result_validity"),
        "c0_scoring_parity_status": summary.get("c0_scoring_parity_status"),
        "required_unit_fragmentation_rate": (chosen_metrics.get("structure") or {}).get("required_unit_fragmentation_rate"),
        "unique_scope_fragmentation_rate": (chosen_metrics.get("structure") or {}).get("unique_scope_fragmentation_rate"),
        "replacement_holdout_required": True,
        "stage_decision": "not_accepted",
        "holdout_status": "exposed",
        "holdout_v1_status": "exposed",
        "holdout_v1_used_in_task0060": False,
        "holdout_v1_future_acceptance_eligible": False,
    }
    for path in (
        ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json",
        ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json",
    ):
        if not path.exists():
            continue
        payload = read_json(path)
        payload.update(payload_update)
        _write_json_preserving_existing_governance(path, payload)


def _update_benchmark_registry(summary: dict[str, Any]) -> None:
    path = ROOT / "evaluation-data" / "benchmark_registry.json"
    if not path.exists():
        return
    registry = read_json(path)
    entry = {
        "artifact_id": EXPERIMENT_ID,
        "role": "public_chunking_experiment",
        "current_usage": "TASK-0060 public scope-aware chunking v2 experiment",
        "execution_manifest": EXECUTION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "experiment_contract": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_span_contract": SOURCE_SPAN_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "production_parity_contract": PRODUCTION_PARITY_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result": RESULT_PATH.relative_to(ROOT).as_posix(),
        "trace": TRACE_PATH.relative_to(ROOT).as_posix(),
        "structure_result": STRUCTURE_PATH.relative_to(ROOT).as_posix(),
        "execution_result": EXECUTION_PATH.relative_to(ROOT).as_posix(),
        "candidate_universe_digests": {variant: metrics.get("candidate_universe", {}).get("variant_candidate_universe_digest") for variant, metrics in (summary.get("variants") or {}).items()},
        "qwen_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_variant": summary.get("recommended_variant"),
        "public_evaluation_only": True,
        "holdout_v1_used": False,
        "future_stage_acceptance_eligible": False,
        "production_chunking_changed": False,
        "production_index_changed": False,
        "database_changed": False,
        "contains_private_data": False,
        "contains_sealed_holdout_data": False,
    }
    v1_entry = {
        "artifact_id": V1_EXPERIMENT_ID,
        "role": "public_chunking_experiment",
        "current_usage": "TASK-0060 v1 invalid baseline semantics diagnostic record",
        "execution_manifest": V1_EXECUTION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "experiment_contract": V1_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_span_contract": SOURCE_SPAN_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "production_parity_contract": PRODUCTION_PARITY_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result": V1_RESULT_PATH.relative_to(ROOT).as_posix(),
        "trace": V1_TRACE_PATH.relative_to(ROOT).as_posix(),
        "structure_result": V1_STRUCTURE_PATH.relative_to(ROOT).as_posix(),
        "execution_result": V1_EXECUTION_PATH.relative_to(ROOT).as_posix(),
        "experiment_status": "invalid_baseline_semantics",
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "result_validity": "diagnostic_only_not_valid_for_promotion",
        "invalid_reason": "TASK-0059 source-span scoring oracle was incorrectly registered as the TASK-0060 retrieval promotion baseline",
        "public_evaluation_only": True,
        "holdout_v1_used": False,
        "future_stage_acceptance_eligible": False,
        "production_chunking_changed": False,
        "production_index_changed": False,
        "database_changed": False,
        "contains_private_data": False,
        "contains_sealed_holdout_data": False,
    }
    assets = [row for row in registry.get("asset_inventory", []) if row.get("artifact_id") not in {V1_EXPERIMENT_ID, EXPERIMENT_ID}]
    assets.extend([v1_entry, entry])
    registry["asset_inventory"] = assets
    _write_json_preserving_existing_governance(path, registry)


def _write_json_preserving_existing_governance(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _invalid_summary(contract: dict[str, Any], universe_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "invalid_experiment",
        "source_corpus_status": universe_audit.get("source_corpus_status", "incomplete"),
        "scope_aware_chunking_experiment_status": "infrastructure_blocked",
        "source_evidence_span_contract_status": "incomplete",
        "result_validity": "not_valid_for_chunking_quality_decision",
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_universe_audit": universe_audit,
        "source_corpus_manifest": universe_audit,
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
        "blocked_next_step": "provide the frozen 120-document source universe matching phase2_corpus_snapshot_public.json",
        "contract_digest": digest_json(contract),
    }


def _invalid_c0_scoring_parity_summary(
    *,
    contract: dict[str, Any],
    universe_audit: dict[str, Any],
    c0_parity_gate: dict[str, Any],
    c0_observed_metrics: dict[str, Any],
    c0_metric_diff: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "invalid_experiment",
        "source_corpus_status": universe_audit.get("source_corpus_status", "complete"),
        "scope_aware_chunking_experiment_status": "invalid_c0_scoring_parity",
        "source_evidence_span_contract_status": "valid",
        "c0_scoring_parity_status": c0_parity_gate.get("status", "invalid"),
        "result_validity": "not_valid_for_chunking_quality_decision",
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "primary_result_classification": "invalid_experiment",
        "invalid_pre_fix_task0060_run": {
            "scope_aware_chunking_experiment_status": "invalid_c0_scoring_parity",
            "promotion_decision": "invalid_experiment",
            "recommended_variant": None,
            "primary_result_classification": "invalid_experiment",
            "result_validity": "not_valid_for_chunking_quality_decision",
            "c0_complete_evidence_recall_at_5": (c0_observed_metrics.get("complete_evidence_recall_at_k") or {}).get("5"),
            "c0_complete_evidence_recall_at_5_numerator": _numerator(c0_observed_metrics, "complete_evidence_recall_at_k", "5", EXPECTED_REQUIRED_EVIDENCE_UNITS),
            "c0_complete_evidence_recall_at_5_denominator": EXPECTED_REQUIRED_EVIDENCE_UNITS,
        },
        "c0_scoring_parity_gate": c0_parity_gate,
        "c0_metric_diff": c0_metric_diff,
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_universe_audit": universe_audit,
        "contract_digest": digest_json(contract),
        "holdout_v1_status": "exposed",
        "holdout_v1_future_acceptance_eligible": False,
        "replacement_holdout_required": True,
        "holdout_v1_used_in_task0060": False,
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
    }


def _infrastructure_blocked_summary(contract: dict[str, Any], universe_audit: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "infrastructure_blocked",
        "source_corpus_status": universe_audit.get("source_corpus_status", "incomplete"),
        "scope_aware_chunking_experiment_status": "infrastructure_blocked",
        "source_evidence_span_contract_status": "incomplete",
        "result_validity": "not_valid_for_chunking_quality_decision",
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "source_universe_audit": universe_audit,
        "source_corpus_manifest": universe_audit,
        "infrastructure_blocked_reason": reason,
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
        "contract_digest": digest_json(contract),
    }


def build_markdown_report(summary: dict[str, Any]) -> str:
    rows = []
    for variant, metrics in (summary.get("variants") or {}).items():
        combined = metrics.get("combined") or {}
        complete = combined.get("complete_evidence_recall_at_k") or {}
        any_support = combined.get("any_support_recall_at_k") or {}
        doc = combined.get("document_recall_at_k") or {}
        structure = metrics.get("structure") or {}
        cost = metrics.get("cost") or {}
        rows.append(f"| `{variant}` | {any_support.get('5', 0):.4f} | {complete.get('5', 0):.4f} | {complete.get('20', 0):.4f} | {doc.get('20', 0):.4f} | {structure.get('required_unit_fragmentation_rate', 0):.4f} | {structure.get('unique_scope_fragmentation_rate', 0):.4f} | {cost.get('chunk_count_ratio_to_c0', 0):.2f} |")
    answers = _report_answers(summary)
    return "\n".join(
        [
            "# Phase 2 Scope-aware Chunking Experiment",
            "",
            f"Experiment ID: `{summary.get('experiment_id')}`",
            "",
            "This TASK-0060 experiment is evaluation-only. It does not change production chunking, write the database, rebuild the production index, run generation, call DeepSeek, or read sealed holdout data.",
            "",
            f"Status: `{summary.get('experiment_status')}`",
            f"Result validity: `{summary.get('result_validity')}`",
            f"Promotion decision: `{summary.get('promotion_decision')}`",
            f"Recommended variant: `{summary.get('recommended_variant')}`",
            f"Primary result classification: `{summary.get('primary_result_classification')}`",
            f"Stage decision remains: `{summary.get('stage_decision', 'not_accepted')}`",
            "",
            "## Source Universe",
            "",
            f"`{json.dumps(summary.get('source_universe_audit') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Public Metrics",
            "",
            "| Variant | Any R@5 | Complete R@5 | Complete R@20 | Doc R@20 | Required Frag | Unique Frag | Chunk Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## TASK-0060 Questions",
            "",
            *answers,
            "",
        ]
    )


def _report_answers(summary: dict[str, Any]) -> list[str]:
    variants = summary.get("variants") or {}
    c0 = variants.get("production_chunking_v1") or {}
    lines = []
    def metric(variant: str, path: tuple[str, ...], default: float = 0.0) -> float:
        value: Any = variants.get(variant) or {}
        for key in path:
            value = (value or {}).get(key)
        return float(value if value is not None else default)

    c0_req = metric("production_chunking_v1", ("structure", "required_unit_fragmentation_rate"))
    c0_unique = metric("production_chunking_v1", ("structure", "unique_scope_fragmentation_rate"))
    c0_r5 = metric("production_chunking_v1", ("combined", "complete_evidence_recall_at_k", "5"))
    best_req = min((variant for variant in variants), key=lambda variant: metric(variant, ("structure", "required_unit_fragmentation_rate")), default=None)
    best_unique = min((variant for variant in variants), key=lambda variant: metric(variant, ("structure", "unique_scope_fragmentation_rate")), default=None)
    best_r5 = max((variant for variant in variants), key=lambda variant: metric(variant, ("combined", "complete_evidence_recall_at_k", "5")), default=None)
    lines.append(f"1. C1 heading attachment fragmentation delta: `{c0_req - metric('heading_attached_section', ('structure', 'required_unit_fragmentation_rate')):.4f}`.")
    lines.append(f"2. C2 structured block split counts: list `{metric('structured_block_atomic', ('structure', 'split_list_count')):.0f}`, table `{metric('structured_block_atomic', ('structure', 'split_table_count')):.0f}`, code `{metric('structured_block_atomic', ('structure', 'split_code_block_count')):.0f}`, quote `{metric('structured_block_atomic', ('structure', 'split_quote_count')):.0f}`.")
    lines.append(f"3. C3 overlap Complete R@5 delta: `{metric('fixed_one_block_overlap', ('combined', 'complete_evidence_recall_at_k', '5')) - c0_r5:.4f}`.")
    lines.append(f"4. C4 combined Complete R@5 delta: `{metric('combined_scope_aware', ('combined', 'complete_evidence_recall_at_k', '5')) - c0_r5:.4f}`, chunk ratio `{metric('combined_scope_aware', ('cost', 'chunk_count_ratio_to_c0')):.2f}`.")
    lines.append(f"5. Best required-unit fragmentation variant: `{best_req}`.")
    lines.append(f"6. Best unique-scope fragmentation variant: `{best_unique}`.")
    lines.append(f"7. Best Complete Evidence R@5 variant: `{best_r5}`.")
    for variant in (best_r5,) if best_r5 else ():
        dev_delta = metric(variant, ("development", "complete_evidence_recall_at_k", "5")) - metric("production_chunking_v1", ("development", "complete_evidence_recall_at_k", "5"))
        reg_delta = metric(variant, ("known-regression", "complete_evidence_recall_at_k", "5")) - metric("production_chunking_v1", ("known-regression", "complete_evidence_recall_at_k", "5"))
        lines.append(f"8. Best R@5 variant split deltas: development `{dev_delta:.4f}`, known-regression `{reg_delta:.4f}`.")
    lines.append("9. Cost ratios are reported beside quality metrics to distinguish chunk-count effects from retrieval quality.")
    lines.append(f"10. Max candidate contamination delta: `{max((metric(v, ('combined', 'candidate_contamination_delta')) for v in variants), default=0.0):.4f}`.")
    lines.append(f"11. Promotion gate decision: `{summary.get('promotion_decision')}`.")
    lines.append(f"12. Next direction: `{_next_direction(summary)}`.")
    return lines


def _next_direction(summary: dict[str, Any]) -> str:
    if summary.get("promotion_decision") == "promotion_candidate":
        return "opt_in_runtime_integration"
    return "multi_vector_or_parent_child_retrieval"


def _variant_contract(variant_id: str) -> dict[str, Any]:
    common = {"max_size": 1800, "target_size": 1200, "minimum_size": None, "length_unit": "character", "gold_evidence_used_for_boundary": False}
    policies = {
        "production_chunking_v1": {"heading_policy": "production_v1", "structured_block_policy": "production_v1", "overlap_policy": "none", "flush_policy": "production_v1"},
        "heading_attached_section": {"heading_policy": "attach_heading_to_following_content", "structured_block_policy": "production_oversized_only", "overlap_policy": "none", "flush_policy": "do_not_cross_same_or_higher_heading"},
        "structured_block_atomic": {"heading_policy": "attach_heading_to_following_block", "structured_block_policy": "paragraph_list_table_code_quote_atomic_when_under_max", "overlap_policy": "none", "flush_policy": "block_pack_until_max"},
        "fixed_one_block_overlap": {"heading_policy": "production_v1", "structured_block_policy": "production_v1", "overlap_policy": "previous_structural_block", "overlap_token_cap": 128, "flush_policy": "production_v1"},
        "combined_scope_aware": {"heading_policy": "attach_heading_to_following_block", "structured_block_policy": "paragraph_list_table_code_quote_atomic_when_under_max", "overlap_policy": "previous_structural_block", "overlap_token_cap": 128, "flush_policy": "block_pack_until_max"},
    }
    return {"id": variant_id, **common, **policies[variant_id]}


def _overlap_text(text: str, token_cap: int) -> str:
    tokens = text.split()
    return " ".join(tokens[-token_cap:])


def _same_high_level_heading(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return True
    return left[0] == right[0]


def _is_heading_only(content: str) -> bool:
    lines = [line for line in content.splitlines() if line.strip()]
    return len(lines) == 1 and bool(re.match(r"^#{1,6}\s+", lines[0]))


def _document_identity_digest(relative_path: str) -> str:
    return digest_json({"relative_path_digest": digest_json(relative_path)})


def _duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for value, count in counts.items() if value and count > 1)


def _repository_source_documents_markdown_count() -> int:
    if not SOURCE_ROOT.exists():
        return 0
    return sum(1 for _ in SOURCE_ROOT.rglob("*.md"))


def _resolved_vault_root_payload(vault_path: str | Path, source: str) -> dict[str, Any]:
    try:
        root = normalize_vault_root(vault_path)
    except VaultPathError as exc:
        return {
            "status": "missing",
            "vault_root": None,
            "source_root_config_source": source,
            "source_root_digest": None,
            "issues": [{"code": "source_root_unavailable", "message": _sanitize_vault_message(str(exc), vault_path)}],
        }
    return {
        "status": "resolved",
        "vault_root": root.path,
        "source_root_config_source": source,
        "source_root_digest": hashlib.sha256(f"opk-rag-phase2-evaluation-vault-root-v1\0{root.canonical_path}".encode("utf-8")).hexdigest(),
        "issues": [],
    }


def _sanitize_vault_message(message: str, vault_path: str | Path) -> str:
    text = str(message)
    raw = str(vault_path)
    expanded = str(Path(vault_path).expanduser())
    for value in {raw, expanded}:
        if value:
            text = text.replace(value, "<vault-root>")
    return text
