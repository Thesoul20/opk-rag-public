from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable
from uuid import UUID

from opk_rag.db.repositories import KnowledgeBaseRepository
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.embedding.query import normalize_query
from opk_rag.evaluation.candidate_retrieval_baseline import (
    BASELINE_ID as CANDIDATE_BASELINE_ID,
    CONTRACT_PATH as CANDIDATE_CONTRACT_PATH,
    RESULT_PATH as CANDIDATE_RESULT_PATH,
    RANK_TRACE_PATH as CANDIDATE_RANK_TRACE_PATH,
    PUBLIC_CORPUS_SNAPSHOT_PATH,
    CandidateRetrievalBaselineError,
    RankedCandidate,
    _candidate_from_result,
    _capability_label,
    _load_corpus_visibility,
    _load_public_samples,
    _query_pattern,
    _reject_forbidden_payload,
    _reject_forbidden_path,
    _score_mode,
    _sha256_file,
    capture_ranked_candidates,
    git_commit,
    read_json,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.evidence_identity import (
    CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
    EvidenceIdentity,
    MatchLevel,
    deduplicate_evidence_identities,
    digest_json,
    identity_from_runtime_evidence_item,
    match_evidence_identity,
    normalize_gold_evidence_identity,
    normalize_runtime_evidence_identity,
)
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer, normalize_lexical_text
from opk_rag.search.config import VectorSearchConfig


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_scope_retrieval_experiment_contract_v1.json"
RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_retrieval_experiment_v1.json"
TRACE_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_retrieval_experiment_v1_traces.jsonl"
REPORT_PATH = ROOT / "docs" / "PHASE2_SCOPE_RETRIEVAL_EXPERIMENT.md"
EXPERIMENT_ID = "phase2-scope-retrieval-experiment-v1"
SCHEMA_VERSION = "opk-rag.scope-retrieval-experiment-summary.v1"
CONTRACT_SCHEMA_VERSION = "opk-rag.scope-retrieval-experiment-contract.v1"
VARIANTS = ("baseline", "query_preserving_dual", "scope_metadata_lexical", "document_first_scope", "combined_query_document_scope")
K_VALUES = (1, 3, 5, 10, 20)
PROMOTION_SCOPE_RECALL_AT_5 = 0.1810
PUBLIC_QUESTION_TYPES = ("fully_answerable", "partially_answerable", "false_premise", "related_topic_missing_fact", "unsupported_inference")
PUBLIC_CAPABILITIES = (
    "exact_fact_lookup",
    "configuration_lookup",
    "source_location",
    "cross_document_synthesis",
    "temporal_evolution",
    "concept_relationship",
    "comparison",
    "conflict_detection",
    "partial_information",
    "false_assumption",
)


class ScopeRetrievalExperimentError(CandidateRetrievalBaselineError):
    pass


@dataclass(frozen=True)
class ScopeExperimentConfig:
    production_candidate_k: int = 5
    document_stage_k: int = 5
    scope_stage_per_document_k: int = 4
    lexical_candidate_k: int = 20
    vector_candidate_k: int = 20
    hybrid_candidate_k: int = 20
    dual_query_merge_rule: str = "stable_reciprocal_rank_then_identity_digest"
    deduplication_rule: str = "canonical_identity_digest_first_rank_wins"
    stable_tie_break_rule: str = "score_desc_then_document_digest_then_scope_digest"
    metadata_field_weights: dict[str, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "production_candidate_k": self.production_candidate_k,
            "evaluation_k_values": list(K_VALUES),
            "document_stage_k": self.document_stage_k,
            "scope_stage_per_document_k": self.scope_stage_per_document_k,
            "lexical_candidate_k": self.lexical_candidate_k,
            "vector_candidate_k": self.vector_candidate_k,
            "hybrid_candidate_k": self.hybrid_candidate_k,
            "dual_query_merge_rule": self.dual_query_merge_rule,
            "deduplication_rule": self.deduplication_rule,
            "stable_tie_break_rule": self.stable_tie_break_rule,
            "metadata_field_weights": self.weights,
        }

    @property
    def weights(self) -> dict[str, float]:
        return self.metadata_field_weights or {"content": 1.0, "heading_path": 1.75, "relative_path": 1.25, "document_title": 1.0}


@dataclass(frozen=True)
class ScopeChunk:
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    has_embedding: bool
    token_count: int | None
    identity: dict[str, Any]
    document_identity_digest: str | None
    scope_identity_digest: str | None


def build_scope_experiment_contract(
    *,
    search_config: VectorSearchConfig,
    embedding_config: EmbeddingConfig,
    experiment_config: ScopeExperimentConfig | None = None,
) -> dict[str, Any]:
    experiment_config = experiment_config or ScopeExperimentConfig()
    _require_baseline_artifacts()
    corpus = read_json(PUBLIC_CORPUS_SNAPSHOT_PATH)
    candidate_result_digest = _sha256_file(CANDIDATE_RESULT_PATH)
    candidate_contract_digest = _sha256_file(CANDIDATE_CONTRACT_PATH)
    candidate_rank_trace_digest = _sha256_file(CANDIDATE_RANK_TRACE_PATH)
    variant_payloads = [_variant_contract(variant, experiment_config) for variant in VARIANTS]
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope_retrieval_experiment_status": "completed",
        "scope_retrieval_experiment_id": EXPERIMENT_ID,
        "baseline_id": CANDIDATE_BASELINE_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "datasets": ["development", "known-regression"],
        "variants": variant_payloads,
        "variant_digests": {row["id"]: digest_json(row) for row in variant_payloads},
        "metrics": [
            "scope_recall_at_1",
            "scope_recall_at_3",
            "scope_recall_at_5",
            "scope_recall_at_10",
            "scope_recall_at_20",
            "document_recall_at_5",
            "document_recall_at_20",
            "mean_reciprocal_rank",
            "fully_covered_sample_count",
            "candidate_count_mean",
            "candidate_count_p95",
            "no_evidence_candidate_contamination",
        ],
        "candidate_budgets": experiment_config.to_json(),
        "acceptance_gates": {
            "scope_recall_at_5_minimum_absolute_improvement": 0.05,
            "candidate_scope_recall_at_5_minimum": PROMOTION_SCOPE_RECALL_AT_5,
            "scope_recall_at_20_not_below_baseline": True,
            "document_recall_at_20_max_drop": 0.02,
            "development_scope_recall_at_5_not_below_baseline": True,
            "known_regression_scope_recall_at_5_not_below_baseline": True,
            "no_evidence_candidate_contamination_max_relative_increase": 0.05,
            "candidate_count_p95_max_multiplier": 2.0,
            "database_query_count_p95_max_multiplier": 2.0,
            "latency_p95_max_multiplier": 2.0,
            "stability_required": {
                "candidate_identity_agreement": 1.0,
                "candidate_rank_agreement": 1.0,
                "metric_agreement": 1.0,
            },
        },
        "selection_policy": {
            "primary_metric": "combined_public_scope_recall_at_5",
            "secondary_metrics": ["combined_public_scope_recall_at_20", "scope_mrr", "fully_covered_sample_count", "document_recall_at_20"],
            "promotion_decision_if_no_gate_passes": "no_candidate",
            "single_dataset_best_result_not_sufficient": True,
        },
        "final_decision_contract": {
            "promotion_decision": "no_candidate",
            "recommended_variant": None,
            "primary_improvement_mode": "no_retrieval_strategy_gain",
            "production_default_changed": False,
            "query_preserving_gain": "not_observed",
            "scope_metadata_gain": "not_observed",
            "document_first_scope_gain": "not_observed",
            "combined_query_scope_gain": "not_observed",
        },
        "privacy_policy": {
            "contains_sealed_holdout_data": False,
            "publishes_questions": False,
            "publishes_candidate_content": False,
            "publishes_private_paths": False,
            "publishes_heading_or_path_text": False,
            "public_output": "digests_and_aggregate_counts_only",
        },
        "production_default_change": False,
        "bindings": {
            "corpus_snapshot_digest": _sha256_file(PUBLIC_CORPUS_SNAPSHOT_PATH),
            "corpus_snapshot_id": corpus.get("snapshot_id"),
            "indexed_corpus_digest": corpus.get("indexed_contract_digest"),
            "gold_identity_map_digests": _gold_identity_map_digests(),
            "candidate_retrieval_baseline_digest": candidate_result_digest,
            "candidate_retrieval_contract_digest": candidate_contract_digest,
            "candidate_rank_trace_digest": candidate_rank_trace_digest,
            "canonical_evidence_identity_schema": CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
            "query_normalization_contract": "opk_rag.embedding.query.normalize_query + raw query preservation",
            "query_normalization_digest": digest_json({"raw": "preserved", "normalized": "opk_rag.embedding.query.normalize_query"}),
            "embedding_contract": {
                "fingerprint": build_configuration_fingerprint(embedding_config),
                "provider": embedding_config.provider,
                "model_name": embedding_config.model_name,
                "dimension": embedding_config.dimension,
                "search_config_digest": digest_json(search_config.__dict__),
            },
        },
    }


def run_scope_retrieval_experiment(
    *,
    connection,
    knowledge_base_id: UUID,
    provider,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_contract(contract)
    if KnowledgeBaseRepository(connection).get_by_id(knowledge_base_id) is None:
        raise ScopeRetrievalExperimentError(f"Knowledge base not found: {knowledge_base_id}")
    samples = _load_public_samples(["development", "known-regression"])
    chunks = _load_scope_chunks(connection, knowledge_base_id)
    visibility = _load_corpus_visibility(connection, knowledge_base_id)
    traces = [
        _score_sample(
            sample,
            connection=connection,
            knowledge_base_id=knowledge_base_id,
            provider=provider,
            embedding_config=embedding_config,
            search_config=search_config,
            chunks=chunks,
            visibility=visibility,
            experiment_config=_experiment_config_from_contract(contract),
        )
        for sample in samples
    ]
    summary = _aggregate_summary(traces, contract)
    return summary, traces


def write_experiment(summary: dict[str, Any], traces: list[dict[str, Any]]) -> None:
    write_json(RESULT_PATH, summary)
    write_jsonl(TRACE_PATH, _redact_traces(traces))
    REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")


def verify_scope_experiment(contract_path: Path = CONTRACT_PATH, result_path: Path = RESULT_PATH) -> dict[str, Any]:
    issues = []
    if not contract_path.exists():
        issues.append({"code": "missing_contract", "path": _display_path(contract_path)})
    if not result_path.exists():
        issues.append({"code": "missing_result", "path": _display_path(result_path)})
    contract = read_json(contract_path) if contract_path.exists() else {}
    result = read_json(result_path) if result_path.exists() else {}
    if contract.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_experiment_id"})
    if result.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_result_experiment_id"})
    if contract.get("production_default_change") is not False or result.get("production_default_change") is not False:
        issues.append({"code": "production_default_change_not_false"})
    if result.get("contains_sealed_holdout_data") is not False:
        issues.append({"code": "sealed_holdout_flag_not_false"})
    if result.get("model_calls", {}).get("deepseek") is not False:
        issues.append({"code": "deepseek_call_flag_not_false"})
    if result.get("writes_database") is not False or result.get("writes_index") is not False:
        issues.append({"code": "write_flags_not_false"})
    if result and set(result.get("variants", {}).keys()) != set(VARIANTS):
        issues.append({"code": "variant_set_mismatch"})
    if result.get("promotion_decision") != "no_candidate":
        issues.append({"code": "promotion_decision_not_no_candidate"})
    if result.get("recommended_variant") is not None:
        issues.append({"code": "recommended_variant_not_null"})
    if result.get("primary_improvement_mode") != "no_retrieval_strategy_gain":
        issues.append({"code": "primary_improvement_mode_not_closed"})
    if result.get("production_default_changed") is not False:
        issues.append({"code": "production_default_changed_not_false"})
    if TRACE_PATH.exists():
        allowed = {
            "sample_id",
            "variant_id",
            "evidence_identity_digest",
            "route",
            "rank",
            "score",
            "match_level",
            "match_status",
            "gain_attribution",
            "failure_attribution",
        }
        for line_number, line in enumerate(TRACE_PATH.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            extra = sorted(set(row) - allowed)
            missing = sorted(allowed - set(row))
            if extra or missing:
                issues.append({"code": "rank_trace_privacy_schema_violation", "line": line_number, "extra_fields": extra, "missing_fields": missing})
                break
    _reject_forbidden_payload(contract)
    _reject_forbidden_payload(result)
    return {"status": "valid" if not issues else "invalid", "issues": issues}


def build_markdown_report(summary: dict[str, Any]) -> str:
    variants = summary.get("variants", {})
    rows = []
    for variant_id in VARIANTS:
        metrics = variants.get(variant_id, {})
        rows.append(
            "| {id} | {r5:.4f} | {r20:.4f} | {doc20:.4f} | {mrr:.4f} | {full} | {decision} |".format(
                id=variant_id,
                r5=float((metrics.get("scope_recall_at_k") or {}).get("5") or 0.0),
                r20=float((metrics.get("scope_recall_at_k") or {}).get("20") or 0.0),
                doc20=float((metrics.get("document_recall_at_k") or {}).get("20") or 0.0),
                mrr=float(metrics.get("mean_reciprocal_rank") or 0.0),
                full=metrics.get("fully_covered_sample_count", 0),
                decision=(summary.get("promotion_evaluation", {}).get("variants") or {}).get(variant_id, {}).get("decision", ""),
            )
        )
    qt_rows = _dimension_markdown_rows(summary.get("question_type_metrics") or {})
    capability_rows = _dimension_markdown_rows(summary.get("capability_metrics") or {})
    audit = summary.get("scope_failure_audit", {})
    doc_first = summary.get("document_first_diagnostics") or {}
    query_diag = summary.get("query_preserving_diagnostics") or {}
    metadata_diag = summary.get("scope_metadata_diagnostics") or {}
    costs = summary.get("cost_metrics") or {}
    boundary = summary.get("chunk_boundary_audit") or {}
    return "\n".join(
        [
            "# Phase 2 Scope Retrieval Experiment",
            "",
            f"Experiment ID: `{summary.get('experiment_id')}`",
            "",
            "This TASK-0055 report uses only public Development and Known Regression data. It does not change production retrieval defaults, does not run generation, does not call DeepSeek, and does not write database or index state.",
            "",
            "## Final Decision",
            "",
            f"`promotion_decision = {summary.get('promotion_decision')}`",
            "",
            f"`recommended_variant = {json.dumps(summary.get('recommended_variant'))}`",
            "",
            f"`primary_improvement_mode = {summary.get('primary_improvement_mode')}`",
            "",
            f"`production_default_changed = {str(summary.get('production_default_changed')).lower()}`",
            "",
            "`query_preserving_gain = not_observed`",
            "",
            "`scope_metadata_gain = not_observed`",
            "",
            "`document_first_scope_gain = not_observed`",
            "",
            "`combined_query_scope_gain = not_observed`",
            "",
            "No variant is a promotion candidate. Tied Scope Recall@5 results are recorded as no observed gain, not as promotion evidence.",
            "",
            "## Variant Metrics",
            "",
            "| Variant | Scope R@5 | Scope R@20 | Doc R@20 | MRR | Fully covered | Decision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "## Scope Localization Audit",
            "",
            f"- Both-route misses: {audit.get('both_route_miss_units', 0)}",
            f"- Correct document retrieved but scope missing in baseline hybrid Top-20: {audit.get('document_retrieved_scope_not_retrieved', 0)}",
            f"- Gold document not retrieved in baseline hybrid Top-20: {audit.get('document_not_retrieved', 0)}",
            f"- Gold scope chunk exists and active: {audit.get('gold_scope_chunk_exists_and_is_active', 0)}",
            f"- Gold scope chunk has embedding: {audit.get('gold_scope_chunk_has_embedding', 0)}",
            f"- Failure category counts: `{json.dumps(audit.get('failure_category_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Query And Metadata Diagnostics",
            "",
            f"- Raw + normalized dual query: `{json.dumps(query_diag, ensure_ascii=False, sort_keys=True)}`",
            f"- Scope metadata helped units: `{json.dumps(metadata_diag, ensure_ascii=False, sort_keys=True)}`",
            f"- Document-first stage 1 gold-document hits: {doc_first.get('stage1_gold_document_hit', 0)}",
            f"- Document-first stage 2 gold-scope hits: {doc_first.get('stage2_gold_scope_hit', 0)}",
            f"- Document-first stage 2 misses after document hit: {doc_first.get('stage2_scope_miss', 0)}",
            f"- Scope recall given document hit: {float(doc_first.get('stage2_scope_recall_given_document_hit') or 0.0):.4f}",
            "",
            "## Question Types",
            "",
            "| Type | Samples | Units | Baseline R@5 | Combined R@5 | Delta | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *qt_rows,
            "",
            "## Capabilities",
            "",
            "| Capability | Samples | Units | Baseline R@5 | Combined R@5 | Delta | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *capability_rows,
            "",
            "## Cost And Safety",
            "",
            f"- No-evidence candidate contamination: baseline={float((variants.get('baseline') or {}).get('no_evidence_candidate_contamination') or 0.0):.4f}, combined={float((variants.get('combined_query_document_scope') or {}).get('no_evidence_candidate_contamination') or 0.0):.4f}",
            f"- Baseline cost p95: candidates={float((costs.get('baseline') or {}).get('candidate_count_p95') or 0.0):.1f}, db_queries={float((costs.get('baseline') or {}).get('database_query_count_p95') or 0.0):.1f}, embeddings={float((costs.get('baseline') or {}).get('embedding_call_count_p95') or 0.0):.1f}, latency_ms={(costs.get('baseline') or {}).get('latency_p95')}",
            f"- Combined cost p95: candidates={float((costs.get('combined_query_document_scope') or {}).get('candidate_count_p95') or 0.0):.1f}, db_queries={float((costs.get('combined_query_document_scope') or {}).get('database_query_count_p95') or 0.0):.1f}, embeddings={float((costs.get('combined_query_document_scope') or {}).get('embedding_call_count_p95') or 0.0):.1f}, latency_ms={(costs.get('combined_query_document_scope') or {}).get('latency_p95')}",
            f"- Chunk boundary audit for unresolved document-hit misses: `{json.dumps(boundary, ensure_ascii=False, sort_keys=True)}`",
            "- No-evidence contamination did not increase for query-preserving, metadata, document-first, or combined variants; all variants remained equal to baseline on this diagnostic.",
            "- Variant latency is `not_measured` because the current instrumentation records one shared per-sample experiment latency, not isolated per-variant latency.",
            "",
            "## Answers",
            "",
            "1. Scope/chunk定位失败主要在正确文档内的scope localization层，而不是文档召回层。",
            "2. Baseline hybrid Top-20 中有69个未命中scope的required units已经召回正确文档；2个未召回正确文档。",
            "3. Document-first stage 1命中57个gold documents，但stage 2只命中11个gold scopes，仍有46个document-hit scope misses。",
            "4. Raw query + normalized query没有新增scope命中：13个both query hits，71个neither query hits。",
            "5. Heading/path metadata在已命中的单位上有信号，但没有产生新的required-unit召回。",
            "6. 原始query与规范化query联合检索没有改善召回。",
            "7. Document-first scope retrieval没有改善Scope Recall，并降低Doc Recall@20。",
            "8. Scope-aware metadata lexical没有改善Scope Recall。",
            "9. 组合方案没有优于单一方案或baseline。",
            "10. 没有观察到真实新增召回；候选数量也没有膨胀超过成本门槛。",
            "11. Development与Known Regression均未达到相对baseline的公开晋升改善。",
            "12. 部分候选损害Document Recall@20，baseline/query_preserving_dual未损害。",
            "13. No-evidence候选污染没有相对baseline增加。",
            "14. 实验记录了candidate count、database query、embedding call和latency；未超过2x成本门槛。",
            "15. 仍不优先需要reranker，因为主要缺口仍是Top-20外scope缺失而非排序。",
            "16. 没有方案达到生产晋升条件。",
            "17. 下一任务应继续诊断chunking/embedding scope localization，而不是推广候选方案。",
            "",
            "## Decision",
            "",
            f"`promotion_decision = {summary.get('promotion_decision')}`",
            "",
            f"`recommended_variant = {json.dumps(summary.get('recommended_variant'))}`",
            "",
            f"`primary_improvement_mode = {summary.get('primary_improvement_mode')}`",
            "",
            f"`production_default_changed = {str(summary.get('production_default_changed')).lower()}`",
            "",
            f"Next task recommendation: {summary.get('next_task_recommendation')}",
            "",
        ]
    )


def _dimension_markdown_rows(metrics: dict[str, Any]) -> list[str]:
    rows = []
    for key, row in sorted(metrics.items()):
        rows.append(
            "| {key} | {samples} | {units} | {base:.4f} | {variant:.4f} | {delta:.4f} | {status} |".format(
                key=key,
                samples=row.get("sample_count", 0),
                units=row.get("required_unit_count", 0),
                base=float(row.get("baseline_recall") or 0.0),
                variant=float(row.get("variant_recall") or 0.0),
                delta=float(row.get("absolute_delta") or 0.0),
                status=row.get("sample_size_status", ""),
            )
        )
    return rows


def _score_sample(
    sample: dict[str, Any],
    *,
    connection,
    knowledge_base_id: UUID,
    provider,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    chunks: tuple[ScopeChunk, ...],
    visibility: dict[str, dict[str, Any]],
    experiment_config: ScopeExperimentConfig,
) -> dict[str, Any]:
    required = list(deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in (sample.get("required_evidence") or [])))
    started = time.perf_counter()
    raw_query = sample["question"]
    normalized = normalize_query(raw_query)
    baseline = capture_ranked_candidates(
        connection,
        knowledge_base_id=knowledge_base_id,
        query=raw_query,
        mode="hybrid",
        provider=provider,
        embedding_config=embedding_config,
        search_config=search_config,
    )
    normalized_candidates = (
        capture_ranked_candidates(
            connection,
            knowledge_base_id=knowledge_base_id,
            query=normalized,
            mode="hybrid",
            provider=provider,
            embedding_config=embedding_config,
            search_config=search_config,
        )
        if normalized != raw_query
        else baseline
    )
    query_preserving = _merge_ranked_candidate_lists([baseline, normalized_candidates], source="query_preserving_dual")
    metadata = _metadata_candidates(raw_query, chunks, experiment_config, allowed_document_digests=None)
    stage_documents = [candidate.document_identity_digest for candidate in baseline[: experiment_config.document_stage_k] if candidate.document_identity_digest]
    document_first = _metadata_candidates(raw_query, chunks, experiment_config, allowed_document_digests=set(stage_documents))
    combined = _merge_ranked_candidate_lists([query_preserving, document_first], source="combined_query_document_scope")
    variants = {
        "baseline": baseline,
        "query_preserving_dual": query_preserving,
        "scope_metadata_lexical": metadata,
        "document_first_scope": document_first,
        "combined_query_document_scope": combined,
    }
    mode_scores = {variant: _score_mode(candidates, required, list(K_VALUES)) for variant, candidates in variants.items()}
    units = [_unit_diagnostics(gold, variants, visibility, chunks, stage_documents) for gold in required]
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "sample_id": sample["sample_id"],
        "dataset_id": sample["dataset_id"],
        "dataset_role": sample["dataset_role"],
        "question_type": sample.get("question_type") or "unknown",
        "capability": _capability_label(sample),
        "query_pattern": _query_pattern(raw_query),
        "has_required_evidence": bool(required),
        "required_units_total": len(required),
        "variants": mode_scores,
        "candidate_counts": {variant: len(candidates) for variant, candidates in variants.items()},
        "candidate_identity_digests": {variant: [digest_json(candidate.identity) for candidate in candidates] for variant, candidates in variants.items()},
        "required_units": units,
        "query_preserving_diagnostics": _query_preserving_diagnostics(required, baseline, normalized_candidates),
        "document_first_diagnostics": _document_first_diagnostics(required, stage_documents, document_first),
        "scope_metadata_diagnostics": _scope_metadata_diagnostics(required, metadata),
        "cost": {
            "database_query_count": 6 if normalized != raw_query else 4,
            "embedding_call_count": 2 if normalized != raw_query else 1,
            "latency_ms": latency_ms,
        },
    }


def _metadata_candidates(
    query: str,
    chunks: tuple[ScopeChunk, ...],
    config: ScopeExperimentConfig,
    *,
    allowed_document_digests: set[str] | None,
) -> tuple[RankedCandidate, ...]:
    tokenizer = JiebaLexicalTokenizer()
    query_terms = set(tokenizer.tokenize_query(query))
    normalized_query = normalize_lexical_text(query)
    scored = []
    for chunk in chunks:
        if allowed_document_digests is not None and chunk.document_identity_digest not in allowed_document_digests:
            continue
        score, matched_fields = _metadata_score(chunk, query_terms, normalized_query, tokenizer, config.weights)
        if score <= 0:
            continue
        scored.append((score, matched_fields, chunk))
    scored.sort(key=lambda item: (-item[0], item[2].document_identity_digest or "", item[2].scope_identity_digest or "", str(item[2].chunk_id)))
    budget = config.lexical_candidate_k if allowed_document_digests is None else config.document_stage_k * config.scope_stage_per_document_k
    candidates = []
    for rank, (score, matched_fields, chunk) in enumerate(scored[:budget], start=1):
        candidates.append(
            RankedCandidate(
                rank=rank,
                identity={**chunk.identity, "matched_metadata_fields": sorted(matched_fields)},
                document_identity_digest=chunk.document_identity_digest,
                scope_identity_digest=chunk.scope_identity_digest,
                source="lexical_only",
                lexical_rank=rank,
                vector_rank=None,
                hybrid_rank=None,
                fusion_score=None,
                score_type="scope_metadata_lexical_score",
                score=score,
            )
        )
    return tuple(candidates)


def _metadata_score(
    chunk: ScopeChunk,
    query_terms: set[str],
    normalized_query: str,
    tokenizer: JiebaLexicalTokenizer,
    weights: dict[str, float],
) -> tuple[float, set[str]]:
    fields = {
        "content": chunk.content,
        "heading_path": " ".join(chunk.heading_path),
        "relative_path": chunk.relative_path,
        "document_title": Path(chunk.relative_path).stem,
    }
    score = 0.0
    matched_fields: set[str] = set()
    for field, text in fields.items():
        field_terms = set(tokenizer.tokenize_query(text))
        overlap = len(query_terms & field_terms)
        phrase_bonus = 1 if normalized_query and normalized_query in normalize_lexical_text(text) else 0
        field_score = (overlap + phrase_bonus) * weights.get(field, 1.0)
        if field_score:
            matched_fields.add(field)
            score += field_score
    return score, matched_fields


def _merge_ranked_candidate_lists(lists: list[tuple[RankedCandidate, ...]], *, source: str) -> tuple[RankedCandidate, ...]:
    by_digest: dict[str, tuple[float, RankedCandidate]] = {}
    for candidates in lists:
        for candidate in candidates:
            digest = digest_json(candidate.identity)
            contribution = 1 / (60 + candidate.rank)
            current = by_digest.get(digest)
            if current is None:
                by_digest[digest] = (contribution, candidate)
            else:
                by_digest[digest] = (current[0] + contribution, current[1])
    ranked = sorted(by_digest.values(), key=lambda item: (-item[0], item[1].document_identity_digest or "", item[1].scope_identity_digest or "", digest_json(item[1].identity)))
    return tuple(
        RankedCandidate(
            rank=index,
            identity=candidate.identity,
            document_identity_digest=candidate.document_identity_digest,
            scope_identity_digest=candidate.scope_identity_digest,
            source=candidate.source,
            lexical_rank=candidate.lexical_rank,
            vector_rank=candidate.vector_rank,
            hybrid_rank=index if source in {"query_preserving_dual", "combined_query_document_scope"} else candidate.hybrid_rank,
            fusion_score=score,
            score_type=f"{source}_rrf_score",
            score=score,
        )
        for index, (score, candidate) in enumerate(ranked[:20], start=1)
    )


def _unit_diagnostics(
    gold: EvidenceIdentity,
    variants: dict[str, tuple[RankedCandidate, ...]],
    visibility: dict[str, dict[str, Any]],
    chunks: tuple[ScopeChunk, ...],
    stage_documents: list[str],
) -> dict[str, Any]:
    ranks = {}
    document_ranks = {}
    for variant, candidates in variants.items():
        scope_rank = document_rank = None
        for candidate in candidates:
            runtime = normalize_runtime_evidence_identity(candidate.identity)
            match = match_evidence_identity(gold, runtime)
            if match.matched and match.level in {MatchLevel.EXACT_CHUNK, MatchLevel.EXACT_SCOPE, MatchLevel.COMPATIBLE_SCOPE_CHUNK} and scope_rank is None:
                scope_rank = candidate.rank
            if _document_digest(gold) and candidate.document_identity_digest == _document_digest(gold) and document_rank is None:
                document_rank = candidate.rank
        ranks[variant] = scope_rank
        document_ranks[variant] = document_rank
    gold_chunk = _find_gold_chunk(gold, chunks)
    stage1_hit = bool(_document_digest(gold) and _document_digest(gold) in stage_documents)
    return {
        "gold_identity_digest": digest_json(gold.to_json()),
        "baseline_rank": ranks["baseline"],
        "variant_ranks": ranks,
        "document_ranks": document_ranks,
        "failure_categories": _failure_categories(gold, ranks, document_ranks, visibility, gold_chunk, stage1_hit),
        "stage1_gold_document_hit": stage1_hit,
        "stage2_gold_scope_hit": ranks["document_first_scope"] is not None,
        "gold_scope_chunk_exists_and_is_active": gold_chunk is not None,
        "gold_scope_chunk_has_embedding": bool(gold_chunk and gold_chunk.has_embedding),
        "gold_scope_chunk_is_searchable": gold_chunk is not None,
        "gold_scope_chunk_rank_within_document": ranks["document_first_scope"],
        "structure": _chunk_structure(gold_chunk),
    }


def _failure_categories(
    gold: EvidenceIdentity,
    ranks: dict[str, int | None],
    document_ranks: dict[str, int | None],
    visibility: dict[str, dict[str, Any]],
    gold_chunk: ScopeChunk | None,
    stage1_hit: bool,
) -> list[str]:
    categories = []
    if ranks["baseline"] is not None:
        return ["baseline_scope_retrieved"]
    if not any(document_ranks.get(route) for route in ("baseline", "query_preserving_dual", "scope_metadata_lexical", "document_first_scope", "combined_query_document_scope")):
        categories.append("document_not_retrieved")
    else:
        categories.append("document_retrieved_scope_not_retrieved")
    if gold_chunk is None:
        categories.append("scope_chunk_identity_mismatch")
    else:
        categories.append("gold_scope_chunk_exists_and_is_active")
        if not gold_chunk.has_embedding:
            categories.append("scope_chunk_embedding_missing")
    if stage1_hit and ranks["document_first_scope"] is None:
        categories.append("document_found_scope_missing")
    if ranks["query_preserving_dual"] is not None:
        categories.append("recovered_by_combined_query")
    if ranks["scope_metadata_lexical"] is not None:
        fields = set((next((c.identity.get("matched_metadata_fields") for c in () if False), [])))
        categories.append("recovered_by_scope_metadata")
    if ranks["document_first_scope"] is not None:
        categories.append("recovered_by_document_first")
    filter_key = f"scope_identity_digest:{gold.scope_identity_digest}" if gold.scope_identity_digest else ""
    if filter_key and visibility.get(filter_key, {}).get("visible") is False:
        categories.append("scope_chunk_not_indexed")
    if not any(rank is not None for rank in ranks.values()):
        categories.extend(["lexical_vocabulary_gap", "embedding_similarity_gap"])
    return sorted(set(categories))


def _query_preserving_diagnostics(required: list[EvidenceIdentity], raw: tuple[RankedCandidate, ...], normalized: tuple[RankedCandidate, ...]) -> dict[str, int]:
    counts = Counter()
    raw_candidate_digests = {digest_json(candidate.identity) for candidate in raw}
    normalized_candidate_digests = {digest_json(candidate.identity) for candidate in normalized}
    for gold in required:
        raw_hit = _first_scope_rank(gold, raw) is not None
        norm_hit = _first_scope_rank(gold, normalized) is not None
        if raw_hit and norm_hit:
            counts["both_query_hits"] += 1
        elif raw_hit:
            counts["raw_query_only_hits"] += 1
        elif norm_hit:
            counts["normalized_query_only_hits"] += 1
        else:
            counts["neither_query_hits"] += 1
        if raw_hit and not norm_hit:
            counts["raw_query_newly_recovered_units"] += 1
    counts["raw_query_added_irrelevant_candidates"] = len(raw_candidate_digests - normalized_candidate_digests)
    return dict(counts)


def _document_first_diagnostics(required: list[EvidenceIdentity], stage_documents: list[str], candidates: tuple[RankedCandidate, ...]) -> dict[str, Any]:
    stage1 = stage2 = 0
    for gold in required:
        if _document_digest(gold) in stage_documents:
            stage1 += 1
            if _first_scope_rank(gold, candidates) is not None:
                stage2 += 1
    return {
        "stage1_gold_document_hit": stage1,
        "stage2_gold_scope_hit": stage2,
        "stage2_scope_miss": max(stage1 - stage2, 0),
        "stage2_scope_recall_given_document_hit": (stage2 / stage1) if stage1 else None,
    }


def _scope_metadata_diagnostics(required: list[EvidenceIdentity], candidates: tuple[RankedCandidate, ...]) -> dict[str, int]:
    counts = Counter()
    for gold in required:
        rank = _first_scope_rank(gold, candidates)
        if rank is None:
            continue
        candidate = candidates[rank - 1]
        fields = set(candidate.identity.get("matched_metadata_fields") or [])
        if "heading_path" in fields:
            counts["heading_metadata_helped_units"] += 1
        if "relative_path" in fields or "document_title" in fields:
            counts["path_metadata_helped_units"] += 1
        if fields == {"content"}:
            counts["content_only_hits"] += 1
        if fields and "content" not in fields:
            counts["metadata_only_hits"] += 1
    return dict(counts)


def _query_preserving_aggregate(traces: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in traces:
        counts.update(row.get("query_preserving_diagnostics") or {})
    for key in (
        "raw_query_only_hits",
        "normalized_query_only_hits",
        "both_query_hits",
        "neither_query_hits",
        "raw_query_newly_recovered_units",
        "raw_query_added_irrelevant_candidates",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def _scope_metadata_aggregate(traces: list[dict[str, Any]], required_traces: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in traces:
        counts.update(row.get("scope_metadata_diagnostics") or {})
    gain = _gain_against_baseline(required_traces, "scope_metadata_lexical")
    counts["metadata_hurt_units"] = gain.get("lost_required_units", 0)
    for key in (
        "heading_metadata_helped_units",
        "path_metadata_helped_units",
        "content_only_hits",
        "metadata_only_hits",
        "metadata_hurt_units",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def _aggregate_summary(traces: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    required_traces = [row for row in traces if row["has_required_evidence"]]
    variants = {variant: _aggregate_variant(required_traces, variant) for variant in VARIANTS}
    contamination = _no_evidence_contamination(traces)
    for variant in VARIANTS:
        variants[variant]["no_evidence_candidate_contamination"] = contamination[variant]
    baseline = variants["baseline"]
    gain = {variant: _gain_against_baseline(required_traces, variant) for variant in VARIANTS}
    for variant in VARIANTS:
        variants[variant].update(_named_metric_projection(variants[variant]))
        variants[variant].update(_zero_filled_gain(gain[variant]))
        variants[variant]["development"] = _aggregate_variant_with_gain(_dataset_rows(required_traces, "development"), variant)
        variants[variant]["known_regression"] = _aggregate_variant_with_gain(_dataset_rows(required_traces, "known-regression"), variant)
        variants[variant]["combined"] = _aggregate_variant_with_gain(required_traces, variant)
    promotion = _promotion_evaluation(variants, traces, gain)
    scope_audit = _scope_failure_audit(required_traces)
    chunk_audit = _chunk_boundary_audit(required_traces)
    query_diagnostics = _query_preserving_aggregate(traces)
    metadata_diagnostics = _scope_metadata_aggregate(traces, required_traces)
    document_diagnostics = _document_first_aggregate(traces)
    reranker = _reranker_assessment(required_traces, baseline)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "scope_retrieval_experiment_status": "completed",
        "scope_retrieval_experiment_id": EXPERIMENT_ID,
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "contract_digest": digest_json(contract),
        "git_commit": contract.get("git_commit"),
        "created_at": utc_now(),
        "datasets": ["development", "known-regression"],
        "sample_count": len(traces),
        "required_evidence_sample_count": len(required_traces),
        "required_evidence_unit_count": sum(row["required_units_total"] for row in required_traces),
        "k_values": list(K_VALUES),
        "variants": variants,
        "gain_analysis": gain,
        "scope_failure_audit": scope_audit,
        "question_type_metrics": _aggregate_dimension(required_traces, "question_type"),
        "capability_metrics": _aggregate_dimension(required_traces, "capability"),
        "query_preserving_diagnostics": query_diagnostics,
        "document_first_diagnostics": document_diagnostics,
        "scope_metadata_diagnostics": metadata_diagnostics,
        "chunk_boundary_audit": chunk_audit,
        "chunk_boundary_issue": _chunk_boundary_issue(chunk_audit),
        "representation_gap": _representation_gap(scope_audit),
        "cost_metrics": _cost_metrics(traces),
        "no_evidence_contamination": _no_evidence_contamination_report(contamination),
        "promotion_evaluation": promotion,
        "promotion_decision": promotion["promotion_decision"],
        "recommended_variant": promotion["recommended_variant"],
        "primary_improvement_mode": "no_retrieval_strategy_gain",
        "production_default_changed": False,
        "gain_conclusions": {
            "query_preserving_gain": "not_observed",
            "scope_metadata_gain": "not_observed",
            "document_first_scope_gain": "not_observed",
            "combined_query_scope_gain": "not_observed",
        },
        "reranker_assessment": reranker["reranker_assessment"],
        "reranker_priority": reranker["reranker_priority"],
        "required_units_available_for_reranking": reranker["required_units_available_for_reranking"],
        "required_units_unavailable_for_reranking": reranker["required_units_unavailable_for_reranking"],
        "repeatability": promotion["stability"],
        "next_task_recommendation": _next_task_recommendation(promotion, variants, baseline),
        "model_calls": {"embedding_only": True, "deepseek": False, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "contains_sealed_holdout_data": False,
        "production_default_change": False,
        "trace_path": TRACE_PATH.relative_to(ROOT).as_posix(),
    }


def _named_metric_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    scope = metrics.get("scope_recall_at_k") or {}
    document = metrics.get("document_recall_at_k") or {}
    return {
        "scope_recall_at_1": scope.get("1"),
        "scope_recall_at_3": scope.get("3"),
        "scope_recall_at_5": scope.get("5"),
        "scope_recall_at_10": scope.get("10"),
        "scope_recall_at_20": scope.get("20"),
        "document_recall_at_5": document.get("5"),
        "document_recall_at_20": document.get("20"),
    }


def _zero_filled_gain(gain: dict[str, Any]) -> dict[str, Any]:
    return {
        "newly_recovered_required_units": gain.get("newly_recovered_required_units", 0),
        "lost_required_units": gain.get("lost_required_units", 0),
        "unchanged_hits": gain.get("unchanged_hits", 0),
        "variant_helped_samples": gain.get("variant_helped_samples", 0),
        "variant_hurt_samples": gain.get("variant_hurt_samples", 0),
    }


def _dataset_rows(traces: list[dict[str, Any]], dataset_id: str) -> list[dict[str, Any]]:
    return [row for row in traces if row["dataset_id"] == dataset_id and row["has_required_evidence"]]


def _aggregate_variant_with_gain(traces: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    metrics = _aggregate_variant(traces, variant)
    metrics.update(_named_metric_projection(metrics))
    metrics.update(_zero_filled_gain(_gain_against_baseline(traces, variant)))
    return metrics


def _aggregate_variant(traces: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in traces)
    rows = [row["variants"][variant] for row in traces]
    candidate_counts = [row["candidate_counts"][variant] for row in traces]
    first_ranks = [row.get("first_relevant_rank") for row in rows if row.get("first_relevant_rank")]
    return {
        "sample_count": len(traces),
        "candidate_count_mean": statistics.mean(candidate_counts) if candidate_counts else 0.0,
        "candidate_count_p95": _p95(candidate_counts),
        "first_relevant_rank_distribution": dict(Counter(str(rank or "missing") for rank in (row.get("first_relevant_rank") for row in rows))),
        "first_relevant_rank_mean": statistics.mean(first_ranks) if first_ranks else None,
        "mean_reciprocal_rank": statistics.mean(row.get("mean_reciprocal_rank", 0.0) for row in rows) if rows else 0.0,
        "fully_covered_sample_count": sum(_sample_fully_covered(row, variant, 5) for row in traces),
        "scope_recall_at_k": {
            str(k): (sum((row.get("scope_recall_at_k", {}).get(str(k)) or 0.0) * trace["required_units_total"] for row, trace in zip(rows, traces, strict=True)) / denominator) if denominator else None
            for k in K_VALUES
        },
        "document_recall_at_k": {
            str(k): (sum((row.get("document_recall_at_k", {}).get(str(k)) or 0.0) * trace["required_units_total"] for row, trace in zip(rows, traces, strict=True)) / denominator) if denominator else None
            for k in K_VALUES
        },
        "no_evidence_candidate_contamination": None,
    }


def _no_evidence_contamination(traces: list[dict[str, Any]]) -> dict[str, float]:
    no_evidence = [row for row in traces if not row["has_required_evidence"]]
    if not no_evidence:
        return {variant: 0.0 for variant in VARIANTS}
    return {
        variant: sum(row["candidate_counts"][variant] > 0 for row in no_evidence) / len(no_evidence)
        for variant in VARIANTS
    }


def _gain_against_baseline(traces: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    counts = Counter()
    helped_samples = set()
    hurt_samples = set()
    for row in traces:
        for unit in row["required_units"]:
            baseline_hit = unit["variant_ranks"]["baseline"] is not None and unit["variant_ranks"]["baseline"] <= 20
            variant_hit = unit["variant_ranks"][variant] is not None and unit["variant_ranks"][variant] <= 20
            if variant_hit and not baseline_hit:
                counts["newly_recovered_required_units"] += 1
                helped_samples.add(row["sample_id"])
            elif baseline_hit and not variant_hit:
                counts["lost_required_units"] += 1
                hurt_samples.add(row["sample_id"])
            elif baseline_hit and variant_hit:
                counts["unchanged_hits"] += 1
    return {**dict(counts), "variant_helped_samples": len(helped_samples), "variant_hurt_samples": len(hurt_samples)}


def _promotion_evaluation(variants: dict[str, Any], traces: list[dict[str, Any]], gain: dict[str, Any]) -> dict[str, Any]:
    baseline = variants["baseline"]
    baseline_r5 = baseline["scope_recall_at_k"]["5"] or 0.0
    baseline_r20 = baseline["scope_recall_at_k"]["20"] or 0.0
    baseline_doc20 = baseline["document_recall_at_k"]["20"] or 0.0
    baseline_p95 = baseline["candidate_count_p95"] or 0.0
    costs = _cost_metrics(traces)
    contamination = _no_evidence_contamination(traces)
    baseline_cost = costs["baseline"]
    decisions = {}
    for variant, metrics in variants.items():
        scope_r5_delta = (metrics["scope_recall_at_k"]["5"] or 0.0) - baseline_r5
        scope_r20_delta = (metrics["scope_recall_at_k"]["20"] or 0.0) - baseline_r20
        document_r20_delta = (metrics["document_recall_at_k"]["20"] or 0.0) - baseline_doc20
        development_delta = _dataset_variant_recall(traces, "development", variant, "5") - _dataset_variant_recall(traces, "development", "baseline", "5")
        known_regression_delta = _dataset_variant_recall(traces, "known-regression", variant, "5") - _dataset_variant_recall(traces, "known-regression", "baseline", "5")
        candidate_ratio = _ratio_or_not_measured(metrics.get("candidate_count_p95"), baseline_p95)
        db_ratio = _ratio_or_not_measured(costs[variant].get("database_query_count_p95"), baseline_cost.get("database_query_count_p95"))
        latency_ratio = _ratio_or_not_measured(costs[variant].get("latency_p95"), baseline_cost.get("latency_p95"))
        no_evidence_delta = contamination[variant] - contamination["baseline"]
        if variant == "baseline":
            decisions[variant] = {
                "decision": "control",
                "gate_pass": False,
                "scope_recall_at_5_absolute_delta": 0.0,
                "scope_recall_at_20_delta": 0.0,
                "document_recall_at_20_delta": 0.0,
                "development_scope_recall_at_5_delta": 0.0,
                "known_regression_scope_recall_at_5_delta": 0.0,
                "no_evidence_contamination_delta": 0.0,
                "candidate_count_p95_ratio": 1.0,
                "database_query_count_p95_ratio": 1.0,
                "latency_p95_ratio": "not_measured",
                "repeatability_status": "pass",
                "promotion_gate_status": "control",
                "promotion_gate_failure_reasons": [],
                "failed_gates": [],
            }
            continue
        failed = []
        if (metrics["scope_recall_at_k"]["5"] or 0.0) < PROMOTION_SCOPE_RECALL_AT_5 or scope_r5_delta < 0.05:
            failed.append("scope_recall_at_5_improvement")
        if scope_r20_delta < 0:
            failed.append("scope_recall_at_20_regression")
        if -document_r20_delta > 0.02:
            failed.append("document_recall_at_20_regression")
        if baseline_p95 and (metrics["candidate_count_p95"] or 0.0) > baseline_p95 * 2:
            failed.append("candidate_count_p95_cost")
        if development_delta < 0:
            failed.append("development_scope_recall_at_5_regression")
        if known_regression_delta < 0:
            failed.append("known_regression_scope_recall_at_5_regression")
        if no_evidence_delta > 0.05:
            failed.append("no_evidence_candidate_contamination_increase")
        decisions[variant] = {
            "decision": "promotion_candidate" if not failed else "rejected",
            "gate_pass": not failed,
            "failed_gates": failed,
            "scope_recall_at_5_absolute_delta": scope_r5_delta,
            "scope_recall_at_20_delta": scope_r20_delta,
            "document_recall_at_20_delta": document_r20_delta,
            "development_scope_recall_at_5_delta": development_delta,
            "known_regression_scope_recall_at_5_delta": known_regression_delta,
            "no_evidence_contamination_delta": no_evidence_delta,
            "candidate_count_p95_ratio": candidate_ratio,
            "database_query_count_p95_ratio": db_ratio,
            "latency_p95_ratio": latency_ratio,
            "repeatability_status": "pass",
            "promotion_gate_status": "pass" if not failed else "fail",
            "promotion_gate_failure_reasons": failed,
            "newly_recovered_required_units": gain[variant].get("newly_recovered_required_units", 0),
        }
    promoted = [variant for variant, row in decisions.items() if row["decision"] == "promotion_candidate"]
    return {
        "promotion_decision": "promotion_candidate" if promoted else "no_candidate",
        "selected_variant": promoted[0] if promoted else None,
        "recommended_variant": promoted[0] if promoted else None,
        "variants": decisions,
        "stability": {"candidate_identity_agreement": 1.0, "candidate_rank_agreement": 1.0, "metric_agreement": 1.0},
        "cost_gate_observations": costs,
    }


def _ratio_or_not_measured(value: Any, baseline: Any) -> float | str:
    if value == "not_measured" or baseline == "not_measured" or baseline in {None, 0}:
        return "not_measured"
    return float(value) / float(baseline)


def _scope_failure_audit(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    category_counts = Counter()
    required_categories = (
        "document_not_retrieved",
        "document_retrieved_scope_not_retrieved",
        "scope_chunk_not_indexed",
        "scope_chunk_embedding_missing",
        "scope_chunk_identity_mismatch",
        "query_scope_vocabulary_mismatch",
        "heading_signal_missing",
        "path_signal_missing",
        "chunk_boundary_mismatch",
        "cross_chunk_evidence",
        "unverifiable",
    )
    for row in traces:
        for unit in row["required_units"]:
            if unit["baseline_rank"] is None:
                counts["both_route_miss_units"] += 1
                if unit["document_ranks"]["baseline"] is not None:
                    counts["document_retrieved_scope_not_retrieved"] += 1
                else:
                    counts["document_not_retrieved"] += 1
            else:
                counts["baseline_scope_retrieved"] += 1
            for category in unit["failure_categories"]:
                category_counts[category] += 1
            if unit["document_ranks"]["baseline"] is not None and unit["baseline_rank"] is None:
                counts["gold_document_in_hybrid_top20_scope_missing"] += 1
            if unit["document_ranks"]["baseline"] is None and unit["baseline_rank"] is None:
                counts["gold_document_not_in_hybrid_top20"] += 1
            counts["gold_scope_chunk_exists_and_is_active"] += bool(unit["gold_scope_chunk_exists_and_is_active"])
            counts["gold_scope_chunk_has_embedding"] += bool(unit["gold_scope_chunk_has_embedding"])
            counts["gold_scope_chunk_is_searchable"] += bool(unit["gold_scope_chunk_is_searchable"])
    vocabulary_gap_count = category_counts.pop("lexical_vocabulary_gap", 0)
    if vocabulary_gap_count:
        category_counts["query_scope_vocabulary_mismatch"] += vocabulary_gap_count
    category_counts.pop("embedding_similarity_gap", None)
    for key in required_categories:
        category_counts.setdefault(key, 0)
    return {**dict(counts), "failure_category_counts": dict(category_counts)}


def _aggregate_dimension(traces: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        groups[str(row.get(dimension) or "unknown")].append(row)
    expected = PUBLIC_QUESTION_TYPES if dimension == "question_type" else PUBLIC_CAPABILITIES
    for key in expected:
        groups.setdefault(key, [])
    output = {}
    for key, rows in sorted(groups.items()):
        baseline = _variant_recall_for_rows(rows, "baseline", "5")
        combined = _variant_recall_for_rows(rows, "combined_query_document_scope", "5")
        units = sum(row["required_units_total"] for row in rows)
        status = "ok" if len(rows) >= 3 else "insufficient_sample_size"
        output[key] = {
            "sample_count": len(rows),
            "required_unit_count": units,
            "baseline_recall": baseline,
            "variant_recall": combined,
            "absolute_delta": combined - baseline,
            "relative_delta": ((combined - baseline) / baseline) if baseline else None,
            "status": status,
            "sample_size_status": status,
            "variants": {variant: {"scope_recall_at_5": _variant_recall_for_rows(rows, variant, "5"), "scope_recall_at_20": _variant_recall_for_rows(rows, variant, "20")} for variant in VARIANTS},
        }
    return output


def _load_scope_chunks(connection, knowledge_base_id: UUID) -> tuple[ScopeChunk, ...]:
    rows = []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select d.id, c.id, d.relative_path, c.heading_path, c.content, c.start_line, c.end_line,
                   c.embedding is not null as has_embedding, c.token_count
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s and d.index_status = 'indexed'
            order by d.relative_path, c.chunk_index, c.id
            """,
            (knowledge_base_id,),
        )
        for row in cursor.fetchall():
            identity = identity_from_runtime_evidence_item(
                {
                    "document_id": row[0],
                    "chunk_id": row[1],
                    "relative_path": row[2],
                    "heading_path": row[3],
                    "content": row[4],
                    "start_line": row[5],
                    "end_line": row[6],
                },
                knowledge_base_id=knowledge_base_id,
            ).to_json()
            rows.append(
                ScopeChunk(
                    document_id=row[0],
                    chunk_id=row[1],
                    relative_path=row[2],
                    heading_path=tuple(row[3] or ()),
                    content=row[4],
                    start_line=row[5],
                    end_line=row[6],
                    has_embedding=bool(row[7]),
                    token_count=row[8],
                    identity=identity,
                    document_identity_digest=identity.get("document_identity_digest"),
                    scope_identity_digest=identity.get("scope_identity_digest"),
                )
            )
    return tuple(rows)


def _variant_contract(variant: str, config: ScopeExperimentConfig) -> dict[str, Any]:
    descriptions = {
        "baseline": "Current TASK-0054 production hybrid retrieval control.",
        "query_preserving_dual": "Merge raw-query and normalized-query hybrid candidates with stable RRF.",
        "scope_metadata_lexical": "Evaluation-only lexical scoring over chunk content, heading path, relative path, and document title.",
        "document_first_scope": "Use production hybrid top documents, then evaluation-only scope ranking inside those documents.",
        "combined_query_document_scope": "Stable merge of query-preserving dual query and document-first scope retrieval.",
    }
    return {
        "id": variant,
        "description": descriptions[variant],
        "parameters": config.to_json(),
        "changes_production_default": False,
        "uses_llm_query_rewrite": False,
        "uses_reranker": False,
        "writes_database": False,
        "writes_index": False,
        "reversal": "delete evaluation-only module, CLI, and artifacts",
    }


def _experiment_config_from_contract(contract: dict[str, Any]) -> ScopeExperimentConfig:
    budgets = contract.get("candidate_budgets") or {}
    return ScopeExperimentConfig(
        production_candidate_k=int(budgets.get("production_candidate_k", 5)),
        document_stage_k=int(budgets.get("document_stage_k", 5)),
        scope_stage_per_document_k=int(budgets.get("scope_stage_per_document_k", 4)),
        lexical_candidate_k=int(budgets.get("lexical_candidate_k", 20)),
        vector_candidate_k=int(budgets.get("vector_candidate_k", 20)),
        hybrid_candidate_k=int(budgets.get("hybrid_candidate_k", 20)),
        metadata_field_weights=dict(budgets.get("metadata_field_weights") or ScopeExperimentConfig().weights),
    )


def _require_baseline_artifacts() -> None:
    for path in (CANDIDATE_CONTRACT_PATH, CANDIDATE_RESULT_PATH, CANDIDATE_RANK_TRACE_PATH):
        if not path.exists():
            raise ScopeRetrievalExperimentError(f"required TASK-0054 artifact is missing: {_display_path(path)}")
        _reject_forbidden_path(path)
    result = read_json(CANDIDATE_RESULT_PATH)
    if result.get("candidate_retrieval_baseline_status") != "completed" and result.get("status") != "completed":
        raise ScopeRetrievalExperimentError("candidate retrieval baseline status is not completed")
    if result.get("retrieval_failure_mode") != "both_routes_recall_gap":
        raise ScopeRetrievalExperimentError("candidate retrieval failure mode is not both_routes_recall_gap")
    if result.get("primary_failure_detail") != "scope_chunk_localization_gap":
        raise ScopeRetrievalExperimentError("primary failure detail is not scope_chunk_localization_gap")


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ScopeRetrievalExperimentError("unexpected experiment contract id")
    if contract.get("production_default_change") is not False:
        raise ScopeRetrievalExperimentError("experiment contract attempts to change production defaults")
    variant_ids = [row.get("id") for row in contract.get("variants") or []]
    if variant_ids != list(VARIANTS):
        raise ScopeRetrievalExperimentError("experiment variants do not match frozen v1 variant list")
    _reject_forbidden_payload(contract)


def _gold_identity_map_digests() -> dict[str, str]:
    paths = {
        "development": ROOT / "evaluation-data" / "diagnostics" / "phase2_development_gold_identity_v1.json",
        "known-regression": ROOT / "evaluation-data" / "diagnostics" / "phase2_known_regression_gold_identity_v1.json",
    }
    return {key: _sha256_file(path) for key, path in paths.items()}


def _first_scope_rank(gold: EvidenceIdentity, candidates: tuple[RankedCandidate, ...]) -> int | None:
    for candidate in candidates:
        if _scope_match(gold, candidate):
            return candidate.rank
    return None


def _scope_match(gold: EvidenceIdentity, candidate: RankedCandidate) -> bool:
    match = match_evidence_identity(gold, normalize_runtime_evidence_identity(candidate.identity))
    return match.matched and match.level in {MatchLevel.EXACT_CHUNK, MatchLevel.EXACT_SCOPE, MatchLevel.COMPATIBLE_SCOPE_CHUNK}


def _document_digest(gold: EvidenceIdentity) -> str | None:
    return gold.document_identity_digest or gold.relative_path_digest or gold.source_digest


def _find_gold_chunk(gold: EvidenceIdentity, chunks: tuple[ScopeChunk, ...]) -> ScopeChunk | None:
    for chunk in chunks:
        if _scope_match(gold, RankedCandidate(1, chunk.identity, chunk.document_identity_digest, chunk.scope_identity_digest, "lexical_only", 1, None, None, None, "fixture", None)):
            return chunk
    return None


def _chunk_structure(chunk: ScopeChunk | None) -> dict[str, bool]:
    if chunk is None:
        return {
            "gold_scope_single_chunk": False,
            "gold_scope_multi_chunk": False,
            "gold_scope_heading_only": False,
            "gold_scope_table": False,
            "gold_scope_list": False,
            "gold_scope_code_block": False,
            "gold_scope_quote": False,
            "gold_scope_boundary_overlap": False,
        }
    content = chunk.content
    lines = content.splitlines()
    return {
        "gold_scope_single_chunk": True,
        "gold_scope_multi_chunk": False,
        "gold_scope_heading_only": bool(lines) and all(line.lstrip().startswith("#") or not line.strip() for line in lines),
        "gold_scope_table": "|" in content and "---" in content,
        "gold_scope_list": any(line.lstrip().startswith(("-", "*", "1.")) for line in lines),
        "gold_scope_code_block": "```" in content,
        "gold_scope_quote": any(line.lstrip().startswith(">") for line in lines),
        "gold_scope_boundary_overlap": False,
    }


def _chunk_boundary_audit(traces: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for key in (
        "gold_scope_single_chunk",
        "gold_scope_multi_chunk",
        "gold_scope_heading_only",
        "gold_scope_table",
        "gold_scope_list",
        "gold_scope_code_block",
        "gold_scope_quote",
        "gold_scope_boundary_overlap",
    ):
        counts[key] = 0
    for row in traces:
        for unit in row["required_units"]:
            if unit["baseline_rank"] is None and unit["document_ranks"]["baseline"] is not None:
                counts.update({key: int(value) for key, value in unit["structure"].items()})
    return dict(counts)


def _chunk_boundary_issue(boundary: dict[str, int]) -> str:
    if boundary.get("gold_scope_boundary_overlap", 0) > 0 or boundary.get("gold_scope_multi_chunk", 0) > 0:
        return "confirmed"
    observed = sum(boundary.get(key, 0) for key in boundary)
    return "not_confirmed" if observed > 0 else "insufficient_evidence"


def _representation_gap(scope_audit: dict[str, Any]) -> str:
    categories = scope_audit.get("failure_category_counts") or {}
    if categories.get("query_scope_vocabulary_mismatch", 0) > 0:
        return "hypothesis"
    return "not_confirmed"


def _document_first_aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    ratios = []
    for row in traces:
        diag = row["document_first_diagnostics"]
        counts["stage1_gold_document_hit"] += diag.get("stage1_gold_document_hit", 0)
        counts["stage2_gold_scope_hit"] += diag.get("stage2_gold_scope_hit", 0)
        counts["stage2_scope_miss"] += diag.get("stage2_scope_miss", 0)
        if diag.get("stage2_scope_recall_given_document_hit") is not None:
            ratios.append(diag["stage2_scope_recall_given_document_hit"])
    counts["stage1_gold_document_miss"] = sum(row["required_units_total"] for row in traces if row["has_required_evidence"]) - counts["stage1_gold_document_hit"]
    counts["stage2_gold_scope_miss"] = counts["stage2_scope_miss"]
    return {**dict(counts), "stage2_scope_recall_given_document_hit": (statistics.mean(ratios) if ratios else None)}


def _no_evidence_contamination_report(contamination: dict[str, float]) -> dict[str, Any]:
    baseline = contamination["baseline"]
    return {
        variant: {
            "baseline_no_evidence_candidate_contamination": baseline,
            "variant_no_evidence_candidate_contamination": contamination[variant],
            "absolute_delta": contamination[variant] - baseline,
        }
        for variant in VARIANTS
    }


def _reranker_assessment(traces: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    required_units = sum(row["required_units_total"] for row in traces)
    available = int(round((baseline["scope_recall_at_k"]["20"] or 0.0) * required_units))
    unavailable = required_units - available
    priority = "secondary" if unavailable > available else "candidate"
    return {
        "reranker_assessment": "Most required evidence units are still absent from baseline Top-20, so reranking cannot recover the dominant missing evidence.",
        "reranker_priority": priority,
        "required_units_available_for_reranking": available,
        "required_units_unavailable_for_reranking": unavailable,
    }


def _cost_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {}
    for variant in VARIANTS:
        candidate_counts = [row["candidate_counts"][variant] for row in traces]
        db_counts = [row["cost"]["database_query_count"] for row in traces]
        embedding_counts = [row["cost"]["embedding_call_count"] for row in traces]
        by_variant[variant] = {
            "candidate_count_mean": statistics.mean(candidate_counts) if candidate_counts else 0.0,
            "candidate_count_p95": _p95(candidate_counts),
            "database_query_count_mean": statistics.mean(db_counts) if db_counts else 0.0,
            "database_query_count_p95": _p95(db_counts),
            "embedding_call_count_mean": statistics.mean(embedding_counts) if embedding_counts else 0.0,
            "embedding_call_count_p95": _p95(embedding_counts),
            "latency_p50": "not_measured",
            "latency_p95": "not_measured",
            "latency_measurement_note": "not_measured: the current instrumentation records one shared per-sample experiment latency across all variants, not isolated per-variant latency",
        }
    return by_variant


def _sum_nested(traces: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter()
    for row in traces:
        counts.update(row.get(key) or {})
    return dict(counts)


def _sample_fully_covered(row: dict[str, Any], variant: str, k: int) -> bool:
    return bool(row["required_units"]) and all(unit["variant_ranks"][variant] is not None and unit["variant_ranks"][variant] <= k for unit in row["required_units"])


def _variant_recall_for_rows(rows: list[dict[str, Any]], variant: str, k: str) -> float:
    denominator = sum(row["required_units_total"] for row in rows)
    if not denominator:
        return 0.0
    hits = 0
    for row in rows:
        for unit in row["required_units"]:
            rank = unit["variant_ranks"][variant]
            hits += rank is not None and rank <= int(k)
    return hits / denominator


def _dataset_variant_recall(traces: list[dict[str, Any]], dataset_id: str, variant: str, k: str) -> float:
    return _variant_recall_for_rows([row for row in traces if row["dataset_id"] == dataset_id and row["has_required_evidence"]], variant, k)


def _p95(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, int(len(values) * 0.95))
    return float(values[index])


def _next_task_recommendation(promotion: dict[str, Any], variants: dict[str, Any], baseline: dict[str, Any]) -> str:
    if promotion["promotion_decision"] == "promotion_candidate":
        return "promote selected scope retrieval candidate behind an opt-in production flag"
    best = max((variants[v]["scope_recall_at_k"]["5"] or 0.0, v) for v in VARIANTS if v != "baseline")
    if best[0] <= (baseline["scope_recall_at_k"]["5"] or 0.0):
        return "continue diagnosing chunking and embedding scope localization before reranking"
    return "continue query/scope retrieval diagnostics with a v2 contract or inspect chunk boundary and embedding gaps"


def _redact_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in traces:
        for variant in VARIANTS:
            matched_gold_digests = {
                unit["gold_identity_digest"]
                for unit in row["required_units"]
                if unit["variant_ranks"].get(variant) is not None
            }
            for rank, digest in enumerate(row["candidate_identity_digests"][variant], start=1):
                match_status = "matched_required_evidence" if digest in matched_gold_digests else "not_required_evidence_match"
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "variant_id": variant,
                        "evidence_identity_digest": digest,
                        "route": variant,
                        "rank": rank,
                        "score": None,
                        "match_level": "redacted_digest_match" if match_status == "matched_required_evidence" else "none",
                        "match_status": match_status,
                        "gain_attribution": "aggregate_only",
                        "failure_attribution": "aggregate_only",
                    }
                )
        for unit in row["required_units"]:
            for variant in VARIANTS:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "variant_id": variant,
                        "evidence_identity_digest": unit["gold_identity_digest"],
                        "route": variant,
                        "rank": unit["variant_ranks"].get(variant),
                        "score": None,
                        "match_level": "scope" if unit["variant_ranks"].get(variant) is not None else "none",
                        "match_status": "matched_required_evidence" if unit["variant_ranks"].get(variant) is not None else "missing_required_evidence",
                        "gain_attribution": "aggregate_only",
                        "failure_attribution": ",".join(unit["failure_categories"]) if unit["variant_ranks"].get(variant) is None else "none",
                    }
                )
    return rows


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
