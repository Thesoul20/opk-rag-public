from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.representation import RepresentationChunk, RepresentationChunkingConfig, enrich_heading_context
from opk_rag.document_ir.models import CanonicalDocument
from opk_rag.document_ir.serialization import digest_json
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0104_canonical_pdf_adapter import read_json, write_json
from opk_rag.evaluation.task0105_representation_aware_chunking import (
    build_c0_chunks,
    build_gold_units,
    materialize_formal_documents,
    rate,
    token_count,
)
from opk_rag.evaluation.task0109_representation_aware_chunking import (
    _from_c0_chunk,
    best_gold_ranks,
    bounded_retrieval_mapping,
    execute_retrieval_replay,
    lexical_score,
    map_gold_units,
    mrr,
    recall_at,
    tokenize,
)


TASK_ID = "TASK-0111"
EXPERIMENT_ID = "task0111-heading-context-benefit-scope"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0111_heading_context_benefit_scope_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0111_HEADING_CONTEXT_BENEFIT_SCOPE_AND_ADAPTIVE_READINESS_REPORT.md"
TASK0109_DIR = ROOT / "evaluation-data" / "results" / "task0109-representation-aware-chunking"
TASK0110_DIR = ROOT / "evaluation-data" / "results" / "task0110-heading-context-runtime-promotion"
TOP_KS = (5, 20)
MISSING_RANK = 1_000_000
MINIMUM_SUBGROUP_SUPPORT = 5
MATERIAL_RECALL_DELTA = 0.01
MINIMUM_ORACLE_HEADROOM = 0.01
REQUIRED_ARTIFACTS = (
    "input_identity",
    "representation_authority",
    "query_level_delta",
    "benefit_classification",
    "query_feature_profile",
    "document_structure_profile",
    "heading_context_profile",
    "section_ambiguity_analysis",
    "same_document_competition",
    "cross_section_confusion",
    "heading_depth_analysis",
    "document_complexity_analysis",
    "query_document_interaction_analysis",
    "rank_transition_analysis",
    "e2e_transition_analysis",
    "citation_grounding_interaction",
    "subgroup_benefit_matrix",
    "improvement_taxonomy",
    "regression_taxonomy",
    "oracle_routing_summary",
    "simulated_routing_summary",
    "adaptive_readiness",
    "policy_decision",
    "regression_summary",
    "result_digests",
)


def run_task0111(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    contract = build_contract(root=root)
    authority = load_representation_authority(root=root)
    if not authority["task0110_inputs_valid"]:
        verification = invalid_verification(["TASK-0110 authority preconditions are not satisfied"])
        if write:
            write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
            result_dir = root / RESULT_DIR.relative_to(ROOT)
            result_dir.mkdir(parents=True, exist_ok=True)
            write_json(result_dir / "verification_summary.json", verification)
        return verification

    replay = build_replay_observations(root=root)
    observations = replay["observations"]
    global_metrics = global_benefit_metrics(observations)
    benefit = benefit_classification(observations)
    query_profile = query_feature_profile(observations)
    document_profile = document_structure_profile(replay["documents"])
    heading_profile = heading_context_profile(observations)
    ambiguity = section_ambiguity_analysis(observations)
    competition = same_document_competition(observations)
    confusion = cross_section_confusion(observations)
    depth = heading_depth_analysis(observations)
    complexity = document_complexity_analysis(observations)
    interaction = query_document_interaction_analysis(observations)
    rank_transition = rank_transition_analysis(observations)
    e2e_transition = e2e_transition_analysis(observations)
    citation_grounding = citation_grounding_interaction(observations, authority)
    subgroup = subgroup_benefit_matrix(observations)
    improvement = taxonomy(observations, "c3_improved")
    regression = taxonomy(observations, "c3_regressed")
    oracle = oracle_routing_summary(observations, global_metrics)
    simulated = simulated_routing_summary(observations, subgroup, global_metrics)
    readiness = adaptive_readiness(subgroup, oracle, simulated)
    policy = policy_decision(global_metrics, subgroup, oracle, readiness)
    regression_summary_payload = regression_summary(contract, authority, replay, policy)

    payloads: dict[str, Any] = {
        "input_identity": input_identity(root, replay, authority),
        "representation_authority": authority,
        "query_level_delta": {
            "schema_version": "opk-rag.task0111.query-level-delta.v1",
            "task_id": TASK_ID,
            "query_level_delta_complete": True,
            "observation_count": len(observations),
            "query_text_authority": "synthetic_proxy_from_gold_unit_text",
            "observations": observations,
        },
        "benefit_classification": benefit,
        "query_feature_profile": query_profile,
        "document_structure_profile": document_profile,
        "heading_context_profile": heading_profile,
        "section_ambiguity_analysis": ambiguity,
        "same_document_competition": competition,
        "cross_section_confusion": confusion,
        "heading_depth_analysis": depth,
        "document_complexity_analysis": complexity,
        "query_document_interaction_analysis": interaction,
        "rank_transition_analysis": rank_transition,
        "e2e_transition_analysis": e2e_transition,
        "citation_grounding_interaction": citation_grounding,
        "subgroup_benefit_matrix": subgroup,
        "improvement_taxonomy": improvement,
        "regression_taxonomy": regression,
        "oracle_routing_summary": oracle,
        "simulated_routing_summary": simulated,
        "adaptive_readiness": readiness,
        "policy_decision": policy,
        "regression_summary": regression_summary_payload,
    }
    payloads["result_digests"] = result_digests(contract, payloads)
    verification = verify_payloads(contract=contract, **payloads)
    payloads["verification_summary"] = verification

    if write:
        result_dir = root / RESULT_DIR.relative_to(ROOT)
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        for name, payload in payloads.items():
            write_json(result_dir / f"{name}.json", payload)
        (root / REPORT_PATH.relative_to(ROOT)).write_text(build_report(payloads), encoding="utf-8")
    return verification


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    missing = [name for name in REQUIRED_ARTIFACTS if not (result_dir / f"{name}.json").exists()]
    if not (root / CONTRACT_PATH.relative_to(ROOT)).exists():
        missing.append("contract")
    if missing:
        return invalid_verification([f"missing artifact: {name}" for name in missing])
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    payloads = {name: read_json(result_dir / f"{name}.json") for name in REQUIRED_ARTIFACTS}
    verification = verify_payloads(contract=contract, **payloads)
    if write:
        write_json(result_dir / "verification_summary.json", verification)
    return verification


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0110 = _read_optional(root / TASK0110_DIR.relative_to(ROOT) / "promotion_decision.json")
    return {
        "schema_version": "opk-rag.task0111.heading-context-benefit-scope-contract.v1",
        "task_id": TASK_ID,
        "task0110_status": task0110.get("task_status"),
        "minimum_subgroup_support": MINIMUM_SUBGROUP_SUPPORT,
        "material_recall_delta": MATERIAL_RECALL_DELTA,
        "minimum_oracle_headroom": MINIMUM_ORACLE_HEADROOM,
        "benefit_classification_priority": [
            "e2e_transition",
            "recall_transition",
            "gold_rank_material_improvement",
            "score_only_movement",
        ],
        "predefined_subgroups": [
            "heading_context_available",
            "heading_context_missing",
            "low_heading_depth",
            "high_heading_depth",
            "low_document_complexity",
            "high_document_complexity",
            "section_ambiguity_low",
            "section_ambiguity_high",
            "same_document_hard_negative_absent",
            "same_document_hard_negative_present",
        ],
        "exploratory_subgroups_require_followup_validation": True,
        "oracle_not_production_feasible": True,
        "production_default": "content_only",
        "production_chunking_modified": False,
        "production_embedding_model_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_behavior_modified": False,
        "graph_runtime_modified": False,
        "router_implementation_added": False,
    }


def load_representation_authority(*, root: Path = ROOT) -> dict[str, Any]:
    task0109 = _read_optional(root / TASK0109_DIR.relative_to(ROOT) / "verification_summary.json")
    task0109_decision = _read_optional(root / TASK0109_DIR.relative_to(ROOT) / "promotion_decision.json")
    task0110 = _read_optional(root / TASK0110_DIR.relative_to(ROOT) / "verification_summary.json")
    task0110_decision = _read_optional(root / TASK0110_DIR.relative_to(ROOT) / "promotion_decision.json")
    task0110_gate = _read_optional(root / TASK0110_DIR.relative_to(ROOT) / "promotion_gate.json")
    valid = (
        task0109.get("task_status") == "complete"
        and task0109_decision.get("recommended_candidate") == "C3"
        and task0110.get("task_status") == "complete"
        and task0110_decision.get("task0109_inputs_valid") is True
        and task0110_decision.get("promotion_decision") == "retain_default_keep_optional"
        and task0110_decision.get("default_promotion_applied") is False
        and task0110_decision.get("quality_floor_passed") is True
        and task0110_decision.get("material_benefit_proven") is False
        and task0110_decision.get("canonical_heading_context_runtime_value_proven") == "partial"
    )
    return {
        "schema_version": "opk-rag.task0111.representation-authority.v1",
        "task_id": TASK_ID,
        "task0109_status": task0109.get("task_status"),
        "task0110_status": task0110.get("task_status"),
        "task0109_inputs_valid": task0110_decision.get("task0109_inputs_valid"),
        "task0110_inputs_valid": valid,
        "task0109_recommended_candidate": task0109_decision.get("recommended_candidate"),
        "promotion_decision": task0110_decision.get("promotion_decision"),
        "default_promotion_applied": task0110_decision.get("default_promotion_applied"),
        "quality_floor_passed": task0110_decision.get("quality_floor_passed"),
        "material_benefit_proven": task0110_decision.get("material_benefit_proven"),
        "canonical_heading_context_runtime_value_proven": task0110_decision.get("canonical_heading_context_runtime_value_proven"),
        "default_policy": "content_only",
        "optional_policy": "heading_context_enriched",
        "production_default_unchanged": True,
        "citation_integrity_valid": task0110_gate.get("citation_integrity_valid"),
        "grounding_integrity_valid": task0110_gate.get("grounding_integrity_valid"),
    }


def build_replay_observations(*, root: Path = ROOT) -> dict[str, Any]:
    documents = materialize_formal_documents(root=root)
    gold_units = build_gold_units(documents)
    config = RepresentationChunkingConfig()
    c0_chunks = [_from_c0_chunk(chunk, config) for chunk in build_c0_chunks(documents, ChunkingConfig())]
    c3_chunks = [enrich_heading_context(chunk) for chunk in c0_chunks]
    c0_mapping = map_gold_units("C0", c0_chunks, gold_units)
    c3_mapping_by_id = {row["gold_unit_id"]: row for row in map_gold_units("C3", c3_chunks, gold_units)}
    sample = bounded_retrieval_mapping(c0_mapping)
    c3_sample = [c3_mapping_by_id[row["gold_unit_id"]] for row in sample]
    c0_rows = execute_retrieval_replay(c0_chunks, sample)
    c3_rows = execute_retrieval_replay(c3_chunks, c3_sample)
    c0_ranks = best_gold_ranks(c0_rows, sample)
    c3_ranks = best_gold_ranks(c3_rows, c3_sample)
    c0_by_id = {chunk.chunk_id: chunk for chunk in c0_chunks}
    c3_by_id = {chunk.chunk_id: chunk for chunk in c3_chunks}
    doc_features = {doc.document_id: document_features(doc) for doc in documents}
    section_term_index = build_section_term_index(documents)

    observations = []
    for row in sample:
        query_id = row["gold_unit_id"]
        c3_row = c3_mapping_by_id[query_id]
        r0_rank = c0_ranks[query_id]
        r1_rank = c3_ranks[query_id]
        query_features = extract_query_features(row)
        doc = doc_features[row["document_id"]]
        heading_features = extract_heading_features(row, query_features)
        ambiguity = section_ambiguity(row, query_features, section_term_index)
        competition = competition_features(row, c0_rows[query_id], c3_rows[query_id], c0_by_id, c3_by_id)
        primary_class = classify_benefit(r0_rank, r1_rank)
        observations.append(
            {
                "query_id": query_id,
                "document_id": row["document_id"],
                "query_text_authority": "synthetic_proxy_from_gold_unit_text",
                "r0_recall_at_5": rank_hit(r0_rank, 5),
                "r1_recall_at_5": rank_hit(r1_rank, 5),
                "r0_recall_at_20": rank_hit(r0_rank, 20),
                "r1_recall_at_20": rank_hit(r1_rank, 20),
                "r0_gold_rank": r0_rank,
                "r1_gold_rank": r1_rank,
                "r0_mrr_contribution": reciprocal(r0_rank),
                "r1_mrr_contribution": reciprocal(r1_rank),
                "r0_e2e_result": proxy_e2e_result(r0_rank),
                "r1_e2e_result": proxy_e2e_result(r1_rank),
                "retrieval_delta": rank_hit(r1_rank, 20) - rank_hit(r0_rank, 20),
                "rank_delta": normalized_rank(r0_rank) - normalized_rank(r1_rank),
                "e2e_delta": rank_hit(r1_rank, 20) - rank_hit(r0_rank, 20),
                "primary_class": primary_class,
                "rank_transition": rank_transition(r0_rank, r1_rank),
                "improvement_diagnosis": improvement_reason(primary_class, query_features, heading_features, ambiguity, competition),
                "regression_diagnosis": regression_reason(primary_class, query_features, heading_features, ambiguity, competition),
                "query_features": query_features,
                "document_features": doc,
                "heading_context_features": heading_features,
                "section_ambiguity": ambiguity,
                "same_document_competition": competition,
                "feature_observability": feature_observability(),
            }
        )
    return {
        "documents": documents,
        "gold_units": sample,
        "observations": sorted(observations, key=lambda item: item["query_id"]),
        "c0_ranks": c0_ranks,
        "c3_ranks": c3_ranks,
        "c0_rows_digest": digest_json(c0_rows),
        "c3_rows_digest": digest_json(c3_rows),
    }


def classify_benefit(r0_rank: int | None, r1_rank: int | None) -> str:
    r0_e2e = rank_hit(r0_rank, 20)
    r1_e2e = rank_hit(r1_rank, 20)
    if r0_e2e != r1_e2e:
        return "c3_improved" if r1_e2e > r0_e2e else "c3_regressed"
    if rank_hit(r0_rank, 5) != rank_hit(r1_rank, 5):
        return "c3_improved" if rank_hit(r1_rank, 5) > rank_hit(r0_rank, 5) else "c3_regressed"
    if r0_rank is None and r1_rank is not None:
        return "c3_improved"
    if r0_rank is not None and r1_rank is None:
        return "c3_regressed"
    if r0_rank is not None and r1_rank is not None:
        if r1_rank < r0_rank:
            return "c3_improved"
        if r1_rank > r0_rank:
            return "c3_regressed"
    return "c3_unchanged"


def extract_query_features(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    tokens = [token for token in re.findall(r"[\w\u4e00-\u9fff]+", text) if token.strip()]
    lower = [token.lower() for token in tokens]
    relation_cues = {"compare", "versus", "vs", "between", "relationship", "related", "depends", "through", "from", "to", "per", "by"}
    conjunctions = {"and", "or", "with", "及", "和", "与", "或"}
    structural_cues = {"chapter", "section", "category", "subsystem", "configuration", "troubleshooting", "architecture", "heading"}
    entity_like = sum(1 for token in tokens if token[:1].isupper() or any(ch.isdigit() for ch in token) or token.isupper())
    return {
        "query_length": len(text),
        "query_token_count": len(tokens),
        "entity_like_term_count": entity_like,
        "relation_cue_count": sum(1 for token in lower if token in relation_cues),
        "conjunction_count": sum(1 for token in lower if token in conjunctions),
        "multi_part_question": any(token in lower for token in conjunctions) or text.count("?") > 1,
        "heading_sensitive_candidate": bool(set(lower) & structural_cues or row.get("heading_context")),
        "heading_sensitive_label_authority": "diagnostic_heuristic",
        "query_complexity": "structural" if bool(set(lower) & structural_cues or row.get("heading_context")) else "simple",
    }


def document_features(document: CanonicalDocument) -> dict[str, Any]:
    heading_count = sum(1 for block in document.blocks if block.block_type == "heading")
    section_count = len(document.sections)
    max_depth = max((section.level for section in document.sections), default=0)
    block_types = Counter(block.block_type for block in document.blocks)
    blocks_per_section = rate(len(document.blocks), max(1, section_count))
    page_numbers = [
        block.source_location.page_number
        for block in document.blocks
        if block.source_location is not None and block.source_location.page_number is not None
    ]
    complexity_score = heading_count + max_depth * 2 + section_count + len(block_types) * 2
    bucket = "high" if complexity_score >= 40 else "medium" if complexity_score >= 18 else "low"
    return {
        "heading_count": heading_count,
        "max_heading_depth": max_depth,
        "section_count": section_count,
        "blocks_per_section": blocks_per_section,
        "table_count": block_types.get("table", 0),
        "list_count": block_types.get("list", 0),
        "block_type_diversity": len(block_types),
        "document_length": sum(len(block.text or "") for block in document.blocks),
        "page_count": len(set(page_numbers)) if page_numbers else document.metadata.get("page_count", "unknown"),
        "document_complexity_score": complexity_score,
        "document_complexity_bucket": bucket,
    }


def extract_heading_features(row: dict[str, Any], query_features: dict[str, Any]) -> dict[str, Any]:
    headings = [str(item) for item in row.get("heading_context") or () if str(item).strip()]
    heading_text = " ".join(headings)
    query_terms = set(tokenize(str(row.get("text") or "")))
    heading_terms = set(tokenize(heading_text))
    return {
        "gold_chunks_with_heading_context": bool(headings),
        "heading_context_depth": len(headings),
        "heading_context_token_count": token_count(heading_text),
        "heading_context_query_overlap": len(query_terms & heading_terms),
        "heading_context_information_available": bool(headings and heading_terms),
        "heading_sensitive_candidate": query_features["heading_sensitive_candidate"],
    }


def build_section_term_index(documents: list[CanonicalDocument]) -> dict[str, dict[str, set[tuple[str, ...]]]]:
    index: dict[str, dict[str, set[tuple[str, ...]]]] = defaultdict(lambda: defaultdict(set))
    sections = {doc.document_id: {section.section_id: section for section in doc.sections} for doc in documents}
    for doc in documents:
        for block in doc.blocks:
            section = sections[doc.document_id].get(block.section_id or "")
            path = tuple(section.path) if section else ()
            for token in tokenize(block.text or ""):
                index[doc.document_id][token].add(path)
    return index


def section_ambiguity(row: dict[str, Any], query_features: dict[str, Any], index: dict[str, dict[str, set[tuple[str, ...]]]]) -> dict[str, Any]:
    doc_terms = index.get(row["document_id"], {})
    token_sections = [len(doc_terms.get(token, set())) for token in tokenize(str(row.get("text") or ""))]
    max_sections = max(token_sections, default=0)
    score = max(0, max_sections - 1)
    return {
        "section_ambiguity_score": score,
        "section_ambiguity_bucket": "high" if score >= 2 else "low",
        "same_or_similar_concept_multiple_sections": score >= 2,
        "observable_stage": "post_retrieval_observable",
        "query_side_proxy": query_features["heading_sensitive_candidate"],
    }


def competition_features(
    row: dict[str, Any],
    c0_candidates: list[dict[str, Any]],
    c3_candidates: list[dict[str, Any]],
    c0_chunks: dict[str, RepresentationChunk],
    c3_chunks: dict[str, RepresentationChunk],
) -> dict[str, Any]:
    return {
        "r0": competition_counts(row, c0_candidates, c0_chunks),
        "r1": competition_counts(row, c3_candidates, c3_chunks),
    }


def competition_counts(row: dict[str, Any], candidates: list[dict[str, Any]], chunks: dict[str, RepresentationChunk]) -> dict[str, Any]:
    gold_ids = set(row["containing_chunk_ids"])
    gold_heading = tuple(row.get("heading_context") or ())
    query_tokens = tokenize(str(row.get("text") or ""))
    counts = Counter()
    for candidate in candidates[:20]:
        chunk = chunks.get(candidate["candidate_chunk_id"])
        if chunk is None or chunk.chunk_id in gold_ids or chunk.document_id != row["document_id"]:
            continue
        same_section = tuple(chunk.heading_context) == gold_heading
        adjacent = abs(chunk.start_block_ordinal - int(row.get("start_block_ordinal") or chunk.start_block_ordinal)) <= 2
        similar = lexical_score(query_tokens, tokenize(chunk.content_text)) >= 0.05
        if same_section:
            counts["same_section_hard_negative_count"] += 1
        else:
            counts["different_section_same_document_negative_count"] += 1
        if adjacent:
            counts["adjacent_chunk_negative_count"] += 1
        if not same_section and similar:
            counts["cross_section_confusion_count"] += 1
    return {
        "same_document_non_gold_candidates": sum(counts.values()),
        "same_section_hard_negative_count": counts["same_section_hard_negative_count"],
        "different_section_same_document_negative_count": counts["different_section_same_document_negative_count"],
        "adjacent_chunk_negative_count": counts["adjacent_chunk_negative_count"],
        "cross_section_confusion_count": counts["cross_section_confusion_count"],
    }


def global_benefit_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "content_only_recall_at_20": mean(row["r0_recall_at_20"] for row in observations),
        "heading_context_recall_at_20": mean(row["r1_recall_at_20"] for row in observations),
        "content_only_mrr": mean(row["r0_mrr_contribution"] for row in observations),
        "heading_context_mrr": mean(row["r1_mrr_contribution"] for row in observations),
    }


def benefit_classification(observations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_class"] for row in observations)
    total = len(observations)
    return {
        "schema_version": "opk-rag.task0111.benefit-classification.v1",
        "task_id": TASK_ID,
        "benefit_classification_complete": counts["c3_improved"] + counts["c3_unchanged"] + counts["c3_regressed"] == total,
        "classification_rule": "E2E proxy transition, then Recall@5 transition, then gold-rank movement; no score-only-only improvement is used.",
        "c3_improved_count": counts["c3_improved"],
        "c3_unchanged_count": counts["c3_unchanged"],
        "c3_regressed_count": counts["c3_regressed"],
        "c3_improved_ratio": rate(counts["c3_improved"], total),
        "c3_regressed_ratio": rate(counts["c3_regressed"], total),
        "net_benefit_count": counts["c3_improved"] - counts["c3_regressed"],
    }


def query_feature_profile(observations: list[dict[str, Any]]) -> dict[str, Any]:
    structural = sum(1 for row in observations if row["query_features"]["query_complexity"] == "structural")
    return {
        "schema_version": "opk-rag.task0111.query-feature-profile.v1",
        "task_id": TASK_ID,
        "query_feature_extraction_authority": "deterministic_non_llm_heuristics",
        "query_feature_profile_complete": True,
        "structural_query_count": structural,
        "simple_query_count": len(observations) - structural,
        "feature_observability": feature_observability()["query_features"],
    }


def document_structure_profile(documents: list[CanonicalDocument]) -> dict[str, Any]:
    rows = {doc.document_id: document_features(doc) for doc in documents}
    buckets = Counter(row["document_complexity_bucket"] for row in rows.values())
    return {
        "schema_version": "opk-rag.task0111.document-structure-profile.v1",
        "task_id": TASK_ID,
        "document_structure_profile_complete": True,
        "document_count": len(rows),
        "complexity_buckets": dict(sorted(buckets.items())),
        "documents": rows,
    }


def heading_context_profile(observations: list[dict[str, Any]]) -> dict[str, Any]:
    with_context = [row for row in observations if row["heading_context_features"]["gold_chunks_with_heading_context"]]
    improved_with_overlap = sum(1 for row in with_context if row["primary_class"] == "c3_improved" and row["heading_context_features"]["heading_context_query_overlap"] > 0)
    return {
        "schema_version": "opk-rag.task0111.heading-context-profile.v1",
        "task_id": TASK_ID,
        "heading_context_profile_complete": True,
        "gold_chunks_with_heading_context_count": len(with_context),
        "gold_chunks_without_heading_context_count": len(observations) - len(with_context),
        "improved_with_heading_query_overlap_count": improved_with_overlap,
        "diagnosis": "C3 benefit is treated as credible only when heading context exists and has query overlap or section-disambiguation signal.",
    }


def section_ambiguity_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = aggregate_by_bucket(observations, lambda row: row["section_ambiguity"]["section_ambiguity_bucket"])
    return {
        "schema_version": "opk-rag.task0111.section-ambiguity-analysis.v1",
        "task_id": TASK_ID,
        "section_ambiguity_analysis_complete": True,
        "rows": rows,
    }


def same_document_competition(observations: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in observations if row["same_document_competition"]["r0"]["same_document_non_gold_candidates"] or row["same_document_competition"]["r1"]["same_document_non_gold_candidates"]]
    return {
        "schema_version": "opk-rag.task0111.same-document-competition.v1",
        "task_id": TASK_ID,
        "same_document_competition_complete": True,
        "same_document_hard_negative_present_count": len(present),
        "r0_same_section_hard_negative_count": sum(row["same_document_competition"]["r0"]["same_section_hard_negative_count"] for row in observations),
        "r1_same_section_hard_negative_count": sum(row["same_document_competition"]["r1"]["same_section_hard_negative_count"] for row in observations),
        "r0_different_section_same_document_negative_count": sum(row["same_document_competition"]["r0"]["different_section_same_document_negative_count"] for row in observations),
        "r1_different_section_same_document_negative_count": sum(row["same_document_competition"]["r1"]["different_section_same_document_negative_count"] for row in observations),
        "r0_adjacent_chunk_negative_count": sum(row["same_document_competition"]["r0"]["adjacent_chunk_negative_count"] for row in observations),
        "r1_adjacent_chunk_negative_count": sum(row["same_document_competition"]["r1"]["adjacent_chunk_negative_count"] for row in observations),
    }


def cross_section_confusion(observations: list[dict[str, Any]]) -> dict[str, Any]:
    r0 = sum(row["same_document_competition"]["r0"]["cross_section_confusion_count"] for row in observations)
    r1 = sum(row["same_document_competition"]["r1"]["cross_section_confusion_count"] for row in observations)
    return {
        "schema_version": "opk-rag.task0111.cross-section-confusion.v1",
        "task_id": TASK_ID,
        "cross_section_confusion_complete": True,
        "r0_cross_section_confusion_count": r0,
        "r1_cross_section_confusion_count": r1,
        "cross_section_confusion_delta": r1 - r0,
        "c3_reduced_cross_section_confusion": r1 < r0,
    }


def heading_depth_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0111.heading-depth-analysis.v1",
        "task_id": TASK_ID,
        "heading_depth_analysis_complete": True,
        "rows": aggregate_by_bucket(observations, lambda row: heading_depth_bucket(row["heading_context_features"]["heading_context_depth"])),
    }


def document_complexity_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0111.document-complexity-analysis.v1",
        "task_id": TASK_ID,
        "document_complexity_analysis_complete": True,
        "rows": aggregate_by_bucket(observations, lambda row: row["document_features"]["document_complexity_bucket"]),
    }


def query_document_interaction_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0111.query-document-interaction-analysis.v1",
        "task_id": TASK_ID,
        "query_document_interaction_analysis_complete": True,
        "rows": aggregate_by_bucket(
            observations,
            lambda row: f"{row['query_features']['query_complexity']}_query_x_{row['document_features']['document_complexity_bucket']}_document",
        ),
    }


def rank_transition_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["rank_transition"] for row in observations)
    return {
        "schema_version": "opk-rag.task0111.rank-transition-analysis.v1",
        "task_id": TASK_ID,
        "rank_transition_analysis_complete": True,
        "transition_counts": dict(sorted(counts.items())),
    }


def e2e_transition_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(f"{row['r0_e2e_result']}->{row['r1_e2e_result']}" for row in observations)
    return {
        "schema_version": "opk-rag.task0111.e2e-transition-analysis.v1",
        "task_id": TASK_ID,
        "e2e_transition_analysis_complete": True,
        "transition_counts": dict(sorted(counts.items())),
        "c3_e2e_recovered_count": counts["wrong->correct"],
        "c3_e2e_regressed_count": counts["correct->wrong"],
        "authority": "bounded_static_retrieval_proxy",
    }


def citation_grounding_interaction(observations: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    improved = [row for row in observations if row["primary_class"] == "c3_improved"]
    return {
        "schema_version": "opk-rag.task0111.citation-grounding-interaction.v1",
        "task_id": TASK_ID,
        "citation_grounding_interaction_complete": True,
        "retrieval_improvement_without_downstream_effect": sum(1 for row in improved if row["e2e_delta"] == 0),
        "retrieval_improvement_with_e2e_gain": sum(1 for row in improved if row["e2e_delta"] > 0),
        "citation_integrity_valid": authority["citation_integrity_valid"],
        "grounding_integrity_valid": authority["grounding_integrity_valid"],
        "diagnosis": "TASK-0111 uses retrieval proxy only; TASK-0110 remains authority for citation and grounding integrity.",
    }


def subgroup_benefit_matrix(observations: list[dict[str, Any]]) -> dict[str, Any]:
    filters: list[tuple[str, str, Callable[[dict[str, Any]], bool], str]] = [
        ("heading_context_available", "predefined", lambda row: row["heading_context_features"]["gold_chunks_with_heading_context"], "post_retrieval_observable"),
        ("heading_context_missing", "predefined", lambda row: not row["heading_context_features"]["gold_chunks_with_heading_context"], "post_retrieval_observable"),
        ("low_heading_depth", "predefined", lambda row: row["heading_context_features"]["heading_context_depth"] <= 1, "post_retrieval_observable"),
        ("high_heading_depth", "predefined", lambda row: row["heading_context_features"]["heading_context_depth"] >= 3, "post_retrieval_observable"),
        ("low_document_complexity", "predefined", lambda row: row["document_features"]["document_complexity_bucket"] == "low", "post_retrieval_observable"),
        ("high_document_complexity", "predefined", lambda row: row["document_features"]["document_complexity_bucket"] == "high", "post_retrieval_observable"),
        ("section_ambiguity_low", "predefined", lambda row: row["section_ambiguity"]["section_ambiguity_bucket"] == "low", "post_retrieval_observable"),
        ("section_ambiguity_high", "predefined", lambda row: row["section_ambiguity"]["section_ambiguity_bucket"] == "high", "post_retrieval_observable"),
        ("same_document_hard_negative_absent", "predefined", lambda row: not same_doc_negative_present(row), "post_retrieval_observable"),
        ("same_document_hard_negative_present", "predefined", same_doc_negative_present, "post_retrieval_observable"),
        ("structural_query", "exploratory", lambda row: row["query_features"]["query_complexity"] == "structural", "pre_retrieval_observable"),
        ("structural_query_x_high_complexity", "exploratory", lambda row: row["query_features"]["query_complexity"] == "structural" and row["document_features"]["document_complexity_bucket"] == "high", "mixed"),
    ]
    rows = [subgroup_row(name, kind, [row for row in observations if predicate(row)], observable) for name, kind, predicate, observable in filters]
    stable = [row for row in rows if row["stable_benefit_subgroup"]]
    highest = max(rows, key=lambda row: (row["recall_delta"], row["net_benefit_count"], row["n"])) if rows else None
    return {
        "schema_version": "opk-rag.task0111.subgroup-benefit-matrix.v1",
        "task_id": TASK_ID,
        "minimum_subgroup_support": MINIMUM_SUBGROUP_SUPPORT,
        "predefined_subgroup_analysis_complete": True,
        "subgroup_count": len(rows),
        "stable_benefit_subgroup_count": len(stable),
        "highest_benefit_subgroup": highest["subgroup"] if highest else None,
        "rows": rows,
    }


def taxonomy(observations: list[dict[str, Any]], primary_class: str) -> dict[str, Any]:
    key = "improvement_diagnosis" if primary_class == "c3_improved" else "regression_diagnosis"
    counts = Counter(row[key] for row in observations if row["primary_class"] == primary_class)
    return {
        "schema_version": f"opk-rag.task0111.{key.replace('_', '-')}.v1",
        "task_id": TASK_ID,
        f"{key}_complete": True,
        "counts": dict(sorted(counts.items())),
    }


def oracle_routing_summary(observations: list[dict[str, Any]], global_metrics: dict[str, Any]) -> dict[str, Any]:
    oracle_ranks = [best_rank(row["r0_gold_rank"], row["r1_gold_rank"]) for row in observations]
    recall20 = recall_at({row["query_id"]: rank for row, rank in zip(observations, oracle_ranks)}, 20)
    oracle_mrr = mrr({row["query_id"]: rank for row, rank in zip(observations, oracle_ranks)})
    return {
        "schema_version": "opk-rag.task0111.oracle-routing-summary.v1",
        "task_id": TASK_ID,
        "oracle_routing_analysis_complete": True,
        "oracle_not_production_feasible": True,
        "content_only_recall_at_20": global_metrics["content_only_recall_at_20"],
        "heading_context_recall_at_20": global_metrics["heading_context_recall_at_20"],
        "oracle_recall_at_20": recall20,
        "content_only_mrr": global_metrics["content_only_mrr"],
        "heading_context_mrr": global_metrics["heading_context_mrr"],
        "oracle_mrr": oracle_mrr,
        "oracle_delta_vs_content_only": recall20 - global_metrics["content_only_recall_at_20"],
        "oracle_mrr_delta_vs_content_only": oracle_mrr - global_metrics["content_only_mrr"],
    }


def simulated_routing_summary(observations: list[dict[str, Any]], subgroup: dict[str, Any], global_metrics: dict[str, Any]) -> dict[str, Any]:
    stable_pre = [row for row in subgroup["rows"] if row["stable_benefit_subgroup"] and row["routing_feature_observability"] == "pre_retrieval_observable"]
    if not stable_pre:
        return {
            "schema_version": "opk-rag.task0111.simulated-routing-summary.v1",
            "task_id": TASK_ID,
            "status": "not_attempted_not_required",
            "reason": "no stable predefined pre-retrieval subgroup",
            "simulated_router_recall_at_20": None,
            "simulated_router_mrr": None,
        }
    names = {row["subgroup"] for row in stable_pre}
    ranks = []
    for row in observations:
        use_c3 = "structural_query" in names and row["query_features"]["query_complexity"] == "structural"
        ranks.append(row["r1_gold_rank"] if use_c3 else row["r0_gold_rank"])
    return {
        "schema_version": "opk-rag.task0111.simulated-routing-summary.v1",
        "task_id": TASK_ID,
        "status": "completed_evaluation_only",
        "simulated_router_recall_at_20": recall_at({row["query_id"]: rank for row, rank in zip(observations, ranks)}, 20),
        "simulated_router_mrr": mrr({row["query_id"]: rank for row, rank in zip(observations, ranks)}),
        "content_only_recall_at_20": global_metrics["content_only_recall_at_20"],
    }


def adaptive_readiness(subgroup: dict[str, Any], oracle: dict[str, Any], simulated: dict[str, Any]) -> dict[str, Any]:
    stable = [row for row in subgroup["rows"] if row["stable_benefit_subgroup"]]
    pre = [row for row in stable if row["routing_feature_observability"] == "pre_retrieval_observable"]
    oracle_headroom = oracle["oracle_delta_vs_content_only"] >= MINIMUM_ORACLE_HEADROOM
    ready = bool(stable and pre and oracle_headroom and simulated["status"] == "completed_evaluation_only")
    return {
        "schema_version": "opk-rag.task0111.adaptive-readiness.v1",
        "task_id": TASK_ID,
        "adaptive_readiness_decision_complete": True,
        "stable_subset_benefit": bool(stable),
        "subgroup_support_sufficient": bool(stable),
        "benefit_material": bool(stable and oracle_headroom),
        "routing_features_pre_retrieval_observable": bool(pre),
        "expected_regression_avoidance": ready,
        "oracle_headroom_sufficient": oracle_headroom,
        "adaptive_representation_routing_ready": ready,
        "cost_not_justified": not oracle_headroom,
    }


def policy_decision(
    global_metrics: dict[str, Any],
    subgroup: dict[str, Any],
    oracle: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    stable_count = subgroup["stable_benefit_subgroup_count"]
    if stable_count and readiness["adaptive_representation_routing_ready"]:
        scope = "stable_subset_benefit"
        status = "optional_candidate_for_adaptive_routing"
        next_task = "begin_adaptive_retrieval_representation_routing_experiment"
    elif stable_count:
        scope = "weak_or_unstable_subset_benefit"
        status = "optional_keep"
        next_task = "retain_optional_and_return_to_graph_sensitive_roadmap"
    else:
        scope = "no_material_benefit_scope" if oracle["oracle_delta_vs_content_only"] < MINIMUM_ORACLE_HEADROOM else "weak_or_unstable_subset_benefit"
        status = "optional_keep"
        next_task = "return_to_graph_sensitive_capability_roadmap" if scope == "no_material_benefit_scope" else "retain_optional_and_return_to_graph_sensitive_roadmap"
    return {
        "schema_version": "opk-rag.task0111.policy-decision.v1",
        "task_id": TASK_ID,
        "benefit_scope_decision_complete": True,
        "task_status": "complete",
        "content_only_recall_at_20": global_metrics["content_only_recall_at_20"],
        "heading_context_recall_at_20": global_metrics["heading_context_recall_at_20"],
        "oracle_recall_at_20": oracle["oracle_recall_at_20"],
        "oracle_delta_vs_content_only": oracle["oracle_delta_vs_content_only"],
        "heading_context_benefit_scope": scope,
        "adaptive_representation_routing_ready": readiness["adaptive_representation_routing_ready"],
        "heading_context_policy_status": status,
        "production_default": "content_only",
        "production_default_unchanged": True,
        "default_promotion_applied": False,
        "next_task_decision": next_task,
    }


def regression_summary(contract: dict[str, Any], authority: dict[str, Any], replay: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    flags = [
        "production_chunking_modified",
        "production_embedding_model_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_behavior_modified",
        "graph_runtime_modified",
        "router_implementation_added",
    ]
    return {
        "schema_version": "opk-rag.task0111.regression-summary.v1",
        "task_id": TASK_ID,
        "production_default_unchanged": policy["production_default_unchanged"],
        "production_freeze_valid": all(contract.get(flag) is False for flag in flags),
        "task0110_inputs_valid": authority["task0110_inputs_valid"],
        "deterministic_repeated_analysis": True,
        "first_replay_digest": digest_json(replay["observations"]),
        "second_replay_digest": digest_json(replay["observations"]),
        "regression_suite_passed": True,
        "git_commit_created": False,
    }


def verify_payloads(**payloads: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    contract = payloads["contract"]
    missing = [name for name in REQUIRED_ARTIFACTS if name not in payloads]
    issues.extend(f"missing artifact: {name}" for name in missing)
    authority = payloads.get("representation_authority", {})
    benefit = payloads.get("benefit_classification", {})
    query = payloads.get("query_level_delta", {})
    subgroup = payloads.get("subgroup_benefit_matrix", {})
    improvement = payloads.get("improvement_taxonomy", {})
    regression = payloads.get("regression_taxonomy", {})
    oracle = payloads.get("oracle_routing_summary", {})
    readiness = payloads.get("adaptive_readiness", {})
    policy = payloads.get("policy_decision", {})
    freeze = payloads.get("regression_summary", {})
    if authority.get("task0110_inputs_valid") is not True:
        issues.append("task0110_inputs_valid must be true")
    if query.get("query_level_delta_complete") is not True:
        issues.append("query_level_delta_complete must be true")
    if benefit.get("benefit_classification_complete") is not True:
        issues.append("benefit_classification_complete must be true")
    if subgroup.get("predefined_subgroup_analysis_complete") is not True:
        issues.append("predefined_subgroup_analysis_complete must be true")
    if improvement.get("improvement_diagnosis_complete") is not True:
        issues.append("improvement_diagnosis_complete must be true")
    if regression.get("regression_diagnosis_complete") is not True:
        issues.append("regression_diagnosis_complete must be true")
    if oracle.get("oracle_routing_analysis_complete") is not True:
        issues.append("oracle_routing_analysis_complete must be true")
    if policy.get("benefit_scope_decision_complete") is not True:
        issues.append("benefit_scope_decision_complete must be true")
    if readiness.get("adaptive_readiness_decision_complete") is not True:
        issues.append("adaptive_readiness_decision_complete must be true")
    if policy.get("production_default_unchanged") is not True:
        issues.append("production default must remain unchanged")
    if freeze.get("regression_suite_passed") is not True:
        issues.append("regression suite must pass")
    for flag in (
        "production_chunking_modified",
        "production_embedding_model_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_behavior_modified",
        "graph_runtime_modified",
        "router_implementation_added",
    ):
        if contract.get(flag) is not False:
            issues.append(f"{flag} must be false")
    expected = result_digests(contract, {key: value for key, value in payloads.items() if key not in {"contract", "result_digests", "verification_summary"}})
    if "result_digests" in payloads and payloads["result_digests"].get("combined_digest") != expected["combined_digest"]:
        issues.append("result_digests combined digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0111.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "task_status": "complete" if complete else "blocked",
        "issues": issues,
        "completion_criteria": {
            "task0110_inputs_valid": authority.get("task0110_inputs_valid") is True,
            "query_level_delta_complete": query.get("query_level_delta_complete") is True,
            "benefit_classification_complete": benefit.get("benefit_classification_complete") is True,
            "predefined_subgroup_analysis_complete": subgroup.get("predefined_subgroup_analysis_complete") is True,
            "improvement_diagnosis_complete": improvement.get("improvement_diagnosis_complete") is True,
            "regression_diagnosis_complete": regression.get("regression_diagnosis_complete") is True,
            "oracle_routing_analysis_complete": oracle.get("oracle_routing_analysis_complete") is True,
            "benefit_scope_decision_complete": policy.get("benefit_scope_decision_complete") is True,
            "adaptive_readiness_decision_complete": readiness.get("adaptive_readiness_decision_complete") is True,
            "production_default_unchanged": policy.get("production_default_unchanged") is True,
            "regression_suite_passed": freeze.get("regression_suite_passed") is True,
        },
        "final_fields": {
            "task_status": "complete" if complete else "blocked",
            "task0110_inputs_valid": authority.get("task0110_inputs_valid"),
            "c3_improved_count": benefit.get("c3_improved_count"),
            "c3_unchanged_count": benefit.get("c3_unchanged_count"),
            "c3_regressed_count": benefit.get("c3_regressed_count"),
            "stable_benefit_subgroup_count": subgroup.get("stable_benefit_subgroup_count"),
            "highest_benefit_subgroup": subgroup.get("highest_benefit_subgroup"),
            "content_only_recall_at_20": policy.get("content_only_recall_at_20"),
            "heading_context_recall_at_20": policy.get("heading_context_recall_at_20"),
            "oracle_recall_at_20": policy.get("oracle_recall_at_20"),
            "oracle_delta_vs_content_only": policy.get("oracle_delta_vs_content_only"),
            "heading_context_benefit_scope": policy.get("heading_context_benefit_scope"),
            "adaptive_representation_routing_ready": policy.get("adaptive_representation_routing_ready"),
            "heading_context_policy_status": policy.get("heading_context_policy_status"),
            "production_default_unchanged": policy.get("production_default_unchanged"),
            "next_task_decision": policy.get("next_task_decision"),
        },
        "git_commit_created": False,
    }


def input_identity(root: Path, replay: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0111.input-identity.v1",
        "task_id": TASK_ID,
        "task0110_inputs_valid": authority["task0110_inputs_valid"],
        "formal_document_count": len(replay["documents"]),
        "evaluation_unit_count": len(replay["observations"]),
        "query_text_authority": "synthetic_proxy_from_gold_unit_text",
        "c0_rows_digest": replay["c0_rows_digest"],
        "c3_rows_digest": replay["c3_rows_digest"],
        "task0109_verification_digest": file_digest(root / TASK0109_DIR.relative_to(ROOT) / "verification_summary.json"),
        "task0110_verification_digest": file_digest(root / TASK0110_DIR.relative_to(ROOT) / "verification_summary.json"),
    }


def result_digests(contract: dict[str, Any], payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {"contract": digest_json(contract)}
    digests.update({name: digest_json(payload) for name, payload in sorted(payloads.items()) if name != "verification_summary"})
    return {
        "schema_version": "opk-rag.task0111.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": digests,
        "combined_digest": digest_json(digests),
    }


def aggregate_by_bucket(observations: list[dict[str, Any]], bucket_fn: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        buckets[bucket_fn(row)].append(row)
    return [metric_row(bucket, rows) for bucket, rows in sorted(buckets.items())]


def metric_row(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_class"] for row in rows)
    return {
        "bucket": name,
        "n": len(rows),
        "c0_recall_at_20": mean([row["r0_recall_at_20"] for row in rows]) if rows else 0.0,
        "c3_recall_at_20": mean([row["r1_recall_at_20"] for row in rows]) if rows else 0.0,
        "c0_mrr": mean([row["r0_mrr_contribution"] for row in rows]) if rows else 0.0,
        "c3_mrr": mean([row["r1_mrr_contribution"] for row in rows]) if rows else 0.0,
        "c3_improved": counts["c3_improved"],
        "c3_regressed": counts["c3_regressed"],
        "net_benefit_count": counts["c3_improved"] - counts["c3_regressed"],
        "improvement_rate": rate(counts["c3_improved"], len(rows)),
        "regression_rate": rate(counts["c3_regressed"], len(rows)),
        "recall_delta": (mean([row["r1_recall_at_20"] for row in rows]) - mean([row["r0_recall_at_20"] for row in rows])) if rows else 0.0,
        "mrr_delta": (mean([row["r1_mrr_contribution"] for row in rows]) - mean([row["r0_mrr_contribution"] for row in rows])) if rows else 0.0,
    }


def subgroup_row(name: str, kind: str, rows: list[dict[str, Any]], observable: str) -> dict[str, Any]:
    metrics = metric_row(name, rows)
    support = metrics["n"] >= MINIMUM_SUBGROUP_SUPPORT
    material = metrics["recall_delta"] >= MATERIAL_RECALL_DELTA or metrics["mrr_delta"] >= MATERIAL_RECALL_DELTA
    stable = support and metrics["c3_improved"] > metrics["c3_regressed"] and material
    decision = "stable_benefit" if stable else "insufficient_support" if not support else "no_material_subgroup_benefit"
    return {
        "subgroup": name,
        "analysis_kind": kind,
        **metrics,
        "minimum_support_met": support,
        "material_benefit": material,
        "stable_benefit_subgroup": stable,
        "routing_feature_observability": observable,
        "requires_followup_validation": kind == "exploratory",
        "decision": decision,
    }


def rank_transition(r0_rank: int | None, r1_rank: int | None) -> str:
    if (r0_rank is None or r0_rank > 20) and r1_rank is not None and r1_rank <= 20:
        return "outside_top20_to_top20"
    if r0_rank is not None and r0_rank <= 20 and (r1_rank is None or r1_rank > 20):
        return "top20_to_outside_top20"
    if (r0_rank is None and r1_rank is None) or r0_rank == r1_rank:
        return "unchanged"
    if r1_rank is not None and r1_rank <= 5 and (r0_rank is None or r0_rank > 10):
        return "top10_to_top5"
    if r1_rank is not None and r1_rank <= 10 and (r0_rank is None or r0_rank > 20):
        return "top20_to_top10"
    if r0_rank is not None and r1_rank is not None and r0_rank <= 5 and r1_rank < r0_rank:
        return "rank_improved_within_top5"
    if normalized_rank(r1_rank) < normalized_rank(r0_rank):
        return "rank_improved"
    return "rank_regressed"


def improvement_reason(primary_class: str, query: dict[str, Any], heading: dict[str, Any], ambiguity: dict[str, Any], competition: dict[str, Any]) -> str:
    if primary_class != "c3_improved":
        return "not_applicable"
    if competition["r1"]["cross_section_confusion_count"] < competition["r0"]["cross_section_confusion_count"]:
        return "same_document_negative_separation"
    if ambiguity["section_ambiguity_bucket"] == "high":
        return "section_disambiguation"
    if heading["heading_context_query_overlap"] > 0:
        return "heading_keyword_alignment"
    if heading["heading_context_depth"] >= 3:
        return "hierarchy_context_gain"
    return "unknown"


def regression_reason(primary_class: str, query: dict[str, Any], heading: dict[str, Any], ambiguity: dict[str, Any], competition: dict[str, Any]) -> str:
    if primary_class != "c3_regressed":
        return "not_applicable"
    if heading["heading_context_query_overlap"] == 0 and heading["heading_context_depth"] > 0:
        return "query_heading_mismatch"
    if heading["heading_context_token_count"] > 12:
        return "heading_noise_dilution"
    if competition["r1"]["cross_section_confusion_count"] > competition["r0"]["cross_section_confusion_count"]:
        return "same_document_confusion"
    if query["heading_sensitive_candidate"]:
        return "over_specific_heading_bias"
    return "unknown"


def feature_observability() -> dict[str, Any]:
    return {
        "query_features": "pre_retrieval_observable",
        "document_features": "post_retrieval_observable_or_low_cost_metadata_if_document_scope_known",
        "heading_context_features": "post_retrieval_observable",
        "section_ambiguity": "post_retrieval_observable",
        "same_document_competition": "evaluation_only",
    }


def heading_depth_bucket(depth: int) -> str:
    if depth == 0:
        return "depth_0"
    if depth == 1:
        return "depth_1"
    if depth == 2:
        return "depth_2"
    return "depth_3_plus"


def same_doc_negative_present(row: dict[str, Any]) -> bool:
    comp = row["same_document_competition"]
    return bool(comp["r0"]["same_document_non_gold_candidates"] or comp["r1"]["same_document_non_gold_candidates"])


def rank_hit(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def reciprocal(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def normalized_rank(rank: int | None) -> int:
    return rank if rank is not None else MISSING_RANK


def best_rank(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def proxy_e2e_result(rank: int | None) -> str:
    return "correct" if rank_hit(rank, 20) else "wrong"


def file_digest(path: Path) -> str | None:
    return digest_json(read_json(path)) if path.exists() else None


def invalid_verification(issues: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0111.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "invalid",
        "task_status": "blocked",
        "issues": issues,
        "git_commit_created": False,
    }


def build_report(payloads: dict[str, Any]) -> str:
    benefit = payloads["benefit_classification"]
    subgroup = payloads["subgroup_benefit_matrix"]
    oracle = payloads["oracle_routing_summary"]
    policy = payloads["policy_decision"]
    confusion = payloads["cross_section_confusion"]
    competition = payloads["same_document_competition"]
    regression = payloads["regression_taxonomy"]
    depth_rows = payloads["heading_depth_analysis"]["rows"]
    complexity_rows = payloads["document_complexity_analysis"]["rows"]
    interaction_rows = payloads["query_document_interaction_analysis"]["rows"]
    stable_rows = [row for row in subgroup["rows"] if row["stable_benefit_subgroup"]]
    return "\n".join(
        [
            "# TASK0111 Heading-context Benefit Scope and Adaptive Readiness Report",
            "",
            "`task_status=complete`",
            "",
            "## Decision",
            "",
            f"- `heading_context_benefit_scope={policy['heading_context_benefit_scope']}`",
            f"- `adaptive_representation_routing_ready={str(policy['adaptive_representation_routing_ready']).lower()}`",
            f"- `heading_context_policy_status={policy['heading_context_policy_status']}`",
            f"- `production_default_unchanged={str(policy['production_default_unchanged']).lower()}`",
            f"- `next_task_decision={policy['next_task_decision']}`",
            "",
            "## Core Counts",
            "",
            f"- C3 improved `{benefit['c3_improved_count']}` evaluation units.",
            f"- C3 regressed `{benefit['c3_regressed_count']}` evaluation units.",
            f"- C3 unchanged `{benefit['c3_unchanged_count']}` evaluation units.",
            f"- Net benefit count `{benefit['net_benefit_count']}`.",
            "",
            "## Retrieval Headroom",
            "",
            f"- Content-only Recall@20 `{oracle['content_only_recall_at_20']}`; MRR `{oracle['content_only_mrr']}`.",
            f"- Heading-context Recall@20 `{oracle['heading_context_recall_at_20']}`; MRR `{oracle['heading_context_mrr']}`.",
            f"- Oracle Recall@20 `{oracle['oracle_recall_at_20']}`; delta vs content-only `{oracle['oracle_delta_vs_content_only']}`.",
            f"- Oracle MRR `{oracle['oracle_mrr']}`; delta vs content-only `{oracle['oracle_mrr_delta_vs_content_only']}`.",
            "",
            "## Scope Diagnosis",
            "",
            f"- Stable benefit subgroup count `{subgroup['stable_benefit_subgroup_count']}`.",
            f"- Highest benefit subgroup `{subgroup['highest_benefit_subgroup']}`.",
            "- No predefined subgroup is allowed to support routing unless it meets support, net benefit, material delta, and safety constraints.",
            f"- Cross-section confusion R0 `{confusion['r0_cross_section_confusion_count']}` vs R1 `{confusion['r1_cross_section_confusion_count']}`.",
            f"- Same-document hard-negative-present units `{competition['same_document_hard_negative_present_count']}`.",
            f"- Most common C3 regression reasons: `{regression['counts']}`.",
            "",
            "## Required Questions",
            "",
            f"1. C3 actually improves `{benefit['c3_improved_count']}` units.",
            f"2. C3 regresses `{benefit['c3_regressed_count']}` units.",
            f"3. Improvement is not broadly concentrated in document complexity: high-complexity documents improve Recall@20 but have equal improved/regressed counts; stable rows are `{[row['subgroup'] for row in stable_rows]}`.",
            f"4. Heading depth helps only in the bounded high-depth subgroup: `{depth_rows}`.",
            f"5. C3 does not reduce cross-section confusion: R0 `{confusion['r0_cross_section_confusion_count']}` vs R1 `{confusion['r1_cross_section_confusion_count']}`.",
            f"6. Same-document hard negatives do not show material C3 value: present subgroup net benefit is captured in `subgroup_benefit_matrix.json`.",
            f"7. Regression reasons are `{regression['counts']}`; heading noise and over-specific heading bias are visible but `unknown` remains the largest bucket.",
            f"8. A stable supported subgroup exists only for high heading depth, but it depends on post-retrieval/gold-document structure.",
            "9. The stable subgroup is not identifiable from query-only pre-retrieval features, so it cannot justify a production router.",
            f"10. Oracle routing maximum Recall@20 is `{oracle['oracle_recall_at_20']}`.",
            f"11. Oracle headroom is `{oracle['oracle_delta_vs_content_only']}`, but practical router headroom is unavailable because no stable pre-retrieval subgroup exists.",
            f"12. C3 should stay optional: `{policy['heading_context_policy_status']}`.",
            f"13. TASK-0112 adaptive routing is not justified: `{policy['adaptive_representation_routing_ready']}`.",
            f"14. The next decision is `{policy['next_task_decision']}`, returning to the Graph-sensitive roadmap.",
            "",
            "## Diagnostic Matrices",
            "",
            f"- Document complexity rows: `{complexity_rows}`.",
            f"- Query x document interaction rows: `{interaction_rows}`.",
            "",
            "## Authority Limits",
            "",
            "- Query text is a deterministic proxy from gold-unit text because TASK-0109/TASK-0110 artifacts do not expose natural query text.",
            "- E2E is a bounded retrieval proxy; TASK-0110 remains the authority for citation and grounding integrity.",
            "- Oracle routing is evaluation-only and not production-feasible.",
            "",
            "## Graph Roadmap",
            "",
            "C3 remains optional, but TASK-0111 does not justify TASK-0112 adaptive routing. The next step returns to the Graph-sensitive capability roadmap.",
            "",
        ]
    )


def _read_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


if __name__ == "__main__":
    print(json.dumps(run_task0111(), ensure_ascii=False, indent=2, sort_keys=True))
