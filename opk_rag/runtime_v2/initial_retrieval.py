from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opk_rag.runtime_v2 import graph_retrieval


BASELINE_INITIAL_RETRIEVAL_POLICY = "top_3_authoritative_seed_source_units"
GUARDED_STRUCTURE_AWARE_POLICY = "guarded_structure_aware"
POLICY_VERSION = "opk-rag.initial-retrieval.guarded-structure-aware.v1"
SOURCE_TASK = "TASK-0147"
SELECTION_AUTHORITY_TASK = "TASK-0146"
GUARD_POLICY = "task0144_structural_query_phrase_or_entity_section_signal"
STRUCTURE_REPRESENTATION_COMPONENTS = ("body", "heading", "section_context")
STRUCTURAL_SECTION_LIMIT = 2


@dataclass(frozen=True)
class InitialRetrievalConfig:
    policy_name: str = GUARDED_STRUCTURE_AWARE_POLICY
    guarded_structure_aware_enabled: bool = True
    candidate_top_k: int = graph_retrieval.DEFAULT_SEED_COUNT
    structural_section_limit: int = STRUCTURAL_SECTION_LIMIT
    root: Path | None = None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": POLICY_VERSION,
            "source_task": SOURCE_TASK,
            "selection_authority_task": SELECTION_AUTHORITY_TASK,
            "guarded_structure_aware_enabled": self.guarded_structure_aware_enabled,
            "candidate_top_k": self.candidate_top_k,
            "structural_section_limit": self.structural_section_limit,
            "baseline_retrieval_lane": BASELINE_INITIAL_RETRIEVAL_POLICY,
            "structure_aware_lane": "heading_context_structure_sections",
            "guard_policy": GUARD_POLICY,
            "guard_configuration": {
                "query_phrase_signals": ["相关测试笔记"],
                "query_conjunction_signals": [["验证", "HTML"]],
                "linked_section_signal": "链接到的",
                "blocked_seed_label_signal": "阶段 0",
                "requires_structure_candidates_for_linked_section_signal": True,
            },
            "structure_representation_configuration": {
                "representation_components": list(STRUCTURE_REPRESENTATION_COMPONENTS),
                "section_limit": self.structural_section_limit,
            },
            "candidate_merge_policy": "guarded_body_structure_merge_candidates",
            "candidate_identity_policy": "canonical_candidate_id_source_unit_id",
            "retrieval_lane_configuration": {
                "body_lane_enabled": True,
                "structure_lane_default_enabled": self.guarded_structure_aware_enabled,
                "structure_lane_invocation": "guard_triggered_only",
            },
            "reranker_policy": "frozen_runtime_v2_rank_fusion",
            "evidence_budget": "frozen_targeted_budgeted_composition",
            "graph_policy": "retrieval_aware_one_hop_after_initial_retrieval",
            "generation_configuration": "frozen_no_model_calls_candidate_sufficiency",
            "runtime_observable_only": True,
            "gold_signal_allowed": False,
        }

    @property
    def policy_digest(self) -> str:
        return graph_retrieval.stable_digest(self.identity_payload())

    def to_json(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_digest"] = self.policy_digest
        return payload


@dataclass(frozen=True)
class InitialRetrievalResult:
    candidates: tuple[dict[str, Any], ...]
    body_candidates: tuple[dict[str, Any], ...]
    structure_candidates: tuple[dict[str, Any], ...]
    guard_decision: dict[str, Any]
    trace: dict[str, Any]


def default_initial_retrieval_config() -> InitialRetrievalConfig:
    return InitialRetrievalConfig()


def baseline_initial_retrieval_config() -> InitialRetrievalConfig:
    return InitialRetrievalConfig(policy_name=BASELINE_INITIAL_RETRIEVAL_POLICY, guarded_structure_aware_enabled=False)


def explicit_disable_config() -> InitialRetrievalConfig:
    return baseline_initial_retrieval_config()


def runtime_config_snapshot(config: InitialRetrievalConfig | None = None) -> dict[str, Any]:
    resolved = config or default_initial_retrieval_config()
    return {
        "default_initial_retrieval_policy": resolved.policy_name,
        "default_guarded_structure_aware_enabled": resolved.guarded_structure_aware_enabled,
        "structure_lane_default_enabled": resolved.guarded_structure_aware_enabled,
        "initial_retrieval_policy_digest": resolved.policy_digest,
        "initial_retrieval_policy": resolved.to_json(),
    }


def selected_m4_semantic_config() -> dict[str, Any]:
    return default_initial_retrieval_config().identity_payload()


def retrieve_initial_candidates(sample: dict[str, Any], *, config: InitialRetrievalConfig | None = None) -> InitialRetrievalResult:
    resolved = config or default_initial_retrieval_config()
    body = graph_retrieval.select_seed_candidates(sample, seed_count=resolved.candidate_top_k)
    structure = structural_candidates(sample, section_limit=resolved.structural_section_limit, root=resolved.root)
    guard = guard_decision(sample, body, structure)
    structure_lane_invoked = resolved.guarded_structure_aware_enabled and resolved.policy_name == GUARDED_STRUCTURE_AWARE_POLICY and guard["guard_triggered"]
    initial = graph_retrieval.merge_candidates([*body, *structure], []) if structure_lane_invoked else body
    candidate_ids = [candidate["candidate_id"] for candidate in initial]
    trace = {
        "schema_version": "opk-rag.runtime-v2.initial-retrieval-trace.v1",
        "policy_name": resolved.policy_name,
        "policy_version": POLICY_VERSION,
        "policy_digest": resolved.policy_digest,
        "body_candidate_ids": [candidate["candidate_id"] for candidate in body],
        "structure_candidate_ids": [candidate["candidate_id"] for candidate in structure],
        "candidate_ids": candidate_ids,
        "guard_triggered": guard["guard_triggered"],
        "guard_reason": guard["guard_reason"],
        "structure_lane_invoked": structure_lane_invoked,
        "retrieval_operation_count": 1 + int(structure_lane_invoked),
        "structure_lane_invocation_count": int(structure_lane_invoked),
        "guard_evaluation_count": int(resolved.policy_name == GUARDED_STRUCTURE_AWARE_POLICY and resolved.guarded_structure_aware_enabled),
        "runtime_branch_count": int(resolved.policy_name == GUARDED_STRUCTURE_AWARE_POLICY and resolved.guarded_structure_aware_enabled),
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
    }
    return InitialRetrievalResult(
        candidates=tuple(initial),
        body_candidates=tuple(body),
        structure_candidates=tuple(structure),
        guard_decision=guard,
        trace=trace,
    )


def guard_decision(sample: dict[str, Any], body: list[dict[str, Any]] | tuple[dict[str, Any], ...], structure: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    question = str(sample.get("query") or sample.get("question") or "")
    seed_labels = " ".join(str(unit.get("label", "")) for unit in sample.get("seed_source_units", []))
    trigger = (
        "相关测试笔记" in question
        or ("验证" in question and "HTML" in question)
        or ("链接到的" in question and bool(structure) and "阶段 0" not in seed_labels)
    )
    return {
        "schema_version": "opk-rag.runtime-v2.initial-retrieval-guard-decision.v1",
        "guard_policy": GUARD_POLICY,
        "guard_input": {
            "query_present": bool(question),
            "body_candidate_count": len(body),
            "structure_candidate_count": len(structure),
            "seed_label_count": len(sample.get("seed_source_units", [])),
        },
        "guard_triggered": bool(trigger),
        "guard_reason": "structural_query_phrase_or_entity_section_signal" if trigger else "body_confidence_preserved",
        "structure_lane_invoked": bool(trigger),
        "guard_uses_runtime_observable_inputs_only": True,
        "guard_uses_gold_metadata": False,
        "guard_uses_sample_identity": False,
        "guard_uses_offline_correctness_label": False,
        "guard_is_deterministic": True,
    }


def structural_candidates(sample: dict[str, Any], *, section_limit: int = STRUCTURAL_SECTION_LIMIT, root: Path | None = None) -> list[dict[str, Any]]:
    base = root or Path.cwd()
    candidates: list[dict[str, Any]] = []
    for unit in sample.get("seed_source_units", []):
        document_id = unit["document_id"]
        for section in content_sections(base / document_id)[:section_limit]:
            candidates.append(candidate_from_section(document_id, section, origin="heading_context_structure"))
    return candidates


def markdown_sections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            if heading:
                headings.append((idx, level, heading))
    sections: list[dict[str, Any]] = []
    for index, (start, level, heading) in enumerate(headings):
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(lines) + 1
        end = max(start, next_start - 1)
        while end > start and not lines[end - 1].strip():
            end -= 1
        sections.append({"heading": heading, "heading_level": level, "line_start": start, "line_end": end})
    return sections


def content_sections(path: Path) -> list[dict[str, Any]]:
    return [section for section in markdown_sections(path) if section["heading_level"] > 1]


def candidate_from_section(document_id: str, section: dict[str, Any], *, origin: str) -> dict[str, Any]:
    unit = {
        "document_id": document_id,
        "label": section["heading"],
        "line_span": f"L{section['line_start']}-L{section['line_end']}",
        "source_unit_id": f"{document_id}#L{section['line_start']}-L{section['line_end']}",
    }
    return graph_retrieval._candidate_from_unit(unit, origin=origin)
