from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


ANSWERING_ACTIONS = {"answer", "partial_answer", "correct_premise"}

FORBIDDEN_RUNTIME_PROPOSAL_FIELDS = {
    "benchmark_correctness",
    "expected_action",
    "forbidden_claims",
    "gold_answerability",
    "gold_evidence",
    "gold_evidence_identities",
    "required_claims",
    "sample_id",
    "stable_candidate_id",
    "task0070_candidate_id",
    "task0075_candidate_id",
}

RAW_PRIVATE_ARTIFACT_FIELDS = {
    "answer",
    "answer_text",
    "content",
    "evidence_text",
    "full_response",
    "prompt",
    "provider_request",
    "provider_response",
    "raw_evidence",
    "raw_prompt",
    "raw_provider_response",
    "retrieved_chunk",
    "text",
}


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "ratio": ratio(numerator, denominator)}


def final_action_correct(expected_action: str, final_action: str) -> bool:
    if expected_action in ANSWERING_ACTIONS:
        return final_action == "answer"
    return final_action == "abstain"


def safe_action_correct(expected_action: str, final_action: str) -> bool:
    return not (expected_action == "abstain" and final_action == "answer")


def collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(collect_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(collect_keys(child))
    return keys


def validate_runtime_deployable_rule(rule: Mapping[str, Any]) -> None:
    present = sorted(FORBIDDEN_RUNTIME_PROPOSAL_FIELDS & collect_keys(rule))
    if present:
        raise ValueError(f"runtime proposal contains forbidden fields: {present}")


def reject_raw_private_fields(value: Any) -> list[str]:
    return sorted(RAW_PRIVATE_ARTIFACT_FIELDS & collect_keys(value))


def counter(items: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
