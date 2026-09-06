from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

TASK_ID = "TASK-0177"
MACHINE_READABLE_OUTPUT_SCHEMA_VERSION = 1
SEARCH_RESPONSE_SCHEMA_ID = "opk-rag.search-response.v1"


class MachineReadableOutputError(ValueError):
    """Raised when a subprocess violates the machine-readable output contract."""


@dataclass(frozen=True, slots=True)
class ParsedMachineReadableOutput:
    payload: dict[str, Any]
    observed_stdout_format: str


def with_search_output_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a search JSON payload with the minimal stable machine-readable contract marker."""

    result = dict(payload)
    result.setdefault("schema_version", SEARCH_RESPONSE_SCHEMA_ID)
    result.setdefault("machine_readable_output_schema_version", MACHINE_READABLE_OUTPUT_SCHEMA_VERSION)
    return result


def classify_stdout_format(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        return "empty_stdout"
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            json_line_count = 0
            for line in lines:
                try:
                    json.loads(line)
                    json_line_count += 1
                except json.JSONDecodeError:
                    pass
            if json_line_count == len(lines):
                return "jsonl"
            if json_line_count:
                first_json = _line_is_json(lines[0])
                last_json = _line_is_json(lines[-1])
                if first_json and not last_json:
                    return "json_plus_log_text"
                if last_json and not first_json:
                    return "log_text_plus_json"
                return "mixed_stdout"
        if exc.msg == "Extra data":
            return "multiple_json_objects"
        return "malformed_json"
    if isinstance(value, dict):
        return "single_json_object"
    if isinstance(value, list):
        return "single_json_array"
    return "malformed_json"


def parse_single_json_stdout(*, exit_code: int, stdout: str, stderr: str = "") -> ParsedMachineReadableOutput:
    """Fail-closed parser for the TASK-0177 canonical contract.

    Contract: exit_code=0 and stdout is exactly one JSON object. Diagnostics belong on stderr.
    """

    if exit_code != 0:
        raise MachineReadableOutputError("subprocess_failure_before_parse")
    observed = classify_stdout_format(stdout)
    if observed != "single_json_object":
        raise MachineReadableOutputError(f"invalid_stdout_format:{observed}")
    payload = json.loads(stdout.strip())
    version = payload.get("machine_readable_output_schema_version", MACHINE_READABLE_OUTPUT_SCHEMA_VERSION)
    if version != MACHINE_READABLE_OUTPUT_SCHEMA_VERSION:
        raise MachineReadableOutputError("unsupported_output_schema_version")
    return ParsedMachineReadableOutput(payload=payload, observed_stdout_format=observed)


def candidate_semantic_signature(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results") or []
    signature = []
    for row in rows:
        signature.append(
            {
                "rank": row.get("rank"),
                "document_id": row.get("document_id"),
                "chunk_id": row.get("chunk_id"),
                "score": row.get("similarity"),
                "vector_similarity": row.get("vector_similarity"),
                "bm25_score": row.get("bm25_score"),
                "rrf_score": row.get("rrf_score"),
                "retrieval_sources": list(row.get("retrieval_sources") or []),
            }
        )
    return signature


def _line_is_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False
