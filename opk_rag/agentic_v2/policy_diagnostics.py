from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import Field, ValidationError

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract, stable_digest

AGENTIC_V2_POLICY_DIAGNOSTIC_VERSION = "opk-rag.agentic-v2.policy-validation-diagnostic.v1"
KNOWN_ACTIONS = {"hybrid_search", "structure_search", "graph_search", "rewrite_query", "inspect_evidence", "finish", "abstain"}


_FENCED_JSON_RE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\s*\Z", re.IGNORECASE | re.DOTALL)


def parse_policy_json_value(raw_output: str) -> Any:
    """Parse one policy JSON value with only whole-response Markdown-fence normalization.

    This intentionally does not extract JSON from surrounding prose. It accepts either raw JSON or a
    single complete ```json ... ``` / ``` ... ``` wrapper, preserving the one-object policy contract.
    """
    text = raw_output.strip()
    if text.startswith("```"):
        match = _FENCED_JSON_RE.fullmatch(text)
        if match is not None:
            text = match.group(1).strip()
    return json.loads(text)


class AgentPolicyValidationDiagnostic(StrictContract):
    contract_version: Literal[AGENTIC_V2_POLICY_DIAGNOSTIC_VERSION] = AGENTIC_V2_POLICY_DIAGNOSTIC_VERSION
    attempt_index: int = Field(ge=0, le=1)
    repair_attempt: bool
    json_parse_valid: bool
    schema_valid: bool
    contract_valid: bool
    failure_code: str | None = None
    failure_detail: str | None = None
    field_paths: tuple[str, ...] = ()
    validation_error_types: tuple[str, ...] = ()
    top_level_keys: tuple[str, ...] = ()
    proposed_action: ActionName | None = None
    proposed_action_present: bool = False
    reason_code: str | None = Field(default=None, max_length=96)
    raw_output_digest: str = Field(min_length=64, max_length=64)
    raw_output_length: int = Field(ge=0)
    raw_output_persisted: bool = False
    hidden_reasoning_persisted: bool = False


def diagnose_policy_output(
    raw_output: str,
    *,
    attempt_index: int,
    repair_attempt: bool,
    validation_error: ValidationError | None = None,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    contract_valid: bool = False,
) -> AgentPolicyValidationDiagnostic:
    parsed: Any = None
    json_parse_valid = False
    keys: tuple[str, ...] = ()
    action = None
    action_present = False
    reason = None
    try:
        parsed = parse_policy_json_value(raw_output)
        json_parse_valid = isinstance(parsed, dict)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        keys = tuple(sorted(str(k) for k in parsed.keys()))
        action_present = "proposed_action" in parsed
        raw_action = parsed.get("proposed_action")
        if isinstance(raw_action, str) and raw_action in KNOWN_ACTIONS:
            action = raw_action
        raw_reason = parsed.get("reason_code")
        if isinstance(raw_reason, str):
            reason = raw_reason[:96]
    paths: list[str] = []
    types: list[str] = []
    if validation_error is not None:
        for row in validation_error.errors():
            loc = ".".join(str(x) for x in row.get("loc", ()))
            if loc:
                paths.append(loc[:160])
            types.append(str(row.get("type", "validation_error"))[:96])
    return AgentPolicyValidationDiagnostic(
        attempt_index=attempt_index,
        repair_attempt=repair_attempt,
        json_parse_valid=json_parse_valid,
        schema_valid=json_parse_valid and validation_error is None and failure_code not in {"policy_schema_validation_failed", "policy_invalid_json"},
        contract_valid=contract_valid,
        failure_code=failure_code,
        failure_detail=failure_detail,
        field_paths=tuple(dict.fromkeys(paths))[:12],
        validation_error_types=tuple(dict.fromkeys(types))[:12],
        top_level_keys=keys[:16],
        proposed_action=action,
        proposed_action_present=action_present,
        reason_code=reason,
        raw_output_digest=stable_digest(raw_output),
        raw_output_length=len(raw_output),
    )


def public_repair_diagnostic(diagnostic: AgentPolicyValidationDiagnostic) -> dict[str, Any]:
    return {
        "failure_code": diagnostic.failure_code,
        "failure_detail": diagnostic.failure_detail,
        "field_paths": list(diagnostic.field_paths),
        "validation_error_types": list(diagnostic.validation_error_types),
        "previous_output_digest": diagnostic.raw_output_digest,
    }
