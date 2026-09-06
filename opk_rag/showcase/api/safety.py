from __future__ import annotations

from typing import Any, Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

from opk_rag.showcase.api.models import ErrorResponse
from opk_rag.showcase.runtime_trace import scan_forbidden_keys, scan_sensitive_values


SAFE_ERROR_MESSAGES = {
    "authority_unavailable": "Showcase authority is unavailable.",
    "runtime_unavailable": "Showcase runtime is unavailable.",
    "trace_not_found": "Trace was not found or has expired.",
    "scenario_not_found": "Showcase scenario was not found.",
    "invalid_request": "The request is invalid.",
    "execution_rejected": "Execution was rejected by the transport boundary.",
    "execution_failed": "Execution failed before a complete trace was available.",
    "conversation_unavailable": "Conversation storage is unavailable.",
    "conversation_session_not_found": "Conversation session was not found.",
    "conversation_execution_failed": "Conversation execution failed safely.",
}


def safe_error(
    code: str,
    *,
    status_code: int = 400,
    trace_id: str | None = None,
    stage: str | None = None,
    retryable: bool = False,
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=SAFE_ERROR_MESSAGES.get(code, "Request could not be completed."),
        trace_id=trace_id,
        stage=stage,
        retryable=retryable,
    ).model_dump()
    return JSONResponse(payload, status_code=status_code)


async def body_size_guard(request: Request, call_next, *, max_body_bytes: int = 8192):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > max_body_bytes:
                return safe_error("invalid_request", status_code=413, stage="transport")
        except ValueError:
            return safe_error("invalid_request", status_code=400, stage="transport")
    return await call_next(request)


def sensitive_scan_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    findings = [
        {"kind": "forbidden_key", "finding": value}
        for value in scan_forbidden_keys(payload)
    ]
    findings.extend(
        {"kind": "sensitive_value", "finding": value}
        for value in scan_sensitive_values(payload)
    )
    return {
        "finding_count": len(findings),
        "findings": findings,
        "sensitive_data_scan_passed": not findings,
    }
