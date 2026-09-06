from __future__ import annotations

import json
from typing import Any, Mapping

from opk_rag.showcase.api.models import TRACE_EVENT_SCHEMA_VERSION, RuntimeTraceEvent
from opk_rag.showcase.runtime_trace import TOP_LEVEL_SECTIONS


TERMINAL_EVENT_TYPES = {
    "completed": "trace_completed",
    "refused": "trace_refused",
    "failed": "trace_failed",
    "partial": "trace_partial",
}


def _event_timestamp(trace: Mapping[str, Any], stage: str) -> str | None:
    meta = trace.get("trace") if isinstance(trace.get("trace"), Mapping) else {}
    if stage == "trace":
        return meta.get("started_at")
    if stage == "outcome":
        return meta.get("completed_at") or meta.get("started_at")
    return meta.get("completed_at") or meta.get("started_at")


def build_events_from_trace(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    meta = trace.get("trace") if isinstance(trace.get("trace"), Mapping) else {}
    trace_id = str(meta.get("trace_id") or "")
    status = str(meta.get("status") or "failed")
    events: list[dict[str, Any]] = []
    sequence = 1
    for stage in TOP_LEVEL_SECTIONS:
        if stage == "outcome":
            continue
        payload = trace.get(stage)
        if not isinstance(payload, Mapping):
            payload = {"value": payload}
        events.append(
            RuntimeTraceEvent(
                event_schema_version=TRACE_EVENT_SCHEMA_VERSION,
                event_id=f"{trace_id}:{sequence}",
                trace_id=trace_id,
                sequence=sequence,
                event_type="stage_snapshot",
                stage=stage,
                timestamp=_event_timestamp(trace, stage),
                payload=dict(payload),
                terminal=False,
            ).model_dump()
        )
        sequence += 1
    terminal_type = TERMINAL_EVENT_TYPES.get(status, "trace_failed")
    events.append(
        RuntimeTraceEvent(
            event_schema_version=TRACE_EVENT_SCHEMA_VERSION,
            event_id=f"{trace_id}:{sequence}",
            trace_id=trace_id,
            sequence=sequence,
            event_type=terminal_type,
            stage="outcome",
            timestamp=_event_timestamp(trace, "outcome"),
            payload={"trace": dict(meta), "outcome": dict(trace.get("outcome") or {})},
            terminal=True,
        ).model_dump()
    )
    return events


def encode_sse_event(event: Mapping[str, Any]) -> str:
    event_id = str(event["event_id"])
    event_type = str(event["event_type"])
    data = json.dumps(dict(event), ensure_ascii=False, sort_keys=True)
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
