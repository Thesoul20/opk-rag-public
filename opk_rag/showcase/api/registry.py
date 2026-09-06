from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Mapping

from opk_rag.showcase.api.events import build_events_from_trace


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    trace: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    created_monotonic: float
    expires_monotonic: float


@dataclass
class TraceRegistry:
    max_trace_count: int = 32
    max_events_per_trace: int = 32
    ttl_seconds: float = 1800.0
    _records: OrderedDict[str, TraceRecord] = field(default_factory=OrderedDict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def put(self, trace: Mapping[str, Any]) -> TraceRecord:
        meta = trace.get("trace") if isinstance(trace.get("trace"), Mapping) else {}
        trace_id = str(meta.get("trace_id") or "")
        if not trace_id:
            raise ValueError("trace_id_missing")
        events = build_events_from_trace(trace)
        if len(events) > self.max_events_per_trace:
            raise ValueError("max_events_per_trace_exceeded")
        now = time.monotonic()
        record = TraceRecord(
            trace_id=trace_id,
            trace=dict(trace),
            events=tuple(events),
            created_monotonic=now,
            expires_monotonic=now + self.ttl_seconds,
        )
        with self._lock:
            self._evict_expired_locked(now)
            self._records[trace_id] = record
            self._records.move_to_end(trace_id)
            self._evict_overflow_locked()
        return record

    def get(self, trace_id: str) -> TraceRecord | None:
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            record = self._records.get(trace_id)
            if record is None:
                return None
            return record

    def replay(self, trace_id: str, *, after_sequence: int = 0) -> tuple[dict[str, Any], ...] | None:
        record = self.get(trace_id)
        if record is None:
            return None
        return tuple(event for event in record.events if int(event["sequence"]) > after_sequence)

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            return {
                "trace_count": len(self._records),
                "max_trace_count": self.max_trace_count,
                "max_events_per_trace": self.max_events_per_trace,
                "ttl_seconds": self.ttl_seconds,
                "trace_registry_bounded": True,
                "trace_registry_ttl_defined": self.ttl_seconds > 0,
            }

    def _evict_expired_locked(self, now: float) -> None:
        expired = [trace_id for trace_id, record in self._records.items() if record.expires_monotonic <= now]
        for trace_id in expired:
            self._records.pop(trace_id, None)

    def _evict_overflow_locked(self) -> None:
        while len(self._records) > self.max_trace_count:
            self._records.popitem(last=False)


def after_sequence_from_last_event_id(value: str | None, *, fallback: int = 0) -> int:
    if not value:
        return fallback
    try:
        return max(0, int(value.rsplit(":", 1)[-1]))
    except (TypeError, ValueError):
        return fallback
