from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from opk_rag.runtime.config_sources import ConfigFieldSource


DoctorCheckStatus = Literal["pass", "warning", "fail", "skipped"]
DoctorOverallStatus = Literal["healthy", "degraded", "unhealthy"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    check_id: str
    component: str
    status: DoctorCheckStatus
    summary: str
    details: tuple[str, ...] = ()
    duration_ms: int = 0
    component_state: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "check_id": self.check_id,
            "component": self.component,
            "status": self.status,
            "summary": self.summary,
            "details": list(self.details),
            "duration_ms": self.duration_ms,
        }
        if self.component_state is not None:
            payload["component_state"] = self.component_state
        return payload


@dataclass(frozen=True, slots=True)
class DoctorReport:
    schema_version: str
    overall_status: DoctorOverallStatus
    remote_requested: bool
    checks: tuple[DoctorCheck, ...]
    summary: str
    config_sources: tuple[ConfigFieldSource, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "overall_status": self.overall_status,
            "remote_requested": self.remote_requested,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.summary,
        }
        if self.config_sources:
            payload["config_sources"] = [source.to_dict() for source in self.config_sources]
        return payload
