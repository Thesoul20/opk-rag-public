from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
INCIDENT_PATH = ROOT / "evaluation-data" / "governance" / "phase2_holdout_v1_exposure_incident.json"

SCHEMA_VERSION = "opk-rag.holdout-exposure-incident.v1"
INCIDENT_ID = "phase2-holdout-v1-exposure-20260728-task0059"
HOLDOUT_ID = "phase2-dogfooding-holdout-v1"

DEFAULT_PUBLIC_AUDIT_EXCLUDES = (
    ".private/",
    ".git/",
    "evaluation-data/private/",
)
FORBIDDEN_HOLDOUT_SCAN_PREFIXES = (
    ".private/evaluation/phase2_holdout",
    ".private/evaluation/phase2_sealed_baseline_v1/",
)
INCIDENT_FORBIDDEN_KEYS = {
    "question",
    "query",
    "gold_answer",
    "gold_evidence",
    "required_evidence",
    "acceptable_evidence",
    "required_claims",
    "optional_claims",
    "forbidden_claims",
    "source_path",
    "relative_path",
    "snippet",
    "line_content",
    "private_path",
    "secret",
}
HOLDOUT_CONTENT_KEY_RE = re.compile(
    r'"(?:question|gold_answer|gold_evidence|required_evidence|acceptable_evidence|required_claims|forbidden_claims)"\s*:',
    re.IGNORECASE,
)


def repo_relative(path: Path, *, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_public_audit_path(path: Path | str, *, root: Path = ROOT) -> bool:
    raw = str(path).replace("\\", "/")
    if Path(raw).is_absolute():
        try:
            raw = Path(raw).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return False
    while raw.startswith("./"):
        raw = raw[2:]
    return not any(raw == prefix.rstrip("/") or raw.startswith(prefix) for prefix in DEFAULT_PUBLIC_AUDIT_EXCLUDES)


def assert_not_forbidden_holdout_scan_path(path: Path | str, *, root: Path = ROOT) -> None:
    raw = str(path).replace("\\", "/")
    if Path(raw).is_absolute():
        try:
            raw = Path(raw).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return
    while raw.startswith("./"):
        raw = raw[2:]
    if any(raw.startswith(prefix) for prefix in FORBIDDEN_HOLDOUT_SCAN_PREFIXES):
        raise ValueError(f"Refusing to recursively scan sealed holdout path: {raw}")


def iter_public_audit_files(
    roots: Iterable[Path],
    *,
    suffixes: set[str] | None = None,
    root: Path = ROOT,
) -> tuple[Path, ...]:
    files: list[Path] = []
    for base in roots:
        assert_not_forbidden_holdout_scan_path(base, root=root)
        if not base.exists():
            continue
        candidates = (base,) if base.is_file() else base.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            if is_public_audit_path(path, root=root):
                files.append(path)
    return tuple(sorted(files, key=lambda item: repo_relative(item, root=root)))


def public_artifact_contains_holdout_payload_shape(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return bool(HOLDOUT_CONTENT_KEY_RE.search(text))


def validate_incident_artifact(incident: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    required = {
        "schema_version": SCHEMA_VERSION,
        "incident_id": INCIDENT_ID,
        "holdout_id": HOLDOUT_ID,
        "incident_type": "accidental_holdout_content_read",
        "detected_during_task": "TASK-0059",
        "content_persisted": False,
        "content_used_for_implementation": False,
        "content_used_for_parameter_selection": False,
        "content_used_for_metric_calculation": False,
        "future_eligibility": "not_eligible_for_future_stage_acceptance",
        "review_status": "contained",
    }
    for key, expected in required.items():
        if incident.get(key) != expected:
            issues.append({"code": f"{key}_mismatch", "message": f"{key} must be {expected!r}"})

    text = json.dumps(incident, ensure_ascii=False, sort_keys=True)
    for key in INCIDENT_FORBIDDEN_KEYS:
        if f'"{key}"' in text:
            issues.append({"code": "incident_forbidden_key", "message": f"Incident artifact must not include {key!r}."})
    if ".private/" in text or "phase2_holdout_v1.jsonl" in text:
        issues.append({"code": "incident_private_path", "message": "Incident artifact must not include private paths."})
    return issues
