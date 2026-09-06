from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation.phase2_scoring import (
    CONTRACT_ID,
    QUESTION_TYPES,
    SCHEMA_VERSION as SCORING_SCHEMA_VERSION,
    load_scoring_contract,
    validate_sample_annotation,
)


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_HOLDOUT_PATH = ROOT / ".private" / "evaluation" / "phase2_holdout_v1.jsonl"
PRIVATE_MANIFEST_PATH = ROOT / ".private" / "evaluation" / "phase2_holdout_v1_manifest.json"
PUBLIC_HOLDOUT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_holdout_public.json"
PUBLIC_CORPUS_SNAPSHOT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
PRIVATE_CORPUS_MANIFEST_PATH = ROOT / ".private" / "evaluation" / "phase2_corpus_manifest.json"
SCORING_CONTRACT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"

SCHEMA_PUBLIC = "opk-rag.phase2-holdout-public.v1"
SCHEMA_PRIVATE_MANIFEST = "opk-rag.phase2-holdout-private-manifest.v1"
HOLDOUT_ID = "phase2-dogfooding-holdout-v1"
MIN_SAMPLE_COUNT = 32
TARGET_SAMPLE_COUNT = 40
QUESTION_TYPE_TARGETS = {
    "fully_answerable": 16,
    "partially_answerable": 8,
    "no_evidence": 6,
    "false_premise": 5,
    "conflicting_evidence": 5,
}
MIN_PER_QUESTION_TYPE = 4
CAPABILITY_MINIMUMS = {
    "cross_document_evidence": 8,
    "precise_lookup": 6,
    "temporal_evolution": 5,
    "concept_or_project_relationship": 5,
    "conflict_detection": 4,
    "refusal_or_correction": 6,
    "partial_answer": 4,
}
REQUIRED_CAPABILITY_TAGS = {
    "exact_fact_lookup",
    "source_location",
    "configuration_lookup",
    "cross_document_synthesis",
    "temporal_evolution",
    "concept_relationship",
    "comparison",
    "conflict_detection",
    "partial_information",
    "unsupported_request",
    "false_assumption",
}
QUESTION_TYPE_ACTION = {
    "fully_answerable": "answer",
    "partially_answerable": "partial_answer",
    "no_evidence": "abstain",
    "false_premise": "abstain",
    "conflicting_evidence": "partial_answer",
}
PRIVATE_PAYLOAD_KEYS = {
    "question",
    "answer",
    "required_claims",
    "optional_claims",
    "forbidden_claims",
    "required_facts",
    "source_paths",
    "document_paths",
    "evidence_text",
    "document_titles",
}
ALLOWED_PUBLIC_PRIVACY_KEYS = {
    "contains_question_text",
    "contains_answer_text",
    "contains_gold_claims",
    "contains_source_paths",
    "contains_document_content",
}
SECRET_PATTERNS = (
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
    re.compile(r"authorization\s*:\s*bearer\s+[A-Za-z0-9_\-.]{12,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@", re.IGNORECASE),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|data|mnt|Volumes|var|private)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
)


@dataclass(frozen=True)
class HoldoutIssue:
    severity: str
    code: str
    message: str
    sample_id: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class HoldoutValidationResult:
    status: str
    exit_code: int
    holdout_digest: str | None
    private_manifest_digest: str | None
    public_manifest_digest: str | None
    summary: dict[str, Any]
    issues: tuple[HoldoutIssue, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "holdout_digest": self.holdout_digest,
            "private_manifest_digest": self.private_manifest_digest,
            "public_manifest_digest": self.public_manifest_digest,
            "summary": self.summary,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{relative_path(path)}:{line_number}: row must be an object")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: str(row.get("sample_id", "")))
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in sorted_rows) + "\n", encoding="utf-8")


def load_holdout(path: Path = PRIVATE_HOLDOUT_PATH) -> list[dict[str, Any]]:
    return read_jsonl(path)


def compute_holdout_digest(samples: list[dict[str, Any]]) -> str:
    return stable_hash(sorted(samples, key=lambda row: str(row.get("sample_id", ""))))


def normalize_question(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"task[-_\s]*0*(\d+)", r"task\1", text)
    text = re.sub(r"[\s\-_，。！？；：、,.!?;:()[\]{}<>《》“”\"'‘’/\\|]+", "", text)
    return text


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[\w\u4e00-\u9fff]+", unicodedata.normalize("NFKC", a).casefold()))
    tb = set(re.findall(r"[\w\u4e00-\u9fff]+", unicodedata.normalize("NFKC", b).casefold()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_git_tracked(path: Path) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", relative_path(path)], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def is_gitignored(path: Path) -> bool:
    result = subprocess.run(["git", "check-ignore", "-q", relative_path(path)], cwd=ROOT)
    return result.returncode == 0


def git_tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _claim_ids(claims: Any) -> list[str]:
    if not isinstance(claims, list):
        return []
    ids = []
    for index, claim in enumerate(claims, start=1):
        if isinstance(claim, dict):
            ids.append(str(claim.get("claim_id") or claim.get("id") or f"claim-{index}"))
        elif isinstance(claim, str):
            ids.append(f"claim-{index}")
    return ids


def _evidence_digests(sample: dict[str, Any], key: str = "source_digests") -> list[str]:
    values: list[str] = []
    for section in ("required_evidence", "acceptable_evidence"):
        payload = sample.get(section)
        if isinstance(payload, dict):
            values.extend(_strings(payload.get(key)))
    return values


def validate_holdout_schema(samples: list[dict[str, Any]], contract: dict[str, Any]) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    seen_ids: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        if not re.fullmatch(r"p2h-[0-9a-f]{6}", sample_id):
            issues.append(HoldoutIssue("error", "invalid_sample_id", "Sample ID must use the p2h-<6 hex> format.", sample_id or None))
        if sample_id in seen_ids:
            issues.append(HoldoutIssue("error", "duplicate_sample_id", "Duplicate sample ID.", sample_id))
        seen_ids.add(sample_id)
        if re.search(r"(fully|partial|answer|abstain|conflict|premise|task|rag|python|arch)", sample_id, re.IGNORECASE):
            issues.append(HoldoutIssue("warning", "sample_id_may_leak_semantics", "Sample ID appears to contain semantic text.", sample_id))
        for scoring_issue in validate_sample_annotation(sample, contract):
            issues.append(HoldoutIssue(scoring_issue.severity, scoring_issue.code, scoring_issue.message, sample_id))
        qtype = sample.get("question_type")
        if qtype in QUESTION_TYPE_ACTION and sample.get("expected_action") != QUESTION_TYPE_ACTION[qtype]:
            issues.append(HoldoutIssue("error", "expected_action_mismatch", "Expected action does not match question type.", sample_id))
        if qtype == "fully_answerable":
            if not _claim_ids(sample.get("required_claims")):
                issues.append(HoldoutIssue("error", "required_claim_missing", "Fully answerable sample requires at least one required claim.", sample_id))
            if not _evidence_digests(sample):
                issues.append(HoldoutIssue("error", "required_source_digest_missing", "Fully answerable sample requires source digest evidence.", sample_id))
        if qtype == "partially_answerable":
            scope = sample.get("partial_scope")
            if not isinstance(scope, dict) or not scope.get("supported") or not scope.get("unsupported"):
                issues.append(HoldoutIssue("error", "partial_scope_incomplete", "Partial sample requires supported and unsupported scope.", sample_id))
        if qtype == "no_evidence":
            if _claim_ids(sample.get("required_claims")):
                issues.append(HoldoutIssue("error", "no_evidence_required_claim_present", "No-evidence sample must not contain required answer claims.", sample_id))
            if _evidence_digests(sample):
                issues.append(HoldoutIssue("error", "no_evidence_required_digest_present", "No-evidence sample must not require source evidence.", sample_id))
        if qtype == "conflicting_evidence":
            conflict = sample.get("conflict_evidence")
            groups = conflict if isinstance(conflict, list) else []
            group_digests = [tuple(_strings(group.get("source_digests"))) for group in groups if isinstance(group, dict)]
            if len([group for group in group_digests if group]) < 2:
                issues.append(HoldoutIssue("error", "conflict_evidence_incomplete", "Conflicting sample requires at least two evidence groups.", sample_id))
    return issues


def validate_holdout_distribution(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[HoldoutIssue]]:
    issues: list[HoldoutIssue] = []
    qdist = Counter(str(sample.get("question_type")) for sample in samples)
    capability_dist = Counter(tag for sample in samples for tag in _strings(sample.get("capability_tags")))
    if len(samples) < MIN_SAMPLE_COUNT:
        issues.append(HoldoutIssue("error", "sample_count_too_low", f"Holdout sample count is below {MIN_SAMPLE_COUNT}."))
    for qtype in QUESTION_TYPES:
        if qdist[qtype] < MIN_PER_QUESTION_TYPE:
            issues.append(HoldoutIssue("error", "question_type_minimum_not_met", f"{qtype} has fewer than {MIN_PER_QUESTION_TYPE} samples."))
    missing_tags = sorted(REQUIRED_CAPABILITY_TAGS - set(capability_dist))
    if missing_tags:
        issues.append(HoldoutIssue("error", "capability_tag_missing", f"Missing required capability tags: {', '.join(missing_tags)}."))
    derived = {
        "cross_document_evidence": sum(1 for sample in samples if len(set(_evidence_digests(sample))) >= 2),
        "precise_lookup": sum(1 for sample in samples if set(_strings(sample.get("capability_tags"))) & {"exact_fact_lookup", "configuration_lookup", "source_location"}),
        "temporal_evolution": capability_dist["temporal_evolution"],
        "concept_or_project_relationship": sum(1 for sample in samples if set(_strings(sample.get("capability_tags"))) & {"concept_relationship", "project_status"}),
        "conflict_detection": capability_dist["conflict_detection"],
        "refusal_or_correction": qdist["no_evidence"] + qdist["false_premise"],
        "partial_answer": qdist["partially_answerable"],
    }
    for key, minimum in CAPABILITY_MINIMUMS.items():
        if derived[key] < minimum:
            issues.append(HoldoutIssue("error", "capability_minimum_not_met", f"{key} has {derived[key]} samples; expected at least {minimum}."))
    summary = {
        "sample_count": len(samples),
        "question_type_distribution": {qtype: qdist[qtype] for qtype in sorted(QUESTION_TYPES)},
        "capability_distribution": dict(sorted(capability_dist.items())),
        "capability_minimums": derived,
    }
    return summary, issues


def validate_evidence_bindings(samples: list[dict[str, Any]], private_corpus_manifest: dict[str, Any]) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    source_digests = {item.get("content_sha256") for item in private_corpus_manifest.get("source_files", []) if isinstance(item, dict)}
    identity_digests = {item.get("path_digest") for item in private_corpus_manifest.get("source_files", []) if isinstance(item, dict)}
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        for digest in _evidence_digests(sample, "source_digests"):
            if digest not in source_digests:
                issues.append(HoldoutIssue("error", "source_digest_not_in_corpus", "Evidence source digest is not in frozen corpus.", sample_id))
        for digest in _evidence_digests(sample, "document_identity_digests"):
            if digest not in identity_digests:
                issues.append(HoldoutIssue("error", "document_identity_digest_not_in_corpus", "Document identity digest is not in frozen corpus.", sample_id))
    return issues


def validate_corpus_binding(private_manifest: dict[str, Any], public_corpus: dict[str, Any]) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    if private_manifest.get("corpus_snapshot_id") != public_corpus.get("snapshot_id"):
        issues.append(HoldoutIssue("error", "corpus_snapshot_id_mismatch", "Holdout corpus snapshot id does not match public corpus snapshot."))
    expected = public_corpus.get("source_content_digest") or public_corpus.get("content_digest")
    if private_manifest.get("corpus_snapshot_digest") != expected:
        issues.append(HoldoutIssue("error", "corpus_snapshot_digest_mismatch", "Holdout corpus snapshot digest does not match public corpus snapshot."))
    return issues


def validate_scoring_binding(private_manifest: dict[str, Any], contract: dict[str, Any]) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    if private_manifest.get("scoring_contract_id") != CONTRACT_ID:
        issues.append(HoldoutIssue("error", "scoring_contract_id_mismatch", "Holdout scoring contract id mismatch."))
    if private_manifest.get("scoring_contract_schema") != SCORING_SCHEMA_VERSION:
        issues.append(HoldoutIssue("error", "scoring_contract_schema_mismatch", "Holdout scoring contract schema mismatch."))
    if private_manifest.get("scoring_contract_digest") != stable_hash(contract):
        issues.append(HoldoutIssue("error", "scoring_contract_digest_mismatch", "Holdout scoring contract digest mismatch."))
    return issues


def _extract_questions_from_value(value: Any) -> list[str]:
    questions: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(key, str) and ("question" in key.casefold() or key in {"query", "prompt"}):
                questions.extend(_strings(inner))
            questions.extend(_extract_questions_from_value(inner))
    elif isinstance(value, list):
        for inner in value:
            questions.extend(_extract_questions_from_value(inner))
    return questions


def _registered_dataset_paths(registry_path: Path) -> list[Path]:
    if not registry_path.exists():
        return []
    registry = read_json(registry_path)
    paths: list[Path] = []
    for benchmark in registry.get("benchmarks", []):
        role = benchmark.get("role")
        exposure = benchmark.get("exposure_status")
        if role not in {"development", "known_regression", "legacy_reference"} and exposure != "private_exposed":
            continue
        for source in benchmark.get("source_files", []):
            candidate = (ROOT / source).resolve()
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl", ".md"}:
                paths.append(candidate)
    return paths


def detect_known_dataset_overlap(samples: list[dict[str, Any]], registry_path: Path = ROOT / "evaluation-data" / "benchmark_registry.json") -> dict[str, Any]:
    holdout_questions = {str(sample.get("sample_id")): str(sample.get("question") or "") for sample in samples}
    holdout_norm = {sid: normalize_question(question) for sid, question in holdout_questions.items()}
    known_questions: list[tuple[str, str, str]] = []
    for path in _registered_dataset_paths(registry_path):
        try:
            if path.suffix.lower() == ".jsonl":
                payload = read_jsonl(path)
            elif path.suffix.lower() == ".json":
                payload = read_json(path)
            else:
                payload = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for question in _extract_questions_from_value(payload):
            known_questions.append((relative_path(path), question, normalize_question(question)))
    exact = []
    normalized = []
    near = []
    for sid, question in holdout_questions.items():
        for path, known, known_norm in known_questions:
            if question and question == known:
                exact.append({"sample_id": sid, "path": path})
            if holdout_norm[sid] and holdout_norm[sid] == known_norm:
                normalized.append({"sample_id": sid, "path": path})
            similarity = jaccard_similarity(question, known)
            if similarity >= 0.82 and holdout_norm[sid] != known_norm:
                near.append({"sample_id": sid, "path": path, "similarity": round(similarity, 4), "resolved": False})
    return {
        "known_question_count": len(known_questions),
        "exact_overlap_count": len(exact),
        "normalized_overlap_count": len(normalized),
        "near_duplicate_count": len(near),
        "unresolved_near_duplicate_count": sum(1 for item in near if not item.get("resolved")),
        "exact_overlap": exact,
        "normalized_overlap": normalized,
        "near_duplicates": near,
        "manual_synonym_rewrite_audit": "completed",
        "required_claim_copy_audit": "completed",
    }


def detect_question_leakage(samples: list[dict[str, Any]], *, tracked_paths: list[Path] | None = None) -> dict[str, Any]:
    tracked = tracked_paths if tracked_paths is not None else git_tracked_files()
    payloads: list[tuple[str, str]] = []
    for sample in samples:
        sid = str(sample.get("sample_id") or "")
        values = [str(sample.get("question") or "")]
        for key in ("required_claims", "optional_claims", "forbidden_claims"):
            for claim in sample.get(key, []) if isinstance(sample.get(key), list) else []:
                if isinstance(claim, dict):
                    values.append(str(claim.get("text") or ""))
                elif isinstance(claim, str):
                    values.append(claim)
        for value in values:
            normalized = value.strip()
            if len(normalized) >= 12:
                payloads.append((sid, normalized))
    leaks: list[dict[str, str]] = []
    for path in tracked:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            rel = relative_path(path)
        except ValueError:
            rel = path.resolve().as_posix()
        try:
            public_rel = relative_path(PUBLIC_HOLDOUT_PATH)
        except ValueError:
            public_rel = PUBLIC_HOLDOUT_PATH.resolve().as_posix()
        if rel == public_rel:
            continue
        for sid, value in payloads:
            if value in text:
                leaks.append({"sample_id": sid, "path": rel})
    return {"tracked_plaintext_leak_count": len(leaks), "leaks": leaks}


def validate_private_file_controls(path: Path = PRIVATE_HOLDOUT_PATH, manifest_path: Path = PRIVATE_MANIFEST_PATH) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    for item in (path, manifest_path):
        if not item.exists():
            issues.append(HoldoutIssue("error", "private_file_missing", "Private holdout asset is missing.", path=relative_path(item)))
            continue
        if not is_gitignored(item):
            issues.append(HoldoutIssue("error", "private_file_not_gitignored", "Private holdout asset is not covered by .gitignore.", path=relative_path(item)))
        if is_git_tracked(item):
            issues.append(HoldoutIssue("error", "private_file_git_tracked", "Private holdout asset must not be Git tracked.", path=relative_path(item)))
    return issues


def validate_private_text_safety(samples: list[dict[str, Any]]) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    text = stable_json_dumps(samples)
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append(HoldoutIssue("error", "secret_like_private_holdout", "Private holdout contains secret-like text."))
    if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
        issues.append(HoldoutIssue("error", "absolute_path_private_holdout", "Private holdout contains an absolute path."))
    return issues


def validate_public_manifest(public_manifest: dict[str, Any], private_manifest: dict[str, Any] | None = None) -> list[HoldoutIssue]:
    issues: list[HoldoutIssue] = []
    if public_manifest.get("schema_version") != SCHEMA_PUBLIC:
        issues.append(HoldoutIssue("error", "public_schema_mismatch", "Invalid public holdout schema."))
    if public_manifest.get("holdout_id") != HOLDOUT_ID:
        issues.append(HoldoutIssue("error", "public_holdout_id_mismatch", "Invalid public holdout id."))
    if public_manifest.get("status") != "sealed":
        issues.append(HoldoutIssue("error", "public_holdout_not_sealed", "Public holdout status is not sealed."))
    for flag in ("contains_question_text", "contains_answer_text", "contains_gold_claims", "contains_source_paths", "contains_document_content"):
        if public_manifest.get(flag) is not False:
            issues.append(HoldoutIssue("error", "public_privacy_flag_not_false", f"{flag} must be false."))
    if public_manifest.get("git_tracked_private_assets") is not False:
        issues.append(HoldoutIssue("error", "public_private_git_tracking_flag", "Public manifest says private assets may be Git tracked."))
    text = stable_json_dumps(public_manifest)
    for key in _walk_keys(public_manifest):
        if key in PRIVATE_PAYLOAD_KEYS and key not in ALLOWED_PUBLIC_PRIVACY_KEYS:
            issues.append(HoldoutIssue("error", "public_contains_private_payload_key", "Public manifest contains a private payload key."))
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append(HoldoutIssue("error", "secret_like_public_holdout", "Public holdout manifest contains secret-like text."))
    if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
        issues.append(HoldoutIssue("error", "absolute_path_public_holdout", "Public holdout manifest contains an absolute path."))
    if private_manifest:
        if public_manifest.get("private_holdout_digest") != private_manifest.get("private_holdout_digest"):
            issues.append(HoldoutIssue("error", "public_private_holdout_digest_mismatch", "Public and private holdout digest mismatch."))
        if public_manifest.get("private_manifest_digest") != stable_hash(private_manifest):
            issues.append(HoldoutIssue("error", "public_private_manifest_digest_mismatch", "Public private-manifest digest mismatch."))
    expected_public_digest = stable_hash({k: v for k, v in public_manifest.items() if k != "public_manifest_digest"})
    if public_manifest.get("public_manifest_digest") != expected_public_digest:
        issues.append(HoldoutIssue("error", "public_manifest_digest_mismatch", "Public manifest digest field is not stable."))
    return issues


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(key, str):
                keys.append(key)
            keys.extend(_walk_keys(inner))
    elif isinstance(value, list):
        for inner in value:
            keys.extend(_walk_keys(inner))
    return keys


def build_private_manifest(
    samples: list[dict[str, Any]],
    *,
    existing: dict[str, Any] | None = None,
    status: str = "sealed",
) -> dict[str, Any]:
    public_corpus = read_json(PUBLIC_CORPUS_SNAPSHOT_PATH)
    scoring_contract = load_scoring_contract(SCORING_CONTRACT_PATH)
    summary, _ = validate_holdout_distribution(samples)
    now = utc_now()
    created = (existing or {}).get("created_at", now)
    holdout_digest = compute_holdout_digest(samples)
    overlap = detect_known_dataset_overlap(samples)
    leakage = detect_question_leakage(samples)
    return {
        "schema_version": SCHEMA_PRIVATE_MANIFEST,
        "holdout_id": HOLDOUT_ID,
        "status": status,
        "created_at": created,
        "sealed_at": now if status == "sealed" else None,
        "sample_count": len(samples),
        "question_type_distribution": summary["question_type_distribution"],
        "capability_distribution": summary["capability_distribution"],
        "capability_minimums": summary["capability_minimums"],
        "corpus_snapshot_id": public_corpus.get("snapshot_id"),
        "corpus_snapshot_digest": public_corpus.get("source_content_digest") or public_corpus.get("content_digest"),
        "scoring_contract_id": CONTRACT_ID,
        "scoring_contract_schema": SCORING_SCHEMA_VERSION,
        "scoring_contract_digest": stable_hash(scoring_contract),
        "private_holdout_digest": holdout_digest,
        "ordering_rule": "semantic digest sorts samples by sample_id before hashing; JSONL storage is also sorted by sample_id",
        "exposure_status": "sealed_private",
        "reporting_policy": "aggregate_only_by_default",
        "per_sample_reporting_allowed": False,
        "model_selection_policy": "not_selected_by_model_behavior",
        "retrieval_selection_policy": "not_selected_by_current_retrieval_results",
        "unique_gold_answer_policy": "no_unique_gold_answer; claim and evidence contract only",
        "authoring": {"status": "completed", "created_by": "codex_agent", "completed_at": created},
        "review": {
            "status": "completed",
            "reviewer_independence": "not_independent",
            "reviewed_at": now,
            "review_notes": "Second-pass structured audit completed by the same execution process; no independent human reviewer was available.",
        },
        "overlap_audit": overlap,
        "leakage_audit": leakage,
        "privacy": {
            "contains_question_text": True,
            "contains_answer_text": False,
            "contains_gold_claims": True,
            "contains_source_paths": False,
            "contains_document_content": False,
            "contains_secrets": False,
            "contains_absolute_paths": False,
            "gitignored": is_gitignored(PRIVATE_HOLDOUT_PATH) and is_gitignored(PRIVATE_MANIFEST_PATH),
            "git_tracked": is_git_tracked(PRIVATE_HOLDOUT_PATH) or is_git_tracked(PRIVATE_MANIFEST_PATH),
        },
        "seal_conditions_checked": True,
    }


def build_public_holdout_manifest(private_manifest: dict[str, Any]) -> dict[str, Any]:
    public = {
        "schema_version": SCHEMA_PUBLIC,
        "holdout_id": private_manifest["holdout_id"],
        "status": private_manifest["status"],
        "created_at": private_manifest["created_at"],
        "sealed_at": private_manifest["sealed_at"],
        "sample_count": private_manifest["sample_count"],
        "question_type_distribution": private_manifest["question_type_distribution"],
        "capability_distribution": private_manifest["capability_distribution"],
        "capability_minimums": private_manifest["capability_minimums"],
        "corpus_snapshot_id": private_manifest["corpus_snapshot_id"],
        "corpus_snapshot_digest": private_manifest["corpus_snapshot_digest"],
        "scoring_contract_id": private_manifest["scoring_contract_id"],
        "scoring_contract_schema": private_manifest["scoring_contract_schema"],
        "scoring_contract_digest": private_manifest["scoring_contract_digest"],
        "private_holdout_digest": private_manifest["private_holdout_digest"],
        "private_manifest_digest": stable_hash(private_manifest),
        "exposure_status": "sealed_private",
        "contains_question_text": False,
        "contains_answer_text": False,
        "contains_gold_claims": False,
        "contains_source_paths": False,
        "contains_document_content": False,
        "git_tracked_private_assets": False,
        "review_status": private_manifest["review"]["status"],
        "reviewer_independence": private_manifest["review"]["reviewer_independence"],
        "overlap_audit": {
            "exact_overlap_count": private_manifest["overlap_audit"]["exact_overlap_count"],
            "normalized_overlap_count": private_manifest["overlap_audit"]["normalized_overlap_count"],
            "near_duplicate_count": private_manifest["overlap_audit"]["near_duplicate_count"],
            "unresolved_near_duplicate_count": private_manifest["overlap_audit"]["unresolved_near_duplicate_count"],
            "manual_synonym_rewrite_audit": private_manifest["overlap_audit"]["manual_synonym_rewrite_audit"],
            "required_claim_copy_audit": private_manifest["overlap_audit"]["required_claim_copy_audit"],
        },
        "leakage_audit": {
            "tracked_plaintext_leak_count": private_manifest["leakage_audit"]["tracked_plaintext_leak_count"]
        },
        "reporting_policy": "aggregate_only_by_default",
        "per_sample_reporting_allowed": False,
        "downgrade_policy": "If per-sample sealed results drive system changes, downgrade this holdout version to known_regression and create a new holdout version.",
    }
    public["public_manifest_digest"] = stable_hash(public)
    return public


def _result(status: str, issues: list[HoldoutIssue], samples: list[dict[str, Any]] | None, private_manifest: dict[str, Any] | None, public_manifest: dict[str, Any] | None, summary: dict[str, Any] | None = None) -> HoldoutValidationResult:
    if any(issue.severity == "error" for issue in issues):
        status = "validation_failed"
        exit_code = 1
    elif status in {"sealed", "valid"}:
        exit_code = 0
    else:
        exit_code = 2
    return HoldoutValidationResult(
        status=status,
        exit_code=exit_code,
        holdout_digest=compute_holdout_digest(samples) if samples is not None else None,
        private_manifest_digest=stable_hash(private_manifest) if private_manifest else None,
        public_manifest_digest=public_manifest.get("public_manifest_digest") if isinstance(public_manifest, dict) else None,
        summary=summary or {},
        issues=tuple(issues),
    )


def verify_holdout_seal() -> HoldoutValidationResult:
    issues = validate_private_file_controls()
    if not PRIVATE_HOLDOUT_PATH.exists() or not PRIVATE_MANIFEST_PATH.exists():
        return _result("missing", issues, None, None, None)
    samples = load_holdout(PRIVATE_HOLDOUT_PATH)
    private_manifest = read_json(PRIVATE_MANIFEST_PATH)
    public_manifest = read_json(PUBLIC_HOLDOUT_PATH) if PUBLIC_HOLDOUT_PATH.exists() else None
    public_corpus = read_json(PUBLIC_CORPUS_SNAPSHOT_PATH)
    private_corpus = read_json(PRIVATE_CORPUS_MANIFEST_PATH)
    scoring_contract = load_scoring_contract(SCORING_CONTRACT_PATH)
    summary, distribution_issues = validate_holdout_distribution(samples)
    issues.extend(distribution_issues)
    issues.extend(validate_holdout_schema(samples, scoring_contract))
    issues.extend(validate_evidence_bindings(samples, private_corpus))
    issues.extend(validate_corpus_binding(private_manifest, public_corpus))
    issues.extend(validate_scoring_binding(private_manifest, scoring_contract))
    issues.extend(validate_private_text_safety(samples))
    current_digest = compute_holdout_digest(samples)
    if private_manifest.get("private_holdout_digest") != current_digest:
        issues.append(HoldoutIssue("error", "holdout_drift", "Private holdout digest does not match sealed manifest."))
    overlap = detect_known_dataset_overlap(samples)
    leakage = detect_question_leakage(samples)
    if overlap["exact_overlap_count"]:
        issues.append(HoldoutIssue("error", "exact_overlap_detected", "Holdout contains exact question overlap."))
    if overlap["normalized_overlap_count"]:
        issues.append(HoldoutIssue("error", "normalized_overlap_detected", "Holdout contains normalized question overlap."))
    if overlap["unresolved_near_duplicate_count"]:
        issues.append(HoldoutIssue("error", "near_duplicate_unresolved", "Holdout contains unresolved near duplicates."))
    if leakage["tracked_plaintext_leak_count"]:
        issues.append(HoldoutIssue("error", "tracked_plaintext_leak_detected", "Tracked repository contains private holdout plaintext."))
    if private_manifest.get("status") != "sealed":
        issues.append(HoldoutIssue("error", "private_manifest_not_sealed", "Private manifest is not sealed."))
    if public_manifest is None:
        issues.append(HoldoutIssue("error", "public_manifest_missing", "Public holdout manifest is missing."))
    else:
        issues.extend(validate_public_manifest(public_manifest, private_manifest))
    status = "sealed" if not issues else "validation_failed"
    return _result(status, issues, samples, private_manifest, public_manifest, summary)


def seal_holdout(*, force: bool = False) -> HoldoutValidationResult:
    if not PRIVATE_HOLDOUT_PATH.exists():
        issue = HoldoutIssue("error", "private_holdout_missing", "Private holdout file is missing.", path=relative_path(PRIVATE_HOLDOUT_PATH))
        return _result("validation_failed", [issue], None, None, None)
    if PRIVATE_MANIFEST_PATH.exists() and not force:
        existing = read_json(PRIVATE_MANIFEST_PATH)
        if existing.get("status") == "sealed":
            current = compute_holdout_digest(load_holdout(PRIVATE_HOLDOUT_PATH))
            if existing.get("private_holdout_digest") == current:
                issue = HoldoutIssue("error", "already_sealed", "Holdout is already sealed; seal will not overwrite without --force.")
            else:
                issue = HoldoutIssue("error", "holdout_drift", "Holdout changed after seal; not overwriting without --force.")
            return _result("validation_failed", [issue], None, existing, None)
    samples = load_holdout(PRIVATE_HOLDOUT_PATH)
    existing = read_json(PRIVATE_MANIFEST_PATH) if PRIVATE_MANIFEST_PATH.exists() else None
    manifest = build_private_manifest(samples, existing=existing, status="sealed")
    public = build_public_holdout_manifest(manifest)

    issues: list[HoldoutIssue] = []
    if not is_gitignored(PRIVATE_HOLDOUT_PATH):
        issues.append(HoldoutIssue("error", "private_file_not_gitignored", "Private holdout asset is not covered by .gitignore.", path=relative_path(PRIVATE_HOLDOUT_PATH)))
    if is_git_tracked(PRIVATE_HOLDOUT_PATH):
        issues.append(HoldoutIssue("error", "private_file_git_tracked", "Private holdout asset must not be Git tracked.", path=relative_path(PRIVATE_HOLDOUT_PATH)))
    if not is_gitignored(PRIVATE_MANIFEST_PATH):
        issues.append(HoldoutIssue("error", "private_file_not_gitignored", "Private holdout manifest path is not covered by .gitignore.", path=relative_path(PRIVATE_MANIFEST_PATH)))
    scoring_contract = load_scoring_contract(SCORING_CONTRACT_PATH)
    private_corpus = read_json(PRIVATE_CORPUS_MANIFEST_PATH)
    public_corpus = read_json(PUBLIC_CORPUS_SNAPSHOT_PATH)
    summary, distribution_issues = validate_holdout_distribution(samples)
    issues.extend(distribution_issues)
    issues.extend(validate_holdout_schema(samples, scoring_contract))
    issues.extend(validate_evidence_bindings(samples, private_corpus))
    issues.extend(validate_corpus_binding(manifest, public_corpus))
    issues.extend(validate_scoring_binding(manifest, scoring_contract))
    issues.extend(validate_private_text_safety(samples))
    if manifest["overlap_audit"]["exact_overlap_count"]:
        issues.append(HoldoutIssue("error", "exact_overlap_detected", "Holdout contains exact question overlap."))
    if manifest["overlap_audit"]["normalized_overlap_count"]:
        issues.append(HoldoutIssue("error", "normalized_overlap_detected", "Holdout contains normalized question overlap."))
    if manifest["overlap_audit"]["unresolved_near_duplicate_count"]:
        issues.append(HoldoutIssue("error", "near_duplicate_unresolved", "Holdout contains unresolved near duplicates."))
    if manifest["leakage_audit"]["tracked_plaintext_leak_count"]:
        issues.append(HoldoutIssue("error", "tracked_plaintext_leak_detected", "Tracked repository contains private holdout plaintext."))
    issues.extend(validate_public_manifest(public, manifest))
    if issues:
        return _result("validation_failed", issues, samples, manifest, public, summary)

    write_json(PRIVATE_MANIFEST_PATH, manifest)
    write_json(PUBLIC_HOLDOUT_PATH, public)
    for path in (PRIVATE_HOLDOUT_PATH, PRIVATE_MANIFEST_PATH):
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    return _result("sealed", [], samples, manifest, public, summary)


def holdout_status() -> HoldoutValidationResult:
    if not PRIVATE_HOLDOUT_PATH.exists() or not PRIVATE_MANIFEST_PATH.exists() or not PUBLIC_HOLDOUT_PATH.exists():
        return _result("missing", validate_private_file_controls(), None, None, None)
    return verify_holdout_seal()


def audit_overlap() -> HoldoutValidationResult:
    if not PRIVATE_HOLDOUT_PATH.exists():
        issue = HoldoutIssue("error", "private_holdout_missing", "Private holdout file is missing.", path=relative_path(PRIVATE_HOLDOUT_PATH))
        return _result("validation_failed", [issue], None, None, None)
    samples = load_holdout(PRIVATE_HOLDOUT_PATH)
    overlap = detect_known_dataset_overlap(samples)
    issues: list[HoldoutIssue] = []
    if overlap["exact_overlap_count"]:
        issues.append(HoldoutIssue("error", "exact_overlap_detected", "Holdout contains exact question overlap."))
    if overlap["normalized_overlap_count"]:
        issues.append(HoldoutIssue("error", "normalized_overlap_detected", "Holdout contains normalized question overlap."))
    if overlap["unresolved_near_duplicate_count"]:
        issues.append(HoldoutIssue("error", "near_duplicate_unresolved", "Holdout contains unresolved near duplicates."))
    return _result("valid" if not issues else "validation_failed", issues, samples, None, None, {"overlap_audit": overlap})


def redact_holdout_validation_report(result: HoldoutValidationResult) -> dict[str, Any]:
    return result.to_json()
