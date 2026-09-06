from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from opk_rag.evaluation.graphrag_readiness import MARKDOWN_LINK_RE, ROOT, SOURCE_DIR, WIKI_LINK_RE, _line_number


CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class LinkRecord:
    link_id: str
    source_document_id: str
    source_section_id: str
    raw_link_text: str
    link_syntax: str
    normalized_target: str
    heading_fragment: str
    block_fragment: str
    alias_text: str
    candidate_targets: tuple[str, ...]
    root_cause_class: str
    resolution_status: str
    resolved_target_id: str | None
    source_edit_required: bool
    resolver_change_required: bool
    evidence: tuple[dict[str, Any], ...]
    line: int
    initial_resolution_status: str
    initial_resolved_target_id: str | None


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


def source_markdown_files(source_dir: Path = SOURCE_DIR) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.md") if path.is_file())


def _strip_wrapping_code_ticks(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == "`" and value[-1] == "`":
        value = value[1:-1].strip()
    return value


def _split_obsidian_target(raw: str) -> tuple[str, str, str, str]:
    target = raw.strip()
    alias = ""
    heading = ""
    block = ""
    if "|" in target:
        target, alias = target.split("|", 1)
        alias = alias.strip()
    if "^" in target:
        target, block = target.split("^", 1)
        block = block.strip()
    if "#" in target:
        target, heading = target.split("#", 1)
        heading = heading.strip()
    return target.strip(), heading, block, alias


def _split_markdown_target(raw: str) -> tuple[str, str, str]:
    target = _strip_wrapping_code_ticks(unquote(raw.strip()))
    target = target.split("?", 1)[0].strip()
    heading = ""
    block = ""
    if "^" in target:
        target, block = target.split("^", 1)
        block = block.strip()
    if "#" in target:
        target, heading = target.split("#", 1)
        heading = heading.strip()
    return target.strip(), heading, block


def _is_external(target: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.IGNORECASE))


def _note_key(value: str) -> str:
    value = unquote(value).replace("\\", "/").strip().strip("/")
    if value.endswith(".md"):
        value = value[:-3]
    if value.startswith("source-documents/"):
        value = value.removeprefix("source-documents/")
    if value.startswith("编程/"):
        value = value.removeprefix("编程/")
    return value.lower()


def build_document_index(source_dir: Path = SOURCE_DIR) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in source_markdown_files(source_dir):
        rel = rel_path(path)
        source_relative = rel.removeprefix("source-documents/")
        candidates = {
            source_relative,
            Path(source_relative).with_suffix("").as_posix(),
            path.stem,
            f"编程/{Path(source_relative).with_suffix('').as_posix()}",
        }
        for candidate in candidates:
            key = _note_key(candidate)
            previous = index.get(key)
            if previous is None:
                index[key] = rel
            elif previous != rel:
                index[key] = ""
    return index


def _is_inside_code_fence(text: str, offset: int) -> bool:
    in_fence = False
    cursor = 0
    for line in text.splitlines(keepends=True):
        if cursor >= offset:
            break
        if CODE_FENCE_RE.match(line):
            in_fence = not in_fence
        cursor += len(line)
    return in_fence


def _candidate_targets(target: str, source_document_id: str) -> tuple[str, ...]:
    if not target:
        return ()
    source_parent = Path(source_document_id.removeprefix("source-documents/")).parent
    values = [
        target,
        f"{target}.md" if not target.endswith(".md") else target,
        (source_parent / target).as_posix(),
        (source_parent / f"{target}.md").as_posix() if not target.endswith(".md") else (source_parent / target).as_posix(),
    ]
    if target.startswith("编程/"):
        stripped = target.removeprefix("编程/")
        values.extend([stripped, f"{stripped}.md" if not stripped.endswith(".md") else stripped])
    return tuple(dict.fromkeys(v for v in values if v))


def resolve_link(raw: str, source_document_id: str, link_syntax: str, note_index: dict[str, str]) -> dict[str, Any]:
    if link_syntax == "markdown":
        target, heading, block = _split_markdown_target(raw)
        alias = ""
    else:
        target, heading, block, alias = _split_obsidian_target(raw)
        target = unquote(target)

    normalized = _note_key(target)
    candidates = _candidate_targets(target, source_document_id)
    if not target:
        return {
            "normalized_target": normalized,
            "heading_fragment": heading,
            "block_fragment": block,
            "alias_text": alias,
            "candidate_targets": candidates,
            "root_cause_class": "parser_false_positive",
            "resolution_status": "invalid_link",
            "resolved_target_id": None,
            "resolver_change_required": False,
        }
    if _is_external(target):
        return {
            "normalized_target": target,
            "heading_fragment": heading,
            "block_fragment": block,
            "alias_text": alias,
            "candidate_targets": (),
            "root_cause_class": "non_note_external_link",
            "resolution_status": "out_of_scope_external_target",
            "resolved_target_id": None,
            "resolver_change_required": False,
        }

    matching = [note_index.get(_note_key(candidate)) for candidate in candidates]
    matching = sorted({item for item in matching if item})
    ambiguous = any(note_index.get(_note_key(candidate)) == "" for candidate in candidates)
    if len(matching) == 1 and not ambiguous:
        root_cause = "valid_obsidian_wikilink_not_supported" if link_syntax.startswith("wiki") else "valid_markdown_relative_path_not_supported"
        if target.startswith("编程/"):
            root_cause = "path_normalization_mismatch"
        return {
            "normalized_target": normalized,
            "heading_fragment": heading,
            "block_fragment": block,
            "alias_text": alias,
            "candidate_targets": candidates,
            "root_cause_class": root_cause,
            "resolution_status": "resolved_deterministically",
            "resolved_target_id": matching[0],
            "resolver_change_required": True,
        }
    if len(matching) > 1 or ambiguous:
        return {
            "normalized_target": normalized,
            "heading_fragment": heading,
            "block_fragment": block,
            "alias_text": alias,
            "candidate_targets": candidates,
            "root_cause_class": "ambiguous_target",
            "resolution_status": "requires_owner_decision",
            "resolved_target_id": None,
            "resolver_change_required": False,
        }

    root_cause = "missing_target_document"
    if target.startswith("../"):
        root_cause = "target_outside_corpus_snapshot"
    elif heading:
        root_cause = "valid_heading_fragment_not_supported"
    elif block:
        root_cause = "valid_block_reference_not_supported"
    return {
        "normalized_target": normalized,
        "heading_fragment": heading,
        "block_fragment": block,
        "alias_text": alias,
        "candidate_targets": candidates,
        "root_cause_class": root_cause,
        "resolution_status": "intentionally_unresolved",
        "resolved_target_id": None,
        "resolver_change_required": False,
    }


def collect_link_records(source_dir: Path = SOURCE_DIR) -> list[dict[str, Any]]:
    note_index = build_document_index(source_dir)
    records: list[dict[str, Any]] = []
    ordinal = 0
    for path in source_markdown_files(source_dir):
        source = rel_path(path)
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            ordinal += 1
            raw = match.group(2)
            initial = {
                "resolved_target": None,
                "status": "external_or_empty" if _is_external(raw.strip()) else "unresolved",
            }
            final = resolve_link(raw, source, "markdown", note_index)
            if _is_inside_code_fence(text, match.start()):
                final = {
                    **final,
                    "root_cause_class": "parser_false_positive",
                    "resolution_status": "invalid_link",
                    "resolved_target_id": None,
                    "resolver_change_required": False,
                }
            records.append(_record(ordinal, source, "markdown", raw, match.group(1), match.start(), text, initial, final))
        for match in WIKI_LINK_RE.finditer(text):
            ordinal += 1
            raw = match.group(1)
            syntax = "obsidian_embed" if match.group(0).startswith("!") else "obsidian_wikilink"
            final = resolve_link(raw, source, syntax, note_index)
            if _is_inside_code_fence(text, match.start()):
                final = {
                    **final,
                    "root_cause_class": "parser_false_positive",
                    "resolution_status": "invalid_link",
                    "resolved_target_id": None,
                    "resolver_change_required": False,
                }
            records.append(_record(ordinal, source, syntax, raw, "", match.start(), text, {"resolved_target": None, "status": "unresolved"}, final))
    return records


def _record(
    ordinal: int,
    source: str,
    syntax: str,
    raw: str,
    label: str,
    offset: int,
    text: str,
    initial: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    line = _line_number(text, offset)
    link_id = f"link:{stable_hash({'source': source, 'line': line, 'raw': raw, 'syntax': syntax})[:16]}"
    evidence = [
        {
            "source": source,
            "line": line,
            "label": label,
            "diagnosis": final["root_cause_class"],
        }
    ]
    return {
        "link_id": link_id,
        "source_document_id": source,
        "source_section_id": f"{source}#L{line}",
        "raw_link_text": raw,
        "link_syntax": syntax,
        "normalized_target": final["normalized_target"],
        "heading_fragment": final["heading_fragment"],
        "block_fragment": final["block_fragment"],
        "alias_text": final["alias_text"],
        "candidate_targets": list(final["candidate_targets"]),
        "root_cause_class": final["root_cause_class"],
        "resolution_status": final["resolution_status"],
        "resolved_target_id": final["resolved_target_id"],
        "source_edit_required": False,
        "resolver_change_required": final["resolver_change_required"],
        "evidence": evidence,
        "line": line,
        "initial_resolution_status": initial["status"],
        "initial_resolved_target_id": initial["resolved_target"],
    }


def summarize_links(records: list[dict[str, Any]]) -> dict[str, Any]:
    root_causes: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in records:
        root_causes[row["root_cause_class"]] = root_causes.get(row["root_cause_class"], 0) + 1
        statuses[row["resolution_status"]] = statuses.get(row["resolution_status"], 0) + 1
    final_resolved = sum(1 for row in records if row["resolved_target_id"])
    return {
        "schema_version": "opk-rag.task0080-link-root-cause-summary.v1",
        "task_id": "TASK-0080",
        "initial_resolved_link_count": sum(1 for row in records if row["initial_resolved_target_id"]),
        "initial_unresolved_link_count": len(records),
        "final_resolved_link_count": final_resolved,
        "final_unresolved_link_count": len(records) - final_resolved,
        "newly_resolved_link_count": sum(1 for row in records if row["resolved_target_id"] and not row["initial_resolved_target_id"]),
        "all_unresolved_links_classified": all(row["root_cause_class"] != "unknown" for row in records),
        "root_cause_counts": dict(sorted(root_causes.items())),
        "resolution_status_counts": dict(sorted(statuses.items())),
        "remaining_unresolved_links": [
            {
                "link_id": row["link_id"],
                "source_document_id": row["source_document_id"],
                "raw_link_text": row["raw_link_text"],
                "root_cause_class": row["root_cause_class"],
                "resolution_status": row["resolution_status"],
            }
            for row in records
            if not row["resolved_target_id"]
        ],
    }


def build_transition(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        rows.append(
            {
                "link_id": row["link_id"],
                "raw_link_text": row["raw_link_text"],
                "source_document_id": row["source_document_id"],
                "previous_resolution_status": row["initial_resolution_status"],
                "final_resolution_status": row["resolution_status"],
                "resolved_target_id": row["resolved_target_id"],
                "root_cause_class": row["root_cause_class"],
                "resolver_change_applied": row["resolver_change_required"],
                "source_edit_applied": False,
                "transition_digest": stable_hash(
                    {
                        "link_id": row["link_id"],
                        "target": row["resolved_target_id"],
                        "status": row["resolution_status"],
                    }
                ),
            }
        )
    return rows
