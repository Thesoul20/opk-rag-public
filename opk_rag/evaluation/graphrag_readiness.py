from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from opk_rag.chunking.chunker import chunk_markdown_document
from opk_rag.chunking.parser import MarkdownParseError, parse_markdown_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "source-documents"
BENCHMARK_DIR = ROOT / "evaluation-data" / "core-rag-benchmark-v1"
RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0079_graphrag_readiness_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0079_GRAPHRAG_INTEGRATION_BOUNDARY_AND_READINESS_REPORT.md"

TASK0078_INPUTS = (
    ROOT / "tasks" / "TASK-0078-close-repository-verification-and-freeze-single-agent-agentic-rag-v1.md",
    ROOT / "evaluation-data" / "contracts" / "task0078_single_agent_rag_v1_freeze_contract.json",
    ROOT / "evaluation-data" / "results" / "task0078-single-agent-rag-v1-freeze" / "summary.json",
    ROOT
    / "evaluation-data"
    / "results"
    / "task0078-single-agent-rag-v1-freeze"
    / "single_agent_rag_v1_freeze_inventory.json",
    ROOT / "docs" / "TASK0078_SINGLE_AGENT_AGENTIC_RAG_V1_FREEZE_REPORT.md",
    BENCHMARK_DIR / "benchmark_manifest.json",
    BENCHMARK_DIR / "question_set.jsonl",
    BENCHMARK_DIR / "annotations.jsonl",
    ROOT / "evaluation-data" / "contracts" / "task0071_governed_agent_recovery_contract.json",
    ROOT / "evaluation-data" / "contracts" / "task0073_post_generation_observability_contract.json",
    ROOT / "evaluation-data" / "contracts" / "task0074_reference_runtime_stability_contract.json",
    ROOT / "evaluation-data" / "contracts" / "task0076_governed_generation_retry_contract.json",
    ROOT / "evaluation-data" / "contracts" / "task0078_single_agent_rag_v1_freeze_contract.json",
)

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
WIKI_LINK_RE = re.compile(r"!?\[\[([^\]\n]+)\]\]")
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_\-/\u4e00-\u9fff]+)")
EXPLICIT_REFERENCE_RE = re.compile(r"(?i)\b(?:see also|reference|references|ref|参考|引用|来源)\b")
RELATION_FRONTMATTER_KEYS = {
    "aliases",
    "alias",
    "tags",
    "tag",
    "related",
    "links",
    "references",
    "source",
    "parent",
    "children",
}


@dataclass(frozen=True)
class DocumentAudit:
    path: str
    title: str | None
    section_count: int
    chunk_count: int
    markdown_links: tuple[dict[str, Any], ...]
    wiki_links: tuple[dict[str, Any], ...]
    embeds: tuple[dict[str, Any], ...]
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    frontmatter_relations: tuple[dict[str, Any], ...]
    explicit_reference_count: int


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{relative_path(path)}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


def _git(args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def _path_modified(path: Path) -> bool:
    rel = relative_path(path)
    output = _git(["status", "--short", "--untracked-files=all", "--", rel])
    return bool(output.strip())


def build_preflight() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0079-preflight.v1",
        "task_id": "TASK-0079",
        "git_status_short_untracked_all": _git(["status", "--short", "--untracked-files=all"]),
        "git_diff_stat": _git(["diff", "--stat"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_head": _git(["rev-parse", "HEAD"]),
        "forbidden_git_mutations_executed": False,
    }


def validate_task0078_inputs() -> tuple[dict[str, Any], bool, bool]:
    input_hashes: dict[str, Any] = {
        "schema_version": "opk-rag.task0079-input-hashes.v1",
        "task_id": "TASK-0079",
        "immutable_inputs": [],
    }
    valid = True
    modified = False
    seen: set[str] = set()
    for path in TASK0078_INPUTS:
        rel = relative_path(path)
        if rel in seen:
            continue
        seen.add(rel)
        exists = path.exists()
        parsed = False
        parse_error = None
        if exists:
            try:
                if path.suffix == ".json":
                    read_json(path)
                elif path.suffix == ".jsonl":
                    read_jsonl(path)
                else:
                    path.read_text(encoding="utf-8")
                parsed = True
            except Exception as exc:  # pragma: no cover - surfaced in artifact
                parse_error = str(exc)
        status_modified = _path_modified(path) if exists else True
        modified = modified or status_modified
        valid = valid and exists and parsed
        input_hashes["immutable_inputs"].append(
            {
                "path": rel,
                "exists": exists,
                "parseable": parsed,
                "sha256": file_digest(path) if exists else None,
                "git_status_modified": status_modified,
                "parse_error": parse_error,
            }
        )

    summary = read_json(ROOT / "evaluation-data" / "results" / "task0078-single-agent-rag-v1-freeze" / "summary.json")
    contract = read_json(ROOT / "evaluation-data" / "contracts" / "task0078_single_agent_rag_v1_freeze_contract.json")
    freeze_ok = summary.get("single_agent_rag_v1_freeze_decision") == "freeze"
    retry_ok = contract.get("generation_retry_default_enabled") is False and contract.get("generation_retry_promoted") is False
    valid = valid and freeze_ok and retry_ok
    input_hashes["task0078_inputs_valid"] = valid
    input_hashes["task0078_inputs_modified"] = modified
    input_hashes["single_agent_rag_v1_freeze_decision"] = summary.get("single_agent_rag_v1_freeze_decision")
    input_hashes["generation_retry_default_enabled"] = contract.get("generation_retry_default_enabled")
    input_hashes["generation_retry_promoted"] = contract.get("generation_retry_promoted")
    return input_hashes, valid, modified


def _markdown_files(source_dir: Path = SOURCE_DIR) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.md") if path.is_file())


def _normalize_note_key(value: str) -> str:
    value = value.strip().split("#", 1)[0].split("|", 1)[0].strip()
    if value.endswith(".md"):
        value = value[:-3]
    return value.replace("\\", "/").strip("/").lower()


def _frontmatter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_frontmatter_values(item))
        return values
    if isinstance(value, dict):
        return [f"{key}:{item}" for key, item in sorted(value.items())]
    return [str(value)]


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _resolve_target(raw_target: str, source_path: str, note_index: dict[str, str]) -> tuple[str | None, str]:
    target = raw_target.strip().split("?", 1)[0].split("#", 1)[0]
    if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.IGNORECASE):
        return None, "external_or_empty"
    source_parent = Path(source_path).parent
    candidates = []
    if target.endswith(".md"):
        candidates.append((source_parent / target).as_posix())
        candidates.append(target)
    else:
        candidates.append((source_parent / f"{target}.md").as_posix())
        candidates.append(f"{target}.md")
        normalized = _normalize_note_key(target)
        if normalized in note_index:
            return note_index[normalized], "resolved_by_note_name"
    for candidate in candidates:
        normalized_candidate = str(Path(candidate)).replace("\\", "/").strip("./")
        if normalized_candidate in set(note_index.values()):
            return normalized_candidate, "resolved_by_relative_path"
    key = _normalize_note_key(target)
    return (note_index[key], "resolved_by_normalized_path") if key in note_index else (None, "unresolved")


def audit_corpus_graph_readiness(source_dir: Path = SOURCE_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    documents: list[DocumentAudit] = []
    parse_errors: list[dict[str, str]] = []
    files = _markdown_files(source_dir)
    note_index: dict[str, str] = {}
    for path in files:
        rel = relative_path(path)
        note_index[_normalize_note_key(Path(rel).with_suffix("").as_posix())] = rel
        note_index[_normalize_note_key(path.stem)] = rel

    edge_rows: list[dict[str, Any]] = []
    graph_edges: list[tuple[str, str, str, str]] = []
    doc_paths = {relative_path(path) for path in files}
    tags_by_doc: dict[str, set[str]] = {}

    for path in files:
        rel = relative_path(path)
        text = path.read_text(encoding="utf-8")
        try:
            parsed = parse_markdown_file(path)
            chunks = chunk_markdown_document(parsed)
        except MarkdownParseError as exc:
            parse_errors.append({"path": rel, "error": str(exc)})
            continue

        md_links = []
        wiki_links = []
        embeds = []
        tags = sorted(set(match.group(1).strip("/") for match in TAG_RE.finditer(text)))
        aliases = []
        frontmatter_relations = []
        for key, value in sorted(parsed.frontmatter.items()):
            values = _frontmatter_values(value)
            if key.lower() in {"aliases", "alias"}:
                aliases.extend(values)
            if key.lower() in {"tags", "tag"}:
                tags.extend(item.lstrip("#") for item in values)
            if key.lower() in RELATION_FRONTMATTER_KEYS:
                frontmatter_relations.append({"key": key, "value_count": len(values), "values": values[:20]})

        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group(2)
            resolved, status = _resolve_target(target, rel, note_index)
            row = {
                "label": match.group(1),
                "target": target,
                "resolved_target": resolved,
                "resolution_status": status,
                "line": _line_number(text, match.start()),
            }
            md_links.append(row)
            edge_rows.append({"authority_level": "G1", "edge_type": "LINKS_TO", "source": rel, **row})
            if resolved:
                graph_edges.append((rel, "LINKS_TO", resolved, "G1"))

        for match in WIKI_LINK_RE.finditer(text):
            raw = match.group(1)
            resolved, status = _resolve_target(raw, rel, note_index)
            row = {
                "target": raw,
                "resolved_target": resolved,
                "resolution_status": status,
                "line": _line_number(text, match.start()),
            }
            if match.group(0).startswith("!"):
                embeds.append(row)
                edge_type = "EMBEDS"
            else:
                wiki_links.append(row)
                edge_type = "LINKS_TO"
            edge_rows.append({"authority_level": "G1", "edge_type": edge_type, "source": rel, **row})
            if resolved:
                graph_edges.append((rel, edge_type, resolved, "G1"))

        for section in parsed.sections:
            graph_edges.append((rel, "CONTAINS", f"{rel}#L{section.start_line}", "G0"))
        for tag in sorted(set(tags)):
            normalized = tag.lower()
            graph_edges.append((rel, "TAGGED_WITH", f"tag:{normalized}", "G1"))
        for alias in sorted(set(aliases)):
            graph_edges.append((rel, "ALIASES", f"alias:{alias.lower()}", "G1"))

        tags_by_doc[rel] = set(tags)
        documents.append(
            DocumentAudit(
                path=rel,
                title=parsed.title,
                section_count=len(parsed.sections),
                chunk_count=len(chunks),
                markdown_links=tuple(md_links),
                wiki_links=tuple(wiki_links),
                embeds=tuple(embeds),
                tags=tuple(sorted(set(tags))),
                aliases=tuple(sorted(set(aliases))),
                frontmatter_relations=tuple(frontmatter_relations),
                explicit_reference_count=len(EXPLICIT_REFERENCE_RE.findall(text)),
            )
        )

    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, _edge_type, target, _authority in graph_edges:
        if source in doc_paths and target in doc_paths:
            adjacency[source].add(target)
            adjacency[target].add(source)
    components = _components(sorted(doc_paths), adjacency)
    edge_counts = Counter(edge_type for _source, edge_type, _target, _authority in graph_edges)
    authority_counts = Counter(authority for _source, _edge_type, _target, authority in graph_edges)
    resolved_link_count = sum(1 for row in edge_rows if row.get("resolved_target"))
    unresolved_link_count = sum(1 for row in edge_rows if not row.get("resolved_target"))

    target_counts = Counter(row["target"] for row in edge_rows)
    alias_counts = Counter(alias.lower() for doc in documents for alias in doc.aliases)
    high_degree = sorted(
        (
            {"path": doc, "degree": len(neighbors)}
            for doc, neighbors in adjacency.items()
            if len(neighbors) >= 3
        ),
        key=lambda row: (-row["degree"], row["path"]),
    )

    corpus = {
        "schema_version": "opk-rag.task0079-corpus-graph-readiness.v1",
        "task_id": "TASK-0079",
        "source_dir": relative_path(source_dir),
        "document_count": len(documents),
        "section_count": sum(doc.section_count for doc in documents),
        "chunk_count": sum(doc.chunk_count for doc in documents),
        "markdown_link_count": sum(len(doc.markdown_links) for doc in documents),
        "wiki_link_count": sum(len(doc.wiki_links) for doc in documents),
        "embed_link_count": sum(len(doc.embeds) for doc in documents),
        "resolved_link_count": resolved_link_count,
        "unresolved_link_count": unresolved_link_count,
        "tag_count": len({tag.lower() for doc in documents for tag in doc.tags}),
        "frontmatter_relation_count": sum(len(doc.frontmatter_relations) for doc in documents),
        "alias_count": len({alias.lower() for doc in documents for alias in doc.aliases}),
        "explicit_reference_count": sum(doc.explicit_reference_count for doc in documents),
        "explicit_graph_edge_count": len(graph_edges),
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "authority_level_counts": dict(sorted(authority_counts.items())),
        "duplicate_link_targets": [
            {"target": target, "count": count} for target, count in sorted(target_counts.items()) if count > 1
        ],
        "ambiguous_aliases": [
            {"alias": alias, "count": count} for alias, count in sorted(alias_counts.items()) if count > 1
        ],
        "unresolved_relative_paths": [
            {"source": row["source"], "target": row["target"], "line": row["line"]}
            for row in edge_rows
            if row["resolution_status"] == "unresolved"
        ],
        "self_links": [
            {"source": row["source"], "target": row["target"], "line": row["line"]}
            for row in edge_rows
            if row.get("resolved_target") == row["source"]
        ],
        "broken_links": [
            {"source": row["source"], "target": row["target"], "line": row["line"]}
            for row in edge_rows
            if row["resolution_status"] == "unresolved"
        ],
        "orphan_documents": sorted(doc for doc in doc_paths if not adjacency.get(doc)),
        "disconnected_component_count": len(components),
        "disconnected_components": components,
        "high_degree_documents": high_degree[:20],
        "relation_types_with_insufficient_support": [
            {"edge_type": edge_type, "count": count}
            for edge_type, count in sorted(edge_counts.items())
            if count < 2
        ],
        "parse_errors": parse_errors,
        "document_summaries": [doc.__dict__ for doc in documents],
    }
    authority = {
        "schema_version": "opk-rag.task0079-graph-edge-authority-audit.v1",
        "task_id": "TASK-0079",
        "authority_levels": {
            "G0": "explicit_structural_edge",
            "G1": "explicit_authored_edge",
            "G2": "deterministically_derived_edge",
            "G3": "model_inferred_edge",
            "unsupported": "unsupported_candidate_edge",
        },
        "counts": {
            "explicit_structural_edge": authority_counts.get("G0", 0),
            "explicit_authored_edge": authority_counts.get("G1", 0),
            "deterministically_derived_edge": 0,
            "model_inferred_edge": 0,
            "unsupported_candidate_edge": 0,
        },
        "minimal_baseline_allowed_authority_levels": ["G0", "G1", "G2"],
        "excluded_from_task0079_baseline": ["G3", "unsupported"],
        "sample_edges": [
            {"source": source, "edge_type": edge_type, "target": target, "authority_level": authority}
            for source, edge_type, target, authority in sorted(graph_edges)[:50]
        ],
    }
    return corpus, authority


def _components(nodes: list[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item[0]))


def build_agent_rag_integration_audit() -> dict[str, Any]:
    points = [
        (
            "before_initial_retrieval",
            ["opk_rag/agent/runtime.py", "opk_rag/agent/policy.py"],
            "May find relation paths before seeds exist.",
            "Requires query-to-graph interpretation before evidence authority exists.",
            "new default decision branch",
            False,
        ),
        (
            "after_lexical_vector_seed_retrieval",
            ["opk_rag/agent/retrieval_round.py", "opk_rag/core_tools/tools.py"],
            "Uses existing retrieved chunks/documents as bounded seeds and leaves initial retrieval unchanged.",
            "Can amplify poor seeds; requires snapshot/corpus identity checks.",
            "optional tool after governed retrieval",
            True,
        ),
        (
            "during_query_reformulation",
            ["opk_rag/agent/query_reformulation.py"],
            "Could generate relation-aware subqueries.",
            "Entangles graph behavior with model reformulation and weakens attribution.",
            "changes reformulation semantics",
            False,
        ),
        (
            "separate_agent_action",
            ["opk_rag/agent/contracts.py", "opk_rag/agent/tool_registry.py"],
            "Clean governance and trace boundary.",
            "Requires expanding available actions and policy prompts.",
            "new disabled action only",
            False,
        ),
        (
            "candidate_expansion",
            ["opk_rag/search/service.py", "opk_rag/evaluation/retrieval.py"],
            "Adds graph neighbors to existing candidates without replacing retrieval.",
            "Needs deduplication and provenance to avoid treating paths as proof.",
            "minimal when opt-in",
            True,
        ),
        (
            "evidence_path_validation",
            ["opk_rag/answer/grounding.py", "opk_rag/answer/validation.py"],
            "Can explain why evidence units are related.",
            "May confuse graph connectivity with answer support.",
            "post-retrieval diagnostics",
            False,
        ),
        (
            "generation_context_organizer",
            ["opk_rag/answer/evidence_context.py"],
            "Can group evidence by document path and relation path.",
            "Risks changing prompt ordering and frozen generation behavior.",
            "generation prompt impact",
            False,
        ),
    ]
    return {
        "schema_version": "opk-rag.task0079-agent-rag-integration-audit.v1",
        "task_id": "TASK-0079",
        "current_execution_path": [
            "User Question",
            "Agent State",
            "Agent Policy",
            "Governed Retrieval Tool",
            "Answerability",
            "Query Reformulation",
            "Multi-Round Retrieval",
            "Generation",
            "Citation Validation",
            "Grounding Validation",
            "Final Answer or Abstention",
        ],
        "integration_points": [
            {
                "integration_point": name,
                "required_code_surfaces": surfaces,
                "advantages": [advantage],
                "risks": [risk],
                "agent_policy_impact": policy_impact,
                "trace_impact": "must record graph eligibility, seeds, edge allowlist, hops, traversed edges, returned evidence, stop reason, and snapshot digest",
                "evaluation_impact": "must compare frozen G0 against opt-in G1 on graph-sensitive slice and negative controls",
                "recommended_for_minimal_baseline": recommended,
            }
            for name, surfaces, advantage, risk, policy_impact, recommended in points
        ],
        "preferred_minimal_integration_boundary": "after_lexical_vector_seed_retrieval_as_opt_in_candidate_expansion",
        "preferred_boundary_reason": "It preserves the frozen initial retrieval and Agent policy while allowing deterministic explicit-edge expansion from already governed evidence seeds.",
    }


def classify_benchmark_coverage() -> dict[str, Any]:
    questions = read_jsonl(BENCHMARK_DIR / "question_set.jsonl")
    annotations = {row["sample_id"]: row for row in read_jsonl(BENCHMARK_DIR / "annotations.jsonl")}
    rows = []
    counts: Counter[str] = Counter()
    graph_sensitive = 0
    negative = 0
    explicit_edges = 0
    g3_required = 0
    for question in questions:
        annotation = annotations[question["sample_id"]]
        required = annotation.get("required_evidence") or []
        required_docs = {
            item.get("identity", {}).get("document_identity_digest")
            for item in required
            if item.get("identity", {}).get("document_identity_digest")
        }
        qtype = question["question_type"]
        text = question["question"]
        if question["answerability_label"] == "unanswerable" and qtype in {"no_evidence", "false_premise"}:
            category = "graph_negative_control"
            negative += 1
        elif len(required_docs) > 1:
            category = "cross_document_synthesis"
        elif len(required) > 1:
            category = "single_document_multi_section"
        else:
            category = "single_unit_lookup"
        if any(term in text.lower() for term in ("关联", "引用", "链接", "related", "reference", "follow")):
            category = "explicit_link_traversal"
            graph_sensitive += 1
            explicit_edges += 1
        elif category in {"cross_document_synthesis"}:
            graph_sensitive += 1
        if "推断" in text or "是否可以推断" in text:
            g3_required += 1
        counts[category] += 1
        rows.append(
            {
                "sample_id": question["sample_id"],
                "dataset_role": question["dataset_role"],
                "question_type": qtype,
                "answerability_label": question["answerability_label"],
                "graph_coverage_category": category,
                "required_evidence_unit_count": len(required),
                "required_document_count": len(required_docs),
                "graph_sensitive_candidate": category in {"explicit_link_traversal", "cross_document_synthesis"},
                "requires_inferred_g3_relationship": "推断" in text or "是否可以推断" in text,
            }
        )
    decision = (
        "existing_benchmark_partially_sufficient"
        if graph_sensitive > 0
        else "new_graph_sensitive_slice_required"
    )
    return {
        "schema_version": "opk-rag.task0079-benchmark-graph-coverage.v1",
        "task_id": "TASK-0079",
        "benchmark_id": "core-rag-benchmark-v1",
        "total_samples": len(rows),
        "samples_per_category": dict(sorted(counts.items())),
        "graph_sensitive_candidate_count": graph_sensitive,
        "graph_negative_control_count": negative,
        "samples_with_explicit_traversable_edges": explicit_edges,
        "samples_requiring_inferred_g3_relationships": g3_required,
        "samples_unsuitable_for_graphrag_evaluation": counts.get("single_unit_lookup", 0),
        "coverage_decision": decision,
        "decision_rationale": "The frozen benchmark has some cross-document and negative-control coverage, but no authoritative sample explicitly requires traversing a corpus graph path.",
        "sample_classifications": rows,
    }


def build_failure_mapping(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in coverage["sample_classifications"]:
        if sample["graph_coverage_category"] == "cross_document_synthesis":
            failure_class = "multi_hop_evidence_candidate"
            plausibly_required = False
            relationships = ["shared answer topic across multiple required documents; explicit graph path not authored in frozen annotations"]
        elif sample["graph_coverage_category"] == "explicit_link_traversal":
            failure_class = "relationship_traversal_candidate"
            plausibly_required = True
            relationships = ["question text indicates reference/link traversal, but frozen annotation lacks path authority"]
        elif sample["graph_coverage_category"] == "graph_negative_control":
            failure_class = "not_graph_addressable"
            plausibly_required = False
            relationships = []
        elif sample["required_evidence_unit_count"] > 1:
            failure_class = "chunk_localization_failure"
            plausibly_required = False
            relationships = []
        else:
            failure_class = "not_graph_addressable"
            plausibly_required = False
            relationships = []
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "current_failure_class": failure_class,
                "required_evidence_units": [],
                "required_relationships": relationships,
                "relationship_authority_level": "unknown_without_source_path_annotation"
                if relationships
                else "not_applicable",
                "single_chunk_answerable": sample["required_evidence_unit_count"] <= 1,
                "ordinary_multi_retrieval_answerable": sample["graph_coverage_category"]
                in {"single_document_multi_section", "cross_document_synthesis"},
                "graph_retrieval_plausibly_required": plausibly_required,
                "evidence": [
                    {
                        "source": "frozen Core RAG Benchmark v1 graph coverage audit",
                        "graph_coverage_category": sample["graph_coverage_category"],
                    }
                ],
            }
        )
    return rows


def build_minimal_graph_schema() -> dict[str, Any]:
    node_types = [
        ("DocumentNode", "G0", "source relative path + file digest", True, "doc:sha256(relative_path)", True, False, ""),
        ("SectionNode", "G0", "heading path + line span", True, "section:sha256(document_id + heading_path + start_line)", True, False, ""),
        ("ChunkNode", "G0", "deterministic chunker output", True, "chunk:sha256(document_id + chunk_index + content_hash)", True, False, ""),
        ("TagNode", "G1", "frontmatter tags and inline tags", True, "tag:normalized_tag", True, False, ""),
        ("AliasNode", "G1", "frontmatter aliases", True, "alias:normalized_alias", False, False, "optional until alias support is observed as useful"),
        ("ReferenceTargetNode", "G1", "markdown/wiki link target", True, "ref:sha256(normalized_target)", True, False, ""),
        ("EntityNode", "G3", "LLM or NLP entity extraction", False, "entity:extraction_policy_digest + normalized_entity", False, True, "requires model-inferred authority"),
    ]
    edge_types = [
        ("CONTAINS", "G0", "parser section and chunk hierarchy", True, True, False, ""),
        ("BELONGS_TO", "G2", "reciprocal of CONTAINS", True, True, False, ""),
        ("PARENT_OF", "G0", "heading hierarchy", True, True, False, ""),
        ("NEXT_SECTION", "G0", "section ordering", True, True, False, ""),
        ("LINKS_TO", "G1/G2", "markdown/wiki links with deterministic resolution", True, True, False, ""),
        ("EMBEDS", "G1/G2", "Obsidian embed links", True, True, False, ""),
        ("TAGGED_WITH", "G1/G2", "frontmatter and inline tags", True, True, False, ""),
        ("ALIASES", "G1/G2", "frontmatter aliases", True, False, False, "optional for first scaffold"),
        ("REFERENCES", "G1", "explicit authored reference lists/citations", True, False, False, "requires stricter parser support"),
        ("MENTIONS", "G3", "entity mention extraction", False, False, True, "model or NLP inferred"),
        ("RELATED_TO", "G3", "semantic relatedness", False, False, True, "unsupported without inference evaluation"),
    ]
    return {
        "schema_version": "opk-rag.task0079-minimal-graph-schema.v1",
        "task_id": "TASK-0079",
        "node_types": [
            {
                "type": t,
                "authority_level": authority,
                "source_field": source,
                "deterministic": deterministic,
                "stable_identifier_rule": rule,
                "required_for_minimal_baseline": required,
                "experimental_only": experimental,
                "exclusion_reason": reason,
            }
            for t, authority, source, deterministic, rule, required, experimental, reason in node_types
        ],
        "edge_types": [
            {
                "type": t,
                "authority_level": authority,
                "source_field": source,
                "deterministic": deterministic,
                "stable_identifier_rule": f"edge:sha256(source_id + type + target_id + provenance_span)",
                "required_for_minimal_baseline": required,
                "experimental_only": experimental,
                "exclusion_reason": reason,
            }
            for t, authority, source, deterministic, required, experimental, reason in edge_types
        ],
        "minimal_schema_summary": "Use DocumentNode, SectionNode, ChunkNode, TagNode, ReferenceTargetNode with CONTAINS, BELONGS_TO, PARENT_OF, NEXT_SECTION, LINKS_TO, EMBEDS, and TAGGED_WITH edges.",
    }


def build_graph_snapshot_contract(corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0079-graph-snapshot-contract.v1",
        "task_id": "TASK-0079",
        "snapshot_fields": {
            "schema_version": "opk-rag.graph-snapshot.v1",
            "corpus_snapshot_id": "active corpus binding id",
            "source_manifest_digest": "sha256 canonical source manifest",
            "graph_build_policy": "deterministic_g0_g2_explicit_edges_only",
            "node_count": 0,
            "edge_count": 0,
            "node_type_counts": {},
            "edge_type_counts": {},
            "nodes_digest": "sha256 canonical ordered nodes",
            "edges_digest": "sha256 canonical ordered edges",
            "build_timestamp_policy": "metadata only; excluded from deterministic snapshot digest",
            "deterministic_rebuild_expected": True,
        },
        "diagnostic_non_authoritative_counts": {
            "document_count": corpus["document_count"],
            "explicit_graph_edge_count": corpus["explicit_graph_edge_count"],
        },
        "node_serialization": "one JSON object per node, ordered by node_id, with source provenance and authority_level",
        "edge_serialization": "one JSON object per edge, ordered by edge_id, with source_id, target_id, edge_type, authority_level, and source span",
        "ordering_rules": ["sort nodes by node_id", "sort edges by edge_id", "sort object keys for digest"],
        "digest_calculation": "sha256 over UTF-8 canonical JSON with sort_keys and compact separators",
        "duplicate_handling": "collapse exact duplicate edge identities and retain all provenance spans",
        "deleted_note_handling": "remove nodes and incident edges during rebuild; incremental mode emits tombstones until compaction",
        "renamed_note_handling": "treat path rename as new DocumentNode unless source manifest supplies stable identity mapping",
        "unresolved_link_handling": "retain ReferenceTargetNode with unresolved status; do not return as evidence",
        "incremental_rebuild_behavior": "recompute changed document subgraph and affected deterministic link resolutions",
        "graph_schema_migration_policy": "schema_version bump plus migration report; old snapshots remain read-only",
    }


def build_graph_retrieval_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0079-graph-retrieval-contract.v1",
        "task_id": "TASK-0079",
        "request_contract": {
            "query": "string",
            "seed_unit_ids": ["string"],
            "seed_document_ids": ["string"],
            "allowed_edge_types": ["CONTAINS", "BELONGS_TO", "PARENT_OF", "NEXT_SECTION", "LINKS_TO", "EMBEDS", "TAGGED_WITH"],
            "max_hops": 2,
            "max_nodes": 50,
            "max_edges": 100,
            "direction": "outbound|inbound|both",
            "require_explicit_edges": True,
            "trace_id": "string",
        },
        "response_contract": {
            "seed_nodes": [],
            "visited_nodes": [],
            "traversed_edges": [],
            "evidence_units": [],
            "evidence_paths": [],
            "truncated": False,
            "stop_reason": "completed|max_hops|max_nodes|max_edges|snapshot_mismatch|no_seed|edge_type_not_allowed",
            "warnings": [],
            "provenance": [],
        },
        "safety_requirements": [
            "unbounded traversal is forbidden",
            "G3 and unsupported inferred relationships are rejected by default",
            "every evidence unit must map back to retrievable source content",
            "graph path connectivity is not proof of answer support",
            "Answerability, Citation, and Grounding validation remain mandatory",
        ],
    }


def build_agent_graph_integration_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0079-agent-graph-integration-contract.v1",
        "task_id": "TASK-0079",
        "candidate_action_name": "graph_expand_evidence",
        "enabled_by_default": False,
        "accepts": {
            "raw_user_query": False,
            "reformulated_queries": False,
            "retrieval_seed_ids": True,
            "explicit_relation_constraints": True,
            "agent_selected_traversal_objective": False,
        },
        "eligibility_conditions": [
            "ordinary retrieval produced seed evidence IDs",
            "question or diagnostic classifier marks relation traversal as relevant",
            "allowed edge types are G0-G2 only",
            "graph snapshot digest matches active corpus snapshot",
            "graph budget remains available",
        ],
        "non_invocation_conditions": [
            "no graph-relevant relation is identified",
            "ordinary retrieval already provides sufficient evidence",
            "request is a simple single-unit lookup",
            "traversal would require unsupported inferred edges",
            "maximum graph budget has been exhausted",
            "graph snapshot does not match active corpus snapshot",
        ],
        "trace_fields": [
            "graph_action_eligible",
            "graph_action_invoked",
            "graph_seed_source",
            "graph_allowed_edge_types",
            "graph_max_hops",
            "graph_nodes_visited",
            "graph_edges_traversed",
            "graph_evidence_units_returned",
            "graph_stop_reason",
            "graph_snapshot_digest",
        ],
    }


def build_candidate_graph_evaluation_spec() -> dict[str, Any]:
    categories = [
        ("explicit-wiki-link-001", "explicit wiki-link traversal", "follow_explicit_link"),
        ("cross-document-reference-001", "cross-document reference following", "follow_reference"),
        ("document-section-linked-document-001", "document section linked document traversal", "expand_from_section_seed"),
        ("two-hop-composition-001", "two-hop evidence composition", "bounded_two_hop_expand"),
        ("entity-linked-notes-001", "entity-centric aggregation from explicitly linked notes", "aggregate_explicit_neighbors"),
        ("temporal-chain-001", "temporal chain across explicitly related notes", "follow_authored_sequence"),
        ("negative-control-001", "negative controls where graph traversal should not be used", "do_not_invoke_graph"),
        ("graph-unanswerable-001", "graph-unanswerable questions", "abstain"),
        ("broken-link-safety-001", "broken-link safety cases", "stop_with_warning"),
        ("cyclic-link-limit-001", "cyclic-link traversal limits", "stop_at_budget"),
    ]
    samples = []
    for candidate_id, question_type, expected_action in categories:
        samples.append(
            {
                "candidate_id": candidate_id,
                "status": "candidate_only",
                "question": f"Candidate {question_type} question requiring owner-authored source paths.",
                "question_type": question_type,
                "expected_action": expected_action,
                "answerability_label": "answerable" if "unanswerable" not in candidate_id and "negative" not in candidate_id else "unanswerable",
                "required_source_units": [],
                "required_graph_path": [],
                "allowed_edge_types": ["LINKS_TO", "CONTAINS", "BELONGS_TO"],
                "maximum_required_hops": 2,
                "forbidden_claims": ["Do not infer semantic relationships not present as G0-G2 edges."],
                "negative_control": "negative" in candidate_id or "broken" in candidate_id or "cyclic" in candidate_id,
                "owner_review_required": True,
            }
        )
    return {
        "schema_version": "opk-rag.task0079-candidate-graph-evaluation-spec.v1",
        "task_id": "TASK-0079",
        "status": "candidate_only",
        "non_authoritative": True,
        "not_for_promotion": True,
        "candidate_samples": samples,
    }


def build_promotion_gates() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0079-graphrag-promotion-gates.v1",
        "task_id": "TASK-0079",
        "retrieval_gates": {
            "graph_sensitive_evidence_recall": "owner_threshold_required",
            "multi_hop_required_evidence_recovery": "owner_threshold_required",
            "graph_negative_control_unnecessary_invocation_rate": "owner_threshold_required",
            "ordinary_retrieval_metric_regression": "owner_threshold_required",
        },
        "answer_gates": {
            "end_to_end_correctness_regression": "owner_threshold_required",
            "safe_action_regression": "owner_threshold_required",
            "over_abstention_regression": "owner_threshold_required",
            "unsupported_answer_rate_regression": "must_not_increase",
            "citation_validity": "retain_frozen_threshold",
            "grounding_validity": "retain_frozen_threshold",
        },
        "runtime_gates": {
            "bounded_traversal": "required",
            "cycle_safety": "required",
            "snapshot_mismatch_safe_failure": "required",
            "deterministic_rebuild": "required",
            "incremental_update_reproducible": "required",
            "p95_latency": "owner_threshold_required",
            "memory_usage": "owner_threshold_required",
        },
        "governance_gates": {
            "opt_in_until_promotion": True,
            "source_provenance_required": True,
            "traceable_graph_paths_required": True,
            "g3_edges_excluded": True,
            "frozen_g0_reproducible": True,
        },
    }


def build_next_task_recommendation(corpus: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    graph_signal = corpus["explicit_graph_edge_count"] > 0 and corpus["section_count"] > 0
    eval_gap = coverage["coverage_decision"] != "existing_benchmark_sufficient"
    if graph_signal and eval_gap:
        decision = "benchmark_gap_must_be_closed_first"
    elif not graph_signal:
        decision = "corpus_graph_signal_insufficient"
    else:
        decision = "proceed_to_minimal_graph_index_scaffold"
    return {
        "schema_version": "opk-rag.task0079-next-task-recommendation.v1",
        "task_id": "TASK-0079",
        "next_task_decision": decision,
        "recommended_task0080_scope_high_level": [
            "deterministic local graph schema",
            "explicit structural and authored edges",
            "snapshot builder and serialization",
            "incremental-update contract",
            "targeted unit tests",
            "no default Agent integration",
            "no Provider calls",
            "no formal promotion experiment",
        ],
        "task0080_task_card_created": False,
        "rationale": "The corpus has deterministic graph signal, but the frozen benchmark lacks authoritative graph-path samples for promotion.",
    }


def build_contract(summary: dict[str, Any], next_task: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0079-graphrag-readiness-contract.v1",
        "task_id": "TASK-0079",
        "task_status": "complete",
        "task0078_inputs_valid": summary["task0078_inputs_valid"],
        "task0078_inputs_modified": summary["task0078_inputs_modified"],
        "single_agent_rag_v1_modified": False,
        "generation_retry_policy_modified": False,
        "generation_retry_default_enabled": False,
        "generation_retry_promoted": False,
        "external_provider_calls": 0,
        "external_database_benchmark_calls": 0,
        "new_vector_retrieval_runs": 0,
        "new_formal_replicates_run": 0,
        "llm_entity_extraction_calls": 0,
        "corpus_graph_readiness_audited": True,
        "benchmark_graph_coverage_audited": True,
        "minimal_graph_schema_defined": True,
        "graph_snapshot_contract_defined": True,
        "graph_retrieval_contract_defined": True,
        "agent_graph_integration_contract_defined": True,
        "candidate_graph_evaluation_spec_created": True,
        "graphrag_promotion_gates_defined": True,
        "minimal_graph_baseline_default_enabled": False,
        "next_task_decision": next_task["next_task_decision"],
    }


def build_report(summary: dict[str, Any], next_task: dict[str, Any]) -> str:
    return f"""# TASK-0079 GraphRAG Integration Boundary and Readiness Report

## Decision

`next_task_decision={next_task["next_task_decision"]}`

GraphRAG is justified only as a bounded retrieval capability for explicit relationship traversal. It is not justified as a production runtime or as a model-inferred knowledge graph in TASK-0079.

## Corpus Signal

The deterministic corpus audit found {summary["document_count"]} Markdown documents, {summary["section_count"]} sections, {summary["chunk_count"]} deterministic chunks, and {summary["explicit_graph_edge_count"]} G0/G1 explicit graph edges. Resolved authored links: {summary["resolved_link_count"]}; unresolved authored links: {summary["unresolved_link_count"]}.

Supported edge authority is limited to G0 explicit structural edges, G1 explicit authored edges, and G2 deterministic derivatives. G3 model-inferred entities, semantic relatedness, causal relationships, coreference, and topical similarity are excluded.

## Integration Boundary

The preferred boundary is opt-in candidate expansion after lexical/vector seed retrieval. This preserves frozen initial retrieval, Agent policy, answerability, generation, citation, and grounding behavior while allowing a future disabled graph tool to expand from governed seed evidence.

## Minimal Schema

The recommended minimal schema uses DocumentNode, SectionNode, ChunkNode, TagNode, and ReferenceTargetNode. Required edge types are CONTAINS, BELONGS_TO, PARENT_OF, NEXT_SECTION, LINKS_TO, EMBEDS, and TAGGED_WITH. Alias and explicit REFERENCES parsing can be added after parser support is stricter; EntityNode, MENTIONS, and RELATED_TO remain experimental-only.

## Benchmark Gap

The frozen Core RAG Benchmark v1 coverage decision is `{summary["benchmark_graph_coverage_decision"]}`. It contains {summary["existing_graph_sensitive_sample_count"]} graph-sensitive candidates by deterministic audit, but no authoritative graph path annotations. TASK-0079 therefore creates a candidate-only, non-authoritative graph evaluation specification and does not modify frozen benchmark labels.

## Future Baseline

The proposed future experiment compares G0, the frozen Single-Agent Agentic RAG v1 baseline, against G1, G0 plus deterministic explicit-edge graph expansion. G1 must remain default-disabled, bounded, local, G0-G2-only, and fully traceable.

## Promotion Gates

Promotion requires improved graph-sensitive evidence recall, no unnecessary graph invocation on negative controls, no ordinary retrieval regression, no answer quality or safe-action regression, retained citation and grounding validity, bounded traversal, deterministic rebuilds, safe snapshot mismatch handling, and owner-approved latency and memory thresholds.

## Required Artifacts

All TASK-0079 artifacts are under `evaluation-data/results/task0079-graphrag-readiness/`, with contract `evaluation-data/contracts/task0079_graphrag_readiness_contract.json`.
"""


def run_audit() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight()
    input_hashes, inputs_valid, inputs_modified = validate_task0078_inputs()
    integration = build_agent_rag_integration_audit()
    corpus, authority = audit_corpus_graph_readiness()
    coverage = classify_benchmark_coverage()
    failure_rows = build_failure_mapping(coverage)
    schema = build_minimal_graph_schema()
    snapshot = build_graph_snapshot_contract(corpus)
    retrieval = build_graph_retrieval_contract()
    agent_contract = build_agent_graph_integration_contract()
    candidate_spec = build_candidate_graph_evaluation_spec()
    gates = build_promotion_gates()
    next_task = build_next_task_recommendation(corpus, coverage)
    summary = {
        "schema_version": "opk-rag.task0079-summary.v1",
        "task_id": "TASK-0079",
        "task_status": "complete",
        "next_task_decision": next_task["next_task_decision"],
        "task0078_inputs_valid": inputs_valid,
        "task0078_inputs_modified": inputs_modified,
        "single_agent_rag_v1_modified": False,
        "default_agent_behavior_modified": False,
        "generation_retry_policy_modified": False,
        "generation_retry_default_enabled": False,
        "generation_retry_promoted": False,
        "corpus_graph_readiness_audited": True,
        "document_count": corpus["document_count"],
        "section_count": corpus["section_count"],
        "chunk_count": corpus["chunk_count"],
        "explicit_graph_edge_count": corpus["explicit_graph_edge_count"],
        "resolved_link_count": corpus["resolved_link_count"],
        "unresolved_link_count": corpus["unresolved_link_count"],
        "benchmark_graph_coverage_audited": True,
        "existing_graph_sensitive_sample_count": coverage["graph_sensitive_candidate_count"],
        "candidate_graph_sensitive_sample_count": len(candidate_spec["candidate_samples"]),
        "benchmark_graph_coverage_decision": coverage["coverage_decision"],
        "minimal_graph_schema_defined": True,
        "graph_snapshot_contract_defined": True,
        "graph_retrieval_contract_defined": True,
        "agent_graph_integration_contract_defined": True,
        "candidate_graph_evaluation_spec_created": True,
        "graphrag_promotion_gates_defined": True,
        "minimal_graph_baseline_default_enabled": False,
        "graph_runtime_enabled": False,
        "graph_agent_action_enabled": False,
        "external_provider_calls": 0,
        "external_database_benchmark_calls": 0,
        "new_vector_retrieval_runs": 0,
        "new_formal_replicates_run": 0,
        "llm_entity_extraction_calls": 0,
    }
    contract = build_contract(summary, next_task)

    artifacts = {
        "preflight.json": preflight,
        "input_hashes.json": input_hashes,
        "agent_rag_integration_audit.json": integration,
        "corpus_graph_readiness.json": corpus,
        "graph_edge_authority_audit.json": authority,
        "benchmark_graph_coverage.json": coverage,
        "minimal_graph_schema.json": schema,
        "graph_snapshot_contract.json": snapshot,
        "graph_retrieval_contract.json": retrieval,
        "agent_graph_integration_contract.json": agent_contract,
        "candidate_graph_evaluation_spec.json": candidate_spec,
        "graphrag_promotion_gates.json": gates,
        "next_task_recommendation.json": next_task,
        "summary.json": summary,
    }
    for filename, value in artifacts.items():
        write_json(RESULTS_DIR / filename, value)
    write_jsonl(RESULTS_DIR / "failure_to_graph_mapping.jsonl", failure_rows)
    write_json(CONTRACT_PATH, contract)
    REPORT_PATH.write_text(build_report(summary, next_task), encoding="utf-8")
    result_digests = {
        "schema_version": "opk-rag.task0079-result-digests.v1",
        "task_id": "TASK-0079",
        "artifact_digests": {
            relative_path(path): file_digest(path)
            for path in sorted([*RESULTS_DIR.glob("*"), CONTRACT_PATH, REPORT_PATH])
            if path.is_file() and path.name != "result_digests.json"
        },
    }
    write_json(RESULTS_DIR / "result_digests.json", result_digests)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TASK-0079 GraphRAG readiness audit.")
    parser.parse_args(argv)
    summary = run_audit()
    print(stable_json_dumps(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
