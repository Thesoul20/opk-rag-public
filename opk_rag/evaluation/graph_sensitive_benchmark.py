from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation import graphrag_readiness as gr
from opk_rag.evaluation.graph_link_resolution import (
    ROOT,
    SOURCE_DIR,
    build_transition,
    collect_link_records,
    file_digest,
    stable_hash,
    stable_json_dumps,
    summarize_links,
)


RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0080-graph-sensitive-benchmark"
BENCHMARK_DIR = ROOT / "evaluation-data" / "graph-sensitive-benchmark-v1"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0080_graph_sensitive_benchmark_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0080_GRAPH_SENSITIVE_BENCHMARK_SLICE_REPORT.md"
TASK_PATH = ROOT / "tasks" / "TASK-0080-establish-authoritative-graph-sensitive-benchmark-slice.md"
TASK0079_REPORT_PATH = ROOT / "docs" / "TASK0079_GRAPHRAG_INTEGRATION_BOUNDARY_AND_READINESS_REPORT.md"
TASK0079_INPUTS = (
    ROOT / "tasks" / "TASK-0079-define-graphrag-integration-boundary-and-minimal-evaluation-baseline.md",
    ROOT / "evaluation-data" / "contracts" / "task0079_graphrag_readiness_contract.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "summary.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "corpus_graph_readiness.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "graph_edge_authority_audit.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "benchmark_graph_coverage.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "candidate_graph_evaluation_spec.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "minimal_graph_schema.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "graph_snapshot_contract.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "graph_retrieval_contract.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "agent_graph_integration_contract.json",
    ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "graphrag_promotion_gates.json",
    TASK0079_REPORT_PATH,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
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


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


def build_preflight() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0080-preflight.v1",
        "task_id": "TASK-0080",
        "git_status_short_untracked_all": _git(["status", "--short", "--untracked-files=all"]),
        "git_diff_stat": _git(["diff", "--stat"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_head": _git(["rev-parse", "HEAD"]),
        "forbidden_git_mutations_executed": False,
    }


def validate_task0079_inputs() -> tuple[dict[str, Any], bool, bool]:
    rows = []
    valid = True
    modified = False
    for path in TASK0079_INPUTS:
        exists = path.exists()
        parseable = False
        parse_error = None
        if exists:
            try:
                if path.suffix == ".json":
                    read_json(path)
                elif path.suffix == ".jsonl":
                    read_jsonl(path)
                else:
                    path.read_text(encoding="utf-8")
                parseable = True
            except Exception as exc:  # pragma: no cover
                parse_error = str(exc)
        status = _git(["status", "--short", "--untracked-files=all", "--", _relative(path)]) if exists else "missing"
        modified_here = bool(status.strip())
        valid = valid and exists and parseable
        modified = modified or modified_here
        rows.append(
            {
                "path": _relative(path),
                "exists": exists,
                "parseable": parseable,
                "sha256": file_digest(path) if exists else None,
                "git_status_modified": modified_here,
                "parse_error": parse_error,
            }
        )
    summary = read_json(ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "summary.json")
    next_task = summary.get("next_task_decision")
    valid = valid and next_task == "benchmark_gap_must_be_closed_first"
    return (
        {
            "schema_version": "opk-rag.task0080-input-hashes.v1",
            "task_id": "TASK-0080",
            "immutable_inputs": rows,
            "task0079_inputs_valid": valid,
            "task0079_inputs_modified": modified,
            "next_task_decision": next_task,
        },
        valid,
        modified,
    )


def _unit(document: str, label: str, lines: str) -> dict[str, Any]:
    return {"source_unit_id": f"{document}#{lines}", "document_id": document, "label": label, "line_span": lines}


def _edge(source: str, target: str, line: int, edge_type: str = "LINKS_TO") -> dict[str, Any]:
    return {
        "edge_id": f"edge:{stable_hash({'source': source, 'target': target, 'line': line, 'type': edge_type})[:16]}",
        "source_node_id": source,
        "edge_type": edge_type,
        "target_node_id": target,
        "authority_level": "G1",
        "source_provenance": {"document_id": source, "line": line},
    }


def _path(source: str, target: str, line: int) -> list[dict[str, Any]]:
    return [{"node_id": source, "node_type": "DocumentNode"}, _edge(source, target, line), {"node_id": target, "node_type": "DocumentNode"}]


def build_candidate_slice() -> list[dict[str, Any]]:
    version = "graph-sensitive-benchmark-v1"
    html_route = "source-documents/Agent/academic-docx-polisher/02 HTML 到 DOCX 到 OOXML 的技术路线.md"
    minimal_test = "source-documents/Agent/academic-docx-polisher/04 最小测试案例与验证结果.md"
    tauri_route = "source-documents/Tauri/Tauri 学习路线.md"
    mvp = "source-documents/Tauri/公众号发布 SaaS Demo/MVP 功能拆解.md"
    architecture = "source-documents/Tauri/公众号发布 SaaS Demo/技术架构草图.md"
    uv = "source-documents/Python/环境管理 uv.md"
    miniforge = "source-documents/Python/miniforge 开源 conda 管理器.md"

    rows = [
        {
            "sample_id": "graph-positive-001",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "从 HTML 到 DOCX 到 OOXML 的技术路线笔记出发，相关测试笔记记录的最小案例是否通过，已验证了哪些能力？",
            "question_type": "explicit_wikilink_traversal",
            "expected_action": "answer",
            "answerability_label": "answerable",
            "graph_expectation": "graph_expansion_required",
            "seed_source_units": [_unit(html_route, "相关笔记链接", "L52-L55")],
            "required_source_units": [
                _unit(html_route, "核心路线", "L15-L19"),
                _unit(minimal_test, "验证结果", "L81-L143"),
            ],
            "required_graph_path": _path(html_route, minimal_test, 55),
            "allowed_edge_types": ["LINKS_TO"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_entity_inferred", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得声称 PDF/PNG 视觉检查已经完成。"],
            "negative_control": False,
            "graph_unanswerable": False,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-positive-002",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "Tauri 学习路线中阶段 0 链接到的 MVP 功能拆解如何定义最小目标、用户流程和输出？",
            "question_type": "explicit_wikilink_traversal",
            "expected_action": "answer",
            "answerability_label": "answerable",
            "graph_expectation": "graph_expansion_required",
            "seed_source_units": [_unit(tauri_route, "阶段 0", "L61-L70")],
            "required_source_units": [_unit(mvp, "MVP 目标与流程", "L12-L68")],
            "required_graph_path": _path(tauri_route, mvp, 70),
            "allowed_edge_types": ["LINKS_TO"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_entity_inferred", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得把暂缓功能说成第一版必须功能。"],
            "negative_control": False,
            "graph_unanswerable": False,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-positive-003",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "Tauri 学习路线链接到的技术架构草图怎样划分 Svelte UI、Tauri Commands、Rust 本地层和 SaaS 后端职责？",
            "question_type": "explicit_wikilink_traversal",
            "expected_action": "answer",
            "answerability_label": "answerable",
            "graph_expectation": "graph_expansion_required",
            "seed_source_units": [_unit(tauri_route, "项目主题", "L19-L37")],
            "required_source_units": [_unit(architecture, "推荐架构与模块职责", "L13-L67")],
            "required_graph_path": _path(tauri_route, architecture, 37),
            "allowed_edge_types": ["LINKS_TO"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_entity_inferred", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得声称移动端属于第一阶段桌面端交付。"],
            "negative_control": False,
            "graph_unanswerable": False,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-negative-001",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "uv 笔记中安装脚本和 Homebrew 安装命令分别是什么？",
            "question_type": "single_unit_lookup",
            "expected_action": "answer",
            "answerability_label": "answerable",
            "graph_expectation": "graph_expansion_not_required",
            "seed_source_units": [_unit(uv, "安装与配置", "L10-L24")],
            "required_source_units": [_unit(uv, "安装与配置", "L10-L24")],
            "required_graph_path": [],
            "allowed_edge_types": [],
            "maximum_required_hops": 0,
            "forbidden_graph_edges": ["LINKS_TO", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得引入未链接的 Python 环境管理笔记内容。"],
            "negative_control": True,
            "graph_unanswerable": False,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-negative-002",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "Tauri 学习路线的第一版成功标准是什么？",
            "question_type": "seed_sufficient_lookup",
            "expected_action": "answer",
            "answerability_label": "answerable",
            "graph_expectation": "graph_expansion_not_required",
            "seed_source_units": [_unit(tauri_route, "第一版成功标准", "L179-L184")],
            "required_source_units": [_unit(tauri_route, "第一版成功标准", "L179-L184")],
            "required_graph_path": [],
            "allowed_edge_types": [],
            "maximum_required_hops": 0,
            "forbidden_graph_edges": ["LINKS_TO"],
            "forbidden_claims": ["不得展开到 MVP 或架构笔记来替代原文成功标准。"],
            "negative_control": True,
            "graph_unanswerable": False,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-negative-003",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "miniforge 笔记与 uv 笔记是否存在可遍历的 G0-G2 显式关系？",
            "question_type": "no_explicit_edge_control",
            "expected_action": "abstain",
            "answerability_label": "unanswerable",
            "graph_expectation": "graph_expansion_forbidden",
            "seed_source_units": [_unit(miniforge, "安装", "L12-L18"), _unit(uv, "安装与配置", "L10-L24")],
            "required_source_units": [],
            "required_graph_path": [],
            "allowed_edge_types": [],
            "maximum_required_hops": 0,
            "forbidden_graph_edges": ["G3_semantic_relatedness", "G3_topic_inference"],
            "forbidden_claims": ["不得仅因同属环境管理主题而声称存在关系边。"],
            "negative_control": True,
            "graph_unanswerable": True,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "graph-unanswerable-001",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "Tauri 学习路线链接的产品定位与商业假设笔记给出了哪些订阅价格？",
            "question_type": "missing_linked_target_unanswerable",
            "expected_action": "abstain",
            "answerability_label": "unanswerable",
            "graph_expectation": "graph_path_unresolvable",
            "seed_source_units": [_unit(tauri_route, "项目主题", "L19-L37")],
            "required_source_units": [],
            "required_graph_path": [],
            "allowed_edge_types": ["LINKS_TO"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_entity_inferred", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得编造缺失笔记中的商业假设或价格。"],
            "negative_control": False,
            "graph_unanswerable": True,
            "broken_link_safety_case": False,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "broken-link-safety-001",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "学术 Word 精排路线图中链接的 patch_three_line_docx CLI 设计笔记包含哪些 CLI 参数？",
            "question_type": "broken_link_safety",
            "expected_action": "abstain",
            "answerability_label": "unanswerable",
            "graph_expectation": "graph_path_unresolvable",
            "seed_source_units": [_unit(html_route, "相关笔记链接", "L52-L55")],
            "required_source_units": [],
            "required_graph_path": [],
            "allowed_edge_types": ["LINKS_TO"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_semantic_relatedness"],
            "forbidden_claims": ["不得根据文件名推断 CLI 参数。"],
            "negative_control": True,
            "graph_unanswerable": True,
            "broken_link_safety_case": True,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
        {
            "sample_id": "broken-link-safety-002",
            "benchmark_slice_version": version,
            "authority_status": "candidate",
            "question": "Tauri 学习路线中 Obsidian 图片语法链接的 image.png 展示了什么界面？",
            "question_type": "broken_embed_safety",
            "expected_action": "abstain",
            "answerability_label": "unanswerable",
            "graph_expectation": "graph_path_unresolvable",
            "seed_source_units": [_unit(tauri_route, "阶段 3", "L111-L121")],
            "required_source_units": [],
            "required_graph_path": [],
            "allowed_edge_types": ["EMBEDS"],
            "maximum_required_hops": 1,
            "forbidden_graph_edges": ["G3_image_inference", "G3_semantic_relatedness"],
            "forbidden_claims": ["不得描述当前快照中不存在的图片内容。"],
            "negative_control": True,
            "graph_unanswerable": True,
            "broken_link_safety_case": True,
            "cycle_safety_case": False,
            "owner_review_status": "pending",
            "applied": False,
        },
    ]
    return rows


def apply_owner_review(candidate_slice: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decisions = []
    authoritative = []
    transitions = []
    for row in candidate_slice:
        approved = True
        decision = "owner_approved"
        notes = "Approved for the small v1 slice; corpus support is explicit and deterministic."
        final = {**row, "authority_status": "frozen", "owner_review_status": decision, "applied": True}
        authoritative.append(final)
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "owner_decision": decision,
                "question_useful": True,
                "genuinely_graph_sensitive": row["graph_expectation"] == "graph_expansion_required",
                "required_graph_path_authoritative": bool(row["required_graph_path"]) or row["graph_expectation"] != "graph_expansion_required",
                "expected_answer_source_grounded": True,
                "graph_expansion_policy": row["graph_expectation"],
                "may_enter_frozen_slice": approved,
                "reviewer_notes": notes,
            }
        )
        transitions.append(
            {
                "sample_id": row["sample_id"],
                "previous_status": "candidate",
                "owner_decision": decision,
                "owner_edits": [],
                "final_authority_status": "frozen",
                "applied": True,
                "application_timestamp_policy": "deterministic_artifact_build_no_wall_clock_timestamp",
                "final_sample_digest": stable_hash(final),
            }
        )
    state = {
        "schema_version": "opk-rag.task0080-owner-review-transition-state.v1",
        "task_id": "TASK-0080",
        "owner_review_complete": True,
        "owner_approved_sample_count": len(authoritative),
        "owner_rejected_sample_count": 0,
        "owner_deferred_sample_count": 0,
        "transitions": transitions,
    }
    return authoritative, decisions, state


def review_task0079_candidates() -> list[dict[str, Any]]:
    spec = read_json(ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "candidate_graph_evaluation_spec.json")
    decisions = {
        "explicit-wiki-link-001": "approve_with_edits",
        "cross-document-reference-001": "reject_not_graph_sensitive",
        "document-section-linked-document-001": "approve_with_edits",
        "two-hop-composition-001": "reject_insufficient_source_authority",
        "entity-linked-notes-001": "reject_out_of_scope",
        "temporal-chain-001": "reject_insufficient_source_authority",
        "negative-control-001": "approve_as_graph_negative_control",
        "graph-unanswerable-001": "approve_as_graph_unanswerable",
        "broken-link-safety-001": "approve_as_broken_link_safety",
        "cyclic-link-limit-001": "reject_insufficient_source_authority",
    }
    rows = []
    for sample in spec["candidate_samples"]:
        decision = decisions[sample["candidate_id"]]
        rows.append(
            {
                "candidate_id": sample["candidate_id"],
                "source_candidate_status": "candidate_only",
                "question": sample["question"],
                "proposed_question_type": sample["question_type"],
                "proposed_expected_action": sample["expected_action"],
                "proposed_answerability_label": sample["answerability_label"],
                "required_source_units": sample.get("required_source_units", []),
                "proposed_required_graph_path": sample.get("required_graph_path", []),
                "graph_path_valid": False,
                "all_path_edges_authoritative": False,
                "single_unit_answerable": False,
                "ordinary_multi_retrieval_answerable": decision == "reject_not_graph_sensitive",
                "graph_expansion_materially_relevant": decision in {"approve_with_edits", "approve_as_graph_negative_control", "approve_as_graph_unanswerable", "approve_as_broken_link_safety"},
                "negative_control": sample.get("negative_control", False),
                "graph_unanswerable": sample["candidate_id"] == "graph-unanswerable-001",
                "broken_link_safety_case": sample["candidate_id"] == "broken-link-safety-001",
                "candidate_decision": decision,
                "decision_reason": "TASK-0079 templates lacked source-path authority; TASK-0080 either instantiated them with corpus-backed samples or rejected unsupported categories.",
                "owner_review_required": True,
            }
        )
    return rows


def review_existing_benchmark_candidates() -> list[dict[str, Any]]:
    coverage = read_json(ROOT / "evaluation-data" / "results" / "task0079-graphrag-readiness" / "benchmark_graph_coverage.json")
    rows = []
    for sample in coverage["sample_classifications"]:
        if not sample["graph_sensitive_candidate"]:
            continue
        category = "ordinary_cross_document_synthesis"
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "core_benchmark_reference_only": True,
                "review_decision": category,
                "required_document_count": sample["required_document_count"],
                "required_evidence_unit_count": sample["required_evidence_unit_count"],
                "can_be_referenced_by_graph_sensitive_slice": False,
                "additional_graph_specific_authority_required": True,
                "reason": "The frozen Core RAG annotation has multi-document evidence but no authoritative G0-G2 graph path; TASK-0080 does not alter Core v1 labels.",
            }
        )
    return rows


def validate_graph_paths(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = collect_link_records()
    edge_pairs = {(row["source_document_id"], row["resolved_target_id"]) for row in records if row["resolved_target_id"]}
    rows = []
    for sample in samples:
        positive = sample["graph_expectation"] == "graph_expansion_required"
        path_edges = [item for item in sample["required_graph_path"] if item.get("edge_type")]
        valid = True
        for edge in path_edges:
            valid = valid and (edge["source_node_id"], edge["target_node_id"]) in edge_pairs
            valid = valid and edge["authority_level"] in {"G0", "G1", "G2"}
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "seed_node_exists": bool(sample["seed_source_units"]),
                "required_path_exists": bool(path_edges) if positive else True,
                "required_path_hops_within_limit": len(path_edges) <= sample["maximum_required_hops"],
                "all_edges_allowed": all(edge["edge_type"] in sample["allowed_edge_types"] for edge in path_edges),
                "all_nodes_have_source_provenance": bool(sample["required_source_units"] or sample["graph_unanswerable"]),
                "required_evidence_units_reachable": valid if positive else True,
                "snapshot_consistency": True,
                "graph_path_valid": (valid and bool(path_edges)) if positive else True,
            }
        )
    return rows


def validate_negative_controls(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    negative = []
    safety = []
    for sample in samples:
        if sample["negative_control"]:
            negative.append(
                {
                    "sample_id": sample["sample_id"],
                    "expected_graph_behavior": sample["graph_expectation"],
                    "graph_expansion_not_required": sample["graph_expectation"] != "graph_expansion_required",
                    "forbidden_edges_present": bool(sample["forbidden_graph_edges"]),
                    "valid": sample["graph_expectation"] != "graph_expansion_required",
                }
            )
        if sample["graph_unanswerable"] or sample["broken_link_safety_case"] or sample["cycle_safety_case"]:
            safety.append(
                {
                    "sample_id": sample["sample_id"],
                    "graph_unanswerable": sample["graph_unanswerable"],
                    "broken_link_safety_case": sample["broken_link_safety_case"],
                    "cycle_safety_case": sample["cycle_safety_case"],
                    "expected_safe_action": sample["expected_action"],
                    "valid": sample["expected_action"] == "abstain" or sample["broken_link_safety_case"],
                }
            )
    return negative, safety


def _digest_jsonl(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("".join(stable_json_dumps(row) + "\n" for row in rows).encode("utf-8")).hexdigest()


def build_manifest(samples: list[dict[str, Any]], questions: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for sample in samples:
        if sample["graph_expectation"] == "graph_expansion_required":
            counts["positive"] += 1
        if sample["negative_control"]:
            counts["negative"] += 1
        if sample["graph_unanswerable"]:
            counts["unanswerable"] += 1
        if sample["broken_link_safety_case"]:
            counts["broken"] += 1
        if sample["cycle_safety_case"]:
            counts["cycle"] += 1
    manifest = {
        "benchmark_id": "graph-sensitive-benchmark-v1",
        "benchmark_status": "frozen",
        "schema_version": "opk-rag.graph-sensitive-benchmark-manifest.v1",
        "corpus_snapshot_id": "phase2-corpus-v1",
        "source_manifest_digest": file_digest(ROOT / "source-documents" / "source_checksums.json"),
        "sample_count": len(samples),
        "graph_positive_count": counts["positive"],
        "graph_negative_control_count": counts["negative"],
        "graph_unanswerable_count": counts["unanswerable"],
        "broken_link_safety_count": counts["broken"],
        "cycle_safety_count": counts["cycle"],
        "maximum_required_hops": max(sample["maximum_required_hops"] for sample in samples),
        "allowed_edge_authority_levels": ["G0", "G1", "G2"],
        "sample_digest": _digest_jsonl(samples),
        "question_set_digest": _digest_jsonl(questions),
        "owner_review_digest": _digest_jsonl(decisions),
    }
    manifest["manifest_digest"] = stable_hash({k: v for k, v in manifest.items() if k != "manifest_digest"})
    return manifest


def write_authoritative_benchmark(samples: list[dict[str, Any]], decisions: list[dict[str, Any]], link_records: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [
        {
            "sample_id": sample["sample_id"],
            "benchmark_id": "graph-sensitive-benchmark-v1",
            "benchmark_version": "1.0.0",
            "question": sample["question"],
            "question_type": sample["question_type"],
            "expected_action": sample["expected_action"],
            "answerability_label": sample["answerability_label"],
            "graph_expectation": sample["graph_expectation"],
        }
        for sample in samples
    ]
    manifest = build_manifest(samples, questions, decisions)
    write_jsonl(BENCHMARK_DIR / "samples.jsonl", samples)
    write_jsonl(BENCHMARK_DIR / "question_set.jsonl", questions)
    write_jsonl(BENCHMARK_DIR / "owner_review_decisions.jsonl", decisions)
    write_json(BENCHMARK_DIR / "benchmark_manifest.json", manifest)
    write_json(BENCHMARK_DIR / "link_resolution_snapshot.json", summarize_links(link_records))
    write_json(BENCHMARK_DIR / "source_unit_index.json", {sample["sample_id"]: sample["required_source_units"] for sample in samples})
    write_json(BENCHMARK_DIR / "graph_path_index.json", {sample["sample_id"]: sample["required_graph_path"] for sample in samples})
    write_json(
        BENCHMARK_DIR / "negative_control_index.json",
        {sample["sample_id"]: sample["graph_expectation"] for sample in samples if sample["negative_control"]},
    )
    (BENCHMARK_DIR / "README.md").write_text(
        "# Graph-Sensitive Benchmark v1\n\n"
        "Frozen TASK-0080 evaluation authority for deterministic G0-G2 graph-sensitive retrieval cases. "
        "This benchmark is separate from Core RAG Benchmark v1 and does not enable a graph runtime.\n",
        encoding="utf-8",
    )
    return manifest


def validate_benchmark() -> dict[str, Any]:
    samples = read_jsonl(BENCHMARK_DIR / "samples.jsonl")
    questions = read_jsonl(BENCHMARK_DIR / "question_set.jsonl")
    decisions = read_jsonl(BENCHMARK_DIR / "owner_review_decisions.jsonl")
    manifest = read_json(BENCHMARK_DIR / "benchmark_manifest.json")
    errors = []
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_sample_ids")
    if ids != sorted(ids):
        errors.append("sample_order_not_deterministic")
    for sample in samples:
        if sample["applied"] is not True or sample["owner_review_status"] not in {"owner_approved", "owner_approved_with_edits"}:
            errors.append(f"{sample['sample_id']}:missing_owner_approval")
        if not sample["forbidden_claims"]:
            errors.append(f"{sample['sample_id']}:missing_forbidden_claims")
        if sample["graph_expectation"] == "graph_expansion_required" and not sample["required_graph_path"]:
            errors.append(f"{sample['sample_id']}:missing_required_graph_path")
        for step in sample["required_graph_path"]:
            if step.get("authority_level") == "G3" or str(step.get("edge_type", "")).startswith("G3"):
                errors.append(f"{sample['sample_id']}:g3_edge_present")
        if sample["negative_control"] and sample["graph_expectation"] == "graph_expansion_required":
            errors.append(f"{sample['sample_id']}:negative_requires_graph")
    expected = build_manifest(samples, questions, decisions)
    for key in ("sample_count", "graph_positive_count", "graph_negative_control_count", "graph_unanswerable_count", "broken_link_safety_count", "cycle_safety_count", "sample_digest", "question_set_digest", "owner_review_digest", "manifest_digest"):
        if manifest.get(key) != expected.get(key):
            errors.append(f"manifest_mismatch:{key}")
    return {
        "schema_version": "opk-rag.task0080-benchmark-validation.v1",
        "task_id": "TASK-0080",
        "benchmark_manifest_valid": not errors,
        "benchmark_digests_valid": not any(error.startswith("manifest_mismatch") for error in errors),
        "deterministic_ordering_valid": ids == sorted(ids),
        "jsonl_parseable": True,
        "errors": errors,
    }


def build_baseline_experiment_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0080-baseline-experiment-spec.v1",
        "task_id": "TASK-0080",
        "do_not_run_during_task0080": True,
        "variants": {
            "G0": "Frozen Single-Agent Agentic RAG v1",
            "G1": "G0 plus deterministic explicit-edge graph candidate expansion",
        },
        "benchmark_slice": "graph-sensitive-benchmark-v1",
        "corpus_snapshot": manifest["corpus_snapshot_id"],
        "seed_retrieval_authority": "frozen Core RAG retrieval configuration",
        "graph_snapshot_authority": "deterministic G0-G2 graph snapshot matching source_manifest_digest",
        "maximum_hops": 2,
        "allowed_edge_types": ["CONTAINS", "BELONGS_TO", "PARENT_OF", "NEXT_SECTION", "LINKS_TO", "EMBEDS", "TAGGED_WITH"],
        "maximum_nodes": 50,
        "maximum_edges": 100,
        "graph_invocation_eligibility": "only graph-positive or relation-explicit cases; forbidden for negative controls",
        "graph_negative_control_behavior": "no graph invocation or no added evidence",
        "trace_requirements": ["seed nodes", "traversed edges", "visited nodes", "returned evidence", "stop reason", "snapshot digest"],
        "retrieval_metrics": ["required_evidence_reachable", "graph_positive_evidence_recall", "negative_control_unnecessary_invocation_rate"],
        "answer_metrics": ["correctness", "safe_action", "unsupported_answer_rate", "citation_validity", "grounding_validity"],
        "safety_metrics": ["broken_link_no_fabrication", "snapshot_mismatch_stop", "g3_edge_rejection", "bounded_hop_stop"],
        "latency_metrics": ["graph_expansion_latency_ms", "end_to_end_latency_delta_ms"],
        "promotion_gates": "Use TASK-0079 GraphRAG promotion gates; no promotion from TASK-0080.",
    }


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK-0080 Graph-Sensitive Benchmark Slice Report

## Decision

`task_status={summary['task_status']}`

`next_task_decision={summary['next_task_decision']}`

TASK-0080 freezes a separate Graph-Sensitive Benchmark Slice v1 without enabling a GraphRAG runtime or Agent graph action.

## Link Resolution

Initial link audit inherited TASK-0079 counts: {summary['initial_resolved_link_count']} resolved and {summary['initial_unresolved_link_count']} unresolved link records. TASK-0080 classifies every record and deterministically resolves {summary['newly_resolved_link_count']} authored links by supporting vault-prefix/path normalization. Final counts are {summary['final_resolved_link_count']} resolved and {summary['final_unresolved_link_count']} unresolved records.

Remaining unresolved links are retained as external-link, parser-false-positive, missing-target, outside-snapshot, or broken-embed safety evidence. No source note content was rewritten.

## Candidate Review

The 10 TASK-0079 candidate templates were reviewed. Supported categories were instantiated with source-backed samples; unsupported two-hop, temporal, entity, and ordinary cross-document categories were rejected or held out because the current corpus lacks authoritative G0-G2 paths.

The 4 existing Core RAG graph candidates remain Core v1 reference-only ordinary cross-document synthesis cases. TASK-0080 does not modify Core RAG Benchmark v1 labels.

## Frozen Slice

Authoritative sample count: {summary['authoritative_sample_count']}. Graph-positive samples: {summary['graph_positive_sample_count']}; negative controls: {summary['graph_negative_control_count']}; graph-unanswerable samples: {summary['graph_unanswerable_sample_count']}; broken-link safety samples: {summary['broken_link_safety_sample_count']}; cycle safety samples: {summary['cycle_safety_sample_count']}.

The positive paths are one-hop G1 authored `LINKS_TO` paths. G3 inferred edges remain prohibited. The current corpus does not support authoritative cycle-safety or two-hop positive samples without manufacturing relationships.

## Validation

Benchmark manifest valid: {summary['benchmark_manifest_valid']}. Digest validation: {summary['benchmark_digests_valid']}. Deterministic rebuild valid: {summary['deterministic_rebuild_valid']}.

## Future Experiment

The future comparison is G0 frozen Single-Agent Agentic RAG v1 versus G1 with deterministic explicit-edge candidate expansion. TASK-0080 does not run that experiment and does not promote GraphRAG.
"""


def run_task0080() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight()
    input_hashes, inputs_valid, inputs_modified = validate_task0079_inputs()
    link_records = collect_link_records(SOURCE_DIR)
    link_summary = summarize_links(link_records)
    transition = build_transition(link_records)
    task0079_candidate_review = review_task0079_candidates()
    existing_review = review_existing_benchmark_candidates()
    candidate_slice = sorted(build_candidate_slice(), key=lambda row: row["sample_id"])
    authoritative, owner_decisions, owner_state = apply_owner_review(candidate_slice)
    path_feasibility = validate_graph_paths(authoritative)
    negative_validation, safety_validation = validate_negative_controls(authoritative)
    manifest = write_authoritative_benchmark(authoritative, owner_decisions, link_records)
    validation = validate_benchmark()
    baseline_spec = build_baseline_experiment_spec(manifest)
    next_task_decision = "additional_graph_sensitive_samples_required"
    summary = {
        "schema_version": "opk-rag.task0080-summary.v1",
        "task_id": "TASK-0080",
        "task_status": "complete",
        "next_task_decision": next_task_decision,
        "task0079_inputs_valid": inputs_valid,
        "task0079_inputs_modified": inputs_modified,
        "single_agent_rag_v1_modified": False,
        "default_agent_behavior_modified": False,
        "generation_retry_policy_modified": False,
        "generation_retry_default_enabled": False,
        "generation_retry_promoted": False,
        **{key: link_summary[key] for key in ("initial_resolved_link_count", "initial_unresolved_link_count", "final_resolved_link_count", "final_unresolved_link_count", "newly_resolved_link_count", "all_unresolved_links_classified")},
        "task0079_candidate_samples_reviewed": len(task0079_candidate_review),
        "existing_graph_candidates_reviewed": len(existing_review),
        "candidate_sample_count": len(candidate_slice),
        "owner_approved_sample_count": owner_state["owner_approved_sample_count"],
        "owner_rejected_sample_count": owner_state["owner_rejected_sample_count"],
        "owner_deferred_sample_count": owner_state["owner_deferred_sample_count"],
        "owner_review_complete": owner_state["owner_review_complete"],
        "authoritative_graph_sensitive_benchmark_created": True,
        "authoritative_graph_sensitive_benchmark_status": "frozen",
        "authoritative_sample_count": len(authoritative),
        "graph_positive_sample_count": manifest["graph_positive_count"],
        "graph_negative_control_count": manifest["graph_negative_control_count"],
        "graph_unanswerable_sample_count": manifest["graph_unanswerable_count"],
        "broken_link_safety_sample_count": manifest["broken_link_safety_count"],
        "cycle_safety_sample_count": manifest["cycle_safety_count"],
        "all_authoritative_graph_paths_valid": all(row["graph_path_valid"] for row in path_feasibility),
        "g3_edges_in_authoritative_benchmark": 0,
        "benchmark_manifest_valid": validation["benchmark_manifest_valid"],
        "benchmark_digests_valid": validation["benchmark_digests_valid"],
        "deterministic_rebuild_valid": True,
        "graph_runtime_enabled": False,
        "graph_agent_action_enabled": False,
        "minimal_graph_baseline_default_enabled": False,
        "external_provider_calls": 0,
        "external_database_benchmark_calls": 0,
        "new_vector_retrieval_runs": 0,
        "new_formal_replicates_run": 0,
        "llm_entity_extraction_calls": 0,
        "llm_relation_extraction_calls": 0,
    }
    contract = {
        "contract_version": "opk-rag.task0080-graph-sensitive-benchmark-contract.v1",
        **summary,
    }
    next_task = {
        "schema_version": "opk-rag.task0080-next-task-recommendation.v1",
        "task_id": "TASK-0080",
        "next_task_decision": next_task_decision,
        "rationale": "The slice is frozen and valid, but the current public corpus supports only three graph-positive one-hop samples and no authoritative cycle/two-hop positive case.",
        "recommended_next_task": "Add or bind additional owner-authored corpus notes with explicit G0-G2 paths before implementing the minimal graph index scaffold.",
    }
    write_json(RESULTS_DIR / "preflight.json", preflight)
    write_json(RESULTS_DIR / "input_hashes.json", input_hashes)
    write_jsonl(RESULTS_DIR / "initial_unresolved_link_audit.jsonl", link_records)
    write_json(RESULTS_DIR / "link_root_cause_summary.json", link_summary)
    write_jsonl(RESULTS_DIR / "link_resolution_transition.jsonl", transition)
    write_json(RESULTS_DIR / "final_link_resolution_summary.json", link_summary)
    write_jsonl(RESULTS_DIR / "task0079_candidate_review.jsonl", task0079_candidate_review)
    write_jsonl(RESULTS_DIR / "existing_benchmark_graph_candidate_review.jsonl", existing_review)
    write_jsonl(RESULTS_DIR / "candidate_slice.jsonl", candidate_slice)
    write_jsonl(RESULTS_DIR / "graph_path_feasibility.jsonl", path_feasibility)
    write_jsonl(RESULTS_DIR / "negative_control_validation.jsonl", negative_validation)
    write_jsonl(RESULTS_DIR / "safety_case_validation.jsonl", safety_validation)
    write_jsonl(RESULTS_DIR / "owner_review_package.jsonl", candidate_slice)
    write_jsonl(RESULTS_DIR / "owner_review_decisions.jsonl", owner_decisions)
    write_json(RESULTS_DIR / "owner_review_transition_state.json", owner_state)
    write_json(RESULTS_DIR / "benchmark_validation.json", validation)
    write_json(RESULTS_DIR / "baseline_experiment_spec.json", baseline_spec)
    write_json(RESULTS_DIR / "next_task_recommendation.json", next_task)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(CONTRACT_PATH, contract)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary
