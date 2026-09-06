from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0218"
SCHEMA_VERSION = "opk-rag.task0218.showcase-scenario-and-demo-query-freeze.v1"
MANIFEST_SCHEMA_VERSION = "opk-rag.showcase-manifest.v1"
RESULT_DIR = ROOT / "evaluation-data/results/task0218-showcase-scenario-and-demo-query-freeze"
MANIFEST_PATH = ROOT / "evaluation-data/showcase/showcase_manifest_v1.json"
CONTRACT_PATH = ROOT / "evaluation-data/contracts/task0218_showcase_scenario_and_demo_query_freeze_contract.json"
GUIDE_PATH = ROOT / "docs/SHOWCASE_SCENARIOS.md"
REPORT_PATH = ROOT / "docs/TASK0218_SHOWCASE_SCENARIO_AND_DEMO_QUERY_FREEZE_REPORT.md"

TASK0214_SUMMARY = ROOT / "evaluation-data/results/task0214-project-showcase-delivery-stage-entry-and-authority-baseline/summary.json"
TASK0217_DIR = ROOT / "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration"

SEMANTIC_QUERY = "uv 如何创建和管理 Python 项目环境？"
STRUCTURE_QUERY = "如何验证 HTML 到 DOCX 到 OOXML 的技术路线，并找到相关测试笔记？"
GRAPH_QUERY = "Tauri 学习路线与公众号发布 SaaS Demo 的 MVP 功能拆解和技术架构草图有什么关系？"
FAIL_CLOSED_QUERY = "当前生产 API 的 SLA 是多少？"
HISTORICAL_GRAPH_QUERY = "academic-docx-polisher 的技术路线、最小测试和 Skill 化路线之间是什么关系？"

MUTATION_GUARDS = {
    "showcase_runtime_mutation_required": False,
    "retrieval_policy_mutation_required": False,
    "reranker_mutation_required": False,
    "embedding_mutation_required": False,
    "graph_runtime_mutation_required": False,
    "agent_policy_mutation_required": False,
    "evaluation_baseline_mutation_required": False,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def canonical_demo_query_payload(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"scenario_id": str(item["scenario_id"]), "query": str(item["query"])}
        for item in sorted(scenarios, key=lambda row: str(row["scenario_id"]))
        if item.get("approved") is True
    ]


def demo_query_digest(scenarios: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(canonical_demo_query_payload(scenarios), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": "S01",
            "name": "baseline_semantic_knowledge_retrieval",
            "query": SEMANTIC_QUERY,
            "command": "search",
            "capabilities": ["semantic_retrieval", "reranking", "evidence_provenance"],
            "expected_behavior": ["vector_retrieval", "reranking", "evidence_selection", "provenance"],
            "runtime_mutation_required": False,
            "runtime_verified": True,
            "stability_status": "verified_current_production_runtime",
            "approved": True,
            "authoritative_evidence": [
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/ordinary_query_control.json",
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/graph_activation_replay.json",
            ],
        },
        {
            "scenario_id": "S02",
            "name": "guarded_structure_aware_bounded_recovery",
            "query": STRUCTURE_QUERY,
            "command": "search",
            "capabilities": ["structure_aware_retrieval", "guarded_agent", "bounded_recovery"],
            "expected_behavior": ["initial_retrieval", "guard_evaluation", "structure_lane", "bounded_runtime_expansion", "reranking", "evidence_selection"],
            "runtime_mutation_required": False,
            "runtime_verified": True,
            "stability_status": "verified_current_production_runtime",
            "approved": True,
            "authoritative_evidence": [
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/guarded_structure_aware_replay.json",
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/guard_recovery_replay.json",
            ],
        },
        {
            "scenario_id": "S03",
            "name": "graph_sensitive_cross_document_retrieval",
            "query": GRAPH_QUERY,
            "command": "search",
            "capabilities": ["graph_sensitive_retrieval", "guarded_agent", "bounded_recovery", "cross_document_provenance"],
            "expected_behavior": ["initial_retrieval", "graph_activation", "one_hop_graph_expansion", "graph_added_candidate", "reranking", "evidence_selection"],
            "runtime_mutation_required": False,
            "runtime_verified": True,
            "stability_status": "verified_current_production_runtime",
            "approved": True,
            "authoritative_evidence": [
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/graph_expansion_replay.json",
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/graph_activation_replay.json",
            ],
        },
        {
            "scenario_id": "S04",
            "name": "evidence_safety_fail_closed",
            "query": FAIL_CLOSED_QUERY,
            "command": "ask",
            "capabilities": ["evidence_safety", "guarded_agent", "fail_closed"],
            "expected_behavior": ["retrieval", "evidence_validation", "answerability_boundary", "refusal"],
            "runtime_mutation_required": False,
            "runtime_verified": True,
            "stability_status": "verified_current_production_runtime",
            "approved": True,
            "authoritative_evidence": [
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/fail_closed_control.json",
                "evaluation-data/results/task0217-public-production-search-guarded-structure-aware-graph-v1-runtime-integration/guard_recovery_replay.json",
            ],
        },
    ]


def validate_runtime_evidence() -> dict[str, Any]:
    ordinary = read_json(TASK0217_DIR / "ordinary_query_control.json")
    structure = read_json(TASK0217_DIR / "guarded_structure_aware_replay.json")
    graph = read_json(TASK0217_DIR / "graph_expansion_replay.json")
    fail_closed = read_json(TASK0217_DIR / "fail_closed_control.json")
    historical_negative = read_json(TASK0217_DIR / "graph_negative_control.json")

    validations = {
        "semantic_query_matches": ordinary.get("query") == SEMANTIC_QUERY,
        "semantic_runtime_passed": ordinary.get("execution_status") == "passed" and int(ordinary.get("result_count") or 0) > 0,
        "structure_query_matches": structure.get("query") == STRUCTURE_QUERY,
        "structure_runtime_passed": structure.get("execution_status") == "passed",
        "structure_lane_invoked": (structure.get("guard_trace") or {}).get("structure_lane_invoked") is True,
        "structure_bounded_recovery": (structure.get("guard_trace") or {}).get("recovery_activated") is True and int((structure.get("guard_trace") or {}).get("recovery_attempt_count") or 0) <= 1,
        "graph_query_matches": graph.get("query") == GRAPH_QUERY,
        "graph_runtime_passed": graph.get("execution_status") == "passed",
        "graph_activated": (graph.get("graph_trace") or {}).get("graph_activated") is True,
        "graph_one_hop": (graph.get("graph_trace") or {}).get("hop_depth") == 1,
        "graph_expanded_candidate_present": int((graph.get("graph_trace") or {}).get("expanded_candidate_count") or 0) > 0,
        "graph_provenance_present": bool((graph.get("graph_trace") or {}).get("candidate_provenance")),
        "fail_closed_query_matches": fail_closed.get("query") == FAIL_CLOSED_QUERY,
        "fail_closed_runtime_passed": fail_closed.get("execution_status") == "passed",
        "fail_closed_status_refused": fail_closed.get("status") == "refused",
        "fail_closed_guard_decision": (fail_closed.get("guard_trace") or {}).get("final_decision") == "fail_closed",
        "historical_graph_query_matches": historical_negative.get("query") == HISTORICAL_GRAPH_QUERY,
        "historical_graph_query_inactive": (historical_negative.get("graph_trace") or {}).get("graph_activated") is False,
        "runtime_gold_metadata_usage": False,
    }
    validations["all_selected_runtime_evidence_valid"] = all(
        value is True for key, value in validations.items() if key != "runtime_gold_metadata_usage"
    ) and validations["runtime_gold_metadata_usage"] is False
    return validations


def build_manifest() -> dict[str, Any]:
    authority = read_json(TASK0214_SUMMARY)
    scenarios = build_scenarios()
    evidence_validation = validate_runtime_evidence()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_version": "Showcase Demo Query Set V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repository_head": git_head(),
        "active_project_stage": "project_showcase_delivery",
        "production_vector_backend": authority.get("production_vector_backend"),
        "production_embedding_model": authority.get("production_embedding_model"),
        "production_reranker_model": authority.get("production_reranker_model"),
        "default_initial_retrieval_policy": authority.get("default_initial_retrieval_policy"),
        "graph_runtime_hop_depth": authority.get("graph_runtime_hop_depth"),
        "scenario_count": len(scenarios),
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "scenarios": scenarios,
        "showcase_demo_query_set_v1_digest": demo_query_digest(scenarios),
        "capability_coverage": {
            "semantic_retrieval_showcase_covered": True,
            "structure_aware_showcase_covered": True,
            "graph_sensitive_showcase_covered": True,
            "guarded_agent_showcase_covered": True,
            "evidence_safety_showcase_covered": True,
            "hybrid_or_lexical_showcase_covered": False,
            "hybrid_or_lexical_showcase_exclusion_explained": True,
        },
        "optional_exclusions": [
            {
                "capability": "hybrid_or_lexical_showcase",
                "reason": "TASK-0215 through TASK-0217 current showcase execution artifacts do not provide a clean production hybrid/BM25-active scenario; no runtime change is justified solely to add one.",
            }
        ],
        "superseded_demo_query_notes": [
            {
                "query": HISTORICAL_GRAPH_QUERY,
                "previous_role": "historical TASK-0214/TASK-0215 graph-sensitive demo query",
                "current_status": "not_authoritative_for_graph_active_showcase",
                "reason": "TASK-0217 proves it remains Graph-inactive under frozen top-3 seed semantics because the selected seeds have no authoritative expandable edge.",
            }
        ],
        "runtime_evidence_validation": evidence_validation,
        "all_approved_scenarios_runtime_verified": evidence_validation["all_selected_runtime_evidence_valid"],
        "all_showcase_claims_evidence_traceable": all(
            all((ROOT / path).exists() for path in item["authoritative_evidence"]) for item in scenarios
        ),
        **MUTATION_GUARDS,
        "future_showcase_tasks_must_use_frozen_query_authority": True,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    scenarios = list(manifest.get("scenarios") or [])
    ids = [str(item.get("scenario_id") or "") for item in scenarios]
    if not 4 <= len(scenarios) <= 6:
        errors.append("scenario_count_out_of_range")
    if len(ids) != len(set(ids)):
        errors.append("duplicate_scenario_id")
    for item in scenarios:
        if item.get("approved") is not True:
            errors.append(f"scenario_not_approved:{item.get('scenario_id')}")
        if not str(item.get("query") or "").strip():
            errors.append(f"empty_query:{item.get('scenario_id')}")
        if not item.get("capabilities"):
            errors.append(f"missing_capability:{item.get('scenario_id')}")
        if not item.get("expected_behavior"):
            errors.append(f"missing_expected_behavior:{item.get('scenario_id')}")
        if not item.get("authoritative_evidence"):
            errors.append(f"missing_evidence:{item.get('scenario_id')}")
        if item.get("runtime_mutation_required") is not False:
            errors.append(f"runtime_mutation_required:{item.get('scenario_id')}")
        for path in item.get("authoritative_evidence") or []:
            if not (ROOT / str(path)).exists():
                errors.append(f"missing_evidence_path:{path}")
    expected_digest = demo_query_digest(scenarios)
    if manifest.get("showcase_demo_query_set_v1_digest") != expected_digest:
        errors.append("demo_query_digest_mismatch")
    coverage = dict(manifest.get("capability_coverage") or {})
    for key in (
        "semantic_retrieval_showcase_covered",
        "structure_aware_showcase_covered",
        "graph_sensitive_showcase_covered",
        "guarded_agent_showcase_covered",
        "evidence_safety_showcase_covered",
    ):
        if coverage.get(key) is not True:
            errors.append(f"missing_required_coverage:{key}")
    if not (coverage.get("hybrid_or_lexical_showcase_covered") is True or coverage.get("hybrid_or_lexical_showcase_exclusion_explained") is True):
        errors.append("hybrid_lexical_unresolved")
    for key, expected in MUTATION_GUARDS.items():
        if manifest.get(key) is not expected:
            errors.append(f"mutation_guard_failed:{key}")
    if manifest.get("active_project_stage") != "project_showcase_delivery":
        errors.append("wrong_project_stage")
    if not str(manifest.get("source_repository_head") or ""):
        errors.append("missing_source_repository_head")
    if manifest.get("all_approved_scenarios_runtime_verified") is not True:
        errors.append("runtime_evidence_not_verified")
    if manifest.get("all_showcase_claims_evidence_traceable") is not True:
        errors.append("claim_evidence_not_traceable")
    return errors


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": "project_showcase_delivery",
        "scenario_count_min": 4,
        "scenario_count_max": 6,
        "required_capabilities": ["semantic_retrieval", "structure_aware_retrieval", "graph_sensitive_retrieval", "guarded_agent", "evidence_safety"],
        "hybrid_or_lexical_optional": True,
        "runtime_mutation_allowed": False,
        "future_showcase_tasks_must_use_frozen_query_authority": True,
        **MUTATION_GUARDS,
    }


def render_guide(manifest: Mapping[str, Any]) -> str:
    scenario_sections = []
    descriptions = {
        "S01": ("先用一个普通问题建立基线，让观众看到系统不是只靠 Graph 才能工作。", "Vector retrieval → BGE reranking → evidence/provenance", "Guard 被评估但不触发恢复。", "Evidence 保留 source/chunk/line provenance。", "观察普通问题只走必要路径，Graph 不会被无条件激活。", "Graph inactive is expected for this scenario."),
        "S02": ("这个问题明确要求把技术路线与测试笔记联系起来，当前 production guard 会触发 structure lane。", "Initial retrieval → Guard → structure expansion → reranking → evidence", "guard_triggered=true；recovery_activated=true；recovery_attempt_count=1。", "结构扩展候选仍进入同一 reranker/evidence 路径。", "观察 retrieval_sources 中的 `structure`，以及 recovery budget 只使用一次。", "This is bounded structure recovery, not an open-ended planner."),
        "S03": ("这是当前 frozen top-3 seed 语义下真实可激活 Graph 的跨文档关系问题。", "Initial retrieval → Graph activation → one-hop LINKS_TO expansion → reranking/evidence", "Guarded recovery activates once; Graph hop depth remains 1。", "Graph-added MVP chunk carries authoritative edge/path provenance。", "观察 `graph_activated=true`、expanded candidate、LINKS_TO relation 和 graph provenance。", "Historical academic-docx query is intentionally not used because it is Graph-inactive under current frozen seeds."),
        "S04": ("这个问题询问知识库没有权威支持的生产 SLA，用来证明系统不会强行编答案。", "Retrieval → evidence/answerability validation → refusal", "Existing Ask boundary returns final_decision=fail_closed。", "Retrieved material is insufficient to support the requested SLA claim, so generation abstains。", "观察 status=refused 与 refusal_reason_code，而不是一个看似合理但无证据的 SLA 数字。", "Fail-closed comes from the existing Ask safety boundary; no threshold was changed for showcase."),
    }
    for item in manifest["scenarios"]:
        why, retrieval, agent, evidence, notice, limitation = descriptions[item["scenario_id"]]
        refs = "\n".join(f"- `{path}`" for path in item["authoritative_evidence"])
        scenario_sections.append(f"""## {item['scenario_id']} — {item['name']}

**User query**

```text
{item['query']}
```

**Why this query is interesting**  
{why}

**Expected retrieval path**  
{retrieval}

**Expected Agent behavior**  
{agent}

**Expected evidence behavior**  
{evidence}

**Capability demonstrated**  
{', '.join(item['capabilities'])}

**What the viewer should notice**  
{notice}

**Authoritative evidence**

{refs}

**Known limitation**  
{limitation}
""")
    return f"""# OPK-RAG Showcase Scenarios — V1 Authority

TASK-0218 freezes the small, authoritative scenario set used by later demo, visualization, video, README and interview work. This document describes current production behavior; it does not reopen RAG optimization.

## Frozen authority

```text
stage={manifest['active_project_stage']}
scenario_count={manifest['scenario_count']}
showcase_demo_query_set_v1_digest={manifest['showcase_demo_query_set_v1_digest']}
production_vector_backend={manifest['production_vector_backend']}
production_embedding_model={manifest['production_embedding_model']}
production_reranker_model={manifest['production_reranker_model']}
default_initial_retrieval_policy={manifest['default_initial_retrieval_policy']}
graph_runtime_hop_depth={manifest['graph_runtime_hop_depth']}
```

{chr(10).join(scenario_sections)}
## Explicit exclusions and historical correction

- Hybrid / lexical is not a V1 showcase scenario because TASK-0215 through TASK-0217 do not provide a clean current production BM25/hybrid-active showcase proof. The runtime is not changed merely to manufacture one.
- The historical query `{HISTORICAL_GRAPH_QUERY}` remains useful as a Graph-negative control, but TASK-0217 proves it does **not** activate Graph under the frozen top-3 seed semantics. It must not be presented as the Graph-active demo.
- The authoritative Graph-active demo is `{GRAPH_QUERY}`.

## Freeze rule

TASK-0219 and later showcase tasks must consume `evaluation-data/showcase/showcase_manifest_v1.json`. Changing these queries requires an explicit showcase-authority revision; it must not happen implicitly inside a demo/UI/video task.
"""


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK0218 Showcase Scenario and Demo Query Freeze Report

```text
task_id=TASK-0218
task_status={summary['task_status']}
showcase_scenario_count={summary['showcase_scenario_count']}
showcase_scenario_set_frozen={str(summary['showcase_scenario_set_frozen']).lower()}
showcase_demo_query_set_frozen={str(summary['showcase_demo_query_set_frozen']).lower()}
showcase_demo_query_set_v1_digest={summary['showcase_demo_query_set_v1_digest']}
all_approved_scenarios_runtime_verified={str(summary['all_approved_scenarios_runtime_verified']).lower()}
showcase_runtime_mutation_required={str(summary['showcase_runtime_mutation_required']).lower()}
```

TASK-0218 freezes four truthful current-production showcase scenarios: ordinary semantic retrieval, guarded structure-aware bounded recovery, one-hop Graph Retrieval V1, and evidence-safety fail-closed behavior. Hybrid/lexical is deliberately excluded because no clean current showcase execution proof exists and adding one does not justify a runtime change.

The historical academic-docx graph query is retained only as a negative-control fact. TASK-0217 proves that it remains Graph-inactive under frozen top-3 seed semantics. TASK-0218 therefore promotes the already-verified Tauri relation query as the V1 Graph-active showcase authority without changing graph policy, seed semantics, hop depth, or runtime code.
"""


def verify_task0218_artifacts() -> dict[str, Any]:
    required = [MANIFEST_PATH, GUIDE_PATH, CONTRACT_PATH, REPORT_PATH, RESULT_DIR / "summary.json", RESULT_DIR / "verification.json"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    errors = validate_manifest(manifest) if manifest else ["missing_manifest"]
    summary = read_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    verification_passed = not missing and not errors and summary.get("task_status") == "complete"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": verification_passed,
        "missing_artifacts": missing,
        "manifest_errors": errors,
        "digest_reproducible": bool(manifest) and manifest.get("showcase_demo_query_set_v1_digest") == demo_query_digest(manifest.get("scenarios") or []),
        "historical_graph_query_not_promoted": bool(manifest) and all(item.get("query") != HISTORICAL_GRAPH_QUERY for item in manifest.get("scenarios") or []),
        "git_commit_created": False,
    }


def run_task0218(*, write: bool = True) -> dict[str, Any]:
    manifest = build_manifest()
    errors = validate_manifest(manifest)
    coverage = manifest["capability_coverage"]
    complete = not errors
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "current_stage": "project_showcase_delivery",
        "showcase_authority_created": complete,
        "showcase_scenario_count": manifest["scenario_count"],
        "showcase_scenario_ids": manifest["scenario_ids"],
        "showcase_scenario_set_frozen": complete,
        "showcase_demo_query_set_frozen": complete,
        "showcase_demo_query_set_v1_digest": manifest["showcase_demo_query_set_v1_digest"],
        **coverage,
        "all_approved_scenarios_runtime_verified": manifest["all_approved_scenarios_runtime_verified"],
        "all_showcase_claims_evidence_traceable": manifest["all_showcase_claims_evidence_traceable"],
        **MUTATION_GUARDS,
        "showcase_manifest_valid": complete,
        "showcase_documentation_ready": complete,
        "future_showcase_tasks_must_use_frozen_query_authority": True,
        "historical_graph_query_promoted": False,
        "authoritative_graph_showcase_query": GRAPH_QUERY,
        "manifest_errors": errors,
        "next_recommended_task": "TASK-0219",
        "git_commit_created": False,
    }
    verification = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "manifest_validation_passed": not errors,
        "manifest_errors": errors,
        "runtime_evidence_validation": manifest["runtime_evidence_validation"],
        "digest_reproducible": manifest["showcase_demo_query_set_v1_digest"] == demo_query_digest(manifest["scenarios"]),
        "historical_graph_query_not_promoted": all(item["query"] != HISTORICAL_GRAPH_QUERY for item in manifest["scenarios"]),
        "verification_passed": complete,
    }
    if write:
        write_json(MANIFEST_PATH, manifest)
        write_json(CONTRACT_PATH, build_contract())
        write_json(RESULT_DIR / "summary.json", summary)
        write_json(RESULT_DIR / "verification.json", verification)
        GUIDE_PATH.write_text(render_guide(manifest), encoding="utf-8")
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary
