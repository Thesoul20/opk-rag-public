from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "opk-rag.showcase.recording-pack.v1"
CLAIM_POLICY_SCHEMA_VERSION = "opk-rag.showcase.claim-policy.v1"
EXPECTED_SCENARIOS = ("S01", "S02", "S03", "S04")
CANONICAL_SCENARIO_ORDER = "S01,S02,S03,S04"

LIVE_COMMANDS = {
    "preflight_list": "uv run opk-rag demo --list",
    "S01": "uv run opk-rag demo --scenario S01",
    "S02": "uv run opk-rag demo --scenario S02",
    "S03": "uv run opk-rag demo --scenario S03",
    "S04": "uv run opk-rag demo --scenario S04",
    "full_sequence": "uv run opk-rag demo",
    "verifier": "uv run python scripts/verify_task0223_showcase_recording_readiness_and_demo_asset_freeze.py",
}

MACHINE_COMMANDS = {
    sid: f"uv run opk-rag demo --scenario {sid} --format json" for sid in EXPECTED_SCENARIOS
}

TECHNICAL_VISUAL_ASSETS = (
    "docs/diagrams/showcase_end_to_end_rag_architecture.mmd",
    "docs/diagrams/showcase_end_to_end_scenario_overlay.mmd",
    "docs/diagrams/showcase_graph_retrieval_s03.mmd",
    "docs/diagrams/showcase_guarded_agent_architecture.mmd",
    "docs/diagrams/showcase_guarded_agent_scenarios.mmd",
)

EXECUTIVE_VISUAL_ASSETS = (
    "docs/diagrams/showcase_end_to_end_rag_executive.mmd",
    "docs/diagrams/showcase_graph_retrieval_s03_executive.mmd",
    "docs/diagrams/showcase_guarded_agent_executive.mmd",
)

PRESENTATION_DOCS = (
    "docs/SHOWCASE_END_TO_END_RAG_STORYBOARD.md",
    "docs/SHOWCASE_END_TO_END_TALK_TRACK.md",
    "docs/SHOWCASE_VIDEO_STORYBOARD.md",
    "docs/SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION.md",
    "docs/SHOWCASE_GUARDED_AGENT_DECISION_TRACE.md",
    "docs/PROJECT_DEMO_RUNBOOK.md",
    "docs/fragments/SHOWCASE_ARCHITECTURE_README.md",
)

FALLBACK_TRACES = {
    sid: f"evaluation-data/results/task0219-unified-showcase-demo-entry-point/scenario_{sid.lower()}.json"
    for sid in EXPECTED_SCENARIOS
}

RECORDING_SEQUENCE = (
    {"step": 1, "action": "open_architecture", "asset": "docs/diagrams/showcase_end_to_end_rag_architecture.mmd", "message": "Introduce the end-to-end governed RAG architecture."},
    {"step": 2, "action": "explain", "topic": "retriever_to_candidate", "message": "Retriever produces Candidates, not answers."},
    {"step": 3, "action": "run_scenario", "scenario_id": "S01", "message": "Normal retrieval continues with recovery 0/1."},
    {"step": 4, "action": "open_agent_view", "asset": "docs/diagrams/showcase_guarded_agent_scenarios.mmd", "message": "Guarded Agent is a bounded controller, not a Planner."},
    {"step": 5, "action": "run_scenario", "scenario_id": "S02", "message": "Structure recovery is conditional and bounded 1/1."},
    {"step": 6, "action": "open_graph_view", "asset": "docs/diagrams/showcase_graph_retrieval_s03.mmd", "message": "Show the authoritative one-hop Graph data path."},
    {"step": 7, "action": "run_scenario", "scenario_id": "S03", "message": "Graph recovery adds a candidate that rejoins normal reranking."},
    {"step": 8, "action": "explain", "topic": "candidate_reranker_evidence", "message": "Candidate and Evidence are distinct; BGE reranks the shared pool before Evidence selection."},
    {"step": 9, "action": "run_scenario", "scenario_id": "S04", "message": "Relevant context can exist while Answerability still fails closed."},
    {"step": 10, "action": "close", "topic": "answerability_fail_closed", "message": "Close with answer authority, grounding, and refusal boundary."},
)

APPROVED_ASSET_CLASSIFICATIONS = {
    # Public diagrams / explanatory docs.
    **{path: "public_safe" for path in TECHNICAL_VISUAL_ASSETS},
    **{path: "public_safe" for path in EXECUTIVE_VISUAL_ASSETS},
    "docs/SHOWCASE_END_TO_END_RAG_STORYBOARD.md": "public_safe",
    "docs/SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION.md": "public_safe",
    "docs/SHOWCASE_GUARDED_AGENT_DECISION_TRACE.md": "public_safe",
    "docs/fragments/SHOWCASE_ARCHITECTURE_README.md": "public_safe",
    # Presenter-only documents.
    "docs/SHOWCASE_END_TO_END_TALK_TRACK.md": "internal_only",
    "docs/SHOWCASE_VIDEO_STORYBOARD.md": "internal_only",
    "docs/PROJECT_DEMO_RUNBOOK.md": "internal_only",
    "docs/SHOWCASE_RECORDING_GUIDE.md": "internal_only",
    "docs/SHOWCASE_RECORDING_SAFETY_CHECKLIST.md": "internal_only",
    # Frozen traces are safe fallbacks but must be explicitly labelled as frozen artifacts.
    **{path: "public_safe" for path in FALLBACK_TRACES.values()},
    # Canonical machine authorities are operational/reference artifacts, not recommended screens.
    "evaluation-data/showcase/showcase_manifest_v1.json": "internal_only",
    "evaluation-data/showcase/graph_retrieval_visualization_v1.json": "internal_only",
    "evaluation-data/showcase/guarded_agent_decision_trace_v1.json": "internal_only",
    "evaluation-data/showcase/end_to_end_rag_storyboard_v1.json": "internal_only",
    "evaluation-data/showcase/showcase_recording_pack_v1.json": "internal_only",
    "evaluation-data/showcase/showcase_claim_policy_v1.json": "internal_only",
    # Explicit screen exclusions.
    ".env": "not_for_recording",
    ".env.local": "not_for_recording",
    "PROJECT_STATE.md": "internal_only",
    "CHANGELOG.md": "internal_only",
}

AFFIRMATIVE_UNSUPPORTED_PATTERNS = (
    re.compile(r"\bsupports?\s+(?:an?\s+)?unrestricted\s+autonomous\s+planner\b", re.I),
    re.compile(r"\bproduction\s+multi-hop\s+graphrag\s+(?:is\s+)?(?:enabled|supported|active)\b", re.I),
    re.compile(r"\bmultimodal(?:/image)?\s+rag\s+(?:is\s+)?(?:enabled|supported|active)\b", re.I),
    re.compile(r"\bmulti-tenant\s+saas\s+(?:is\s+)?(?:enabled|supported|active)\b", re.I),
    re.compile(r"\bdistributed\s+vector\s+cluster\s+(?:is\s+)?(?:enabled|supported|active)\b", re.I),
    re.compile(r"\bzero[- ]hallucination\s+(?:is\s+)?(?:guaranteed|guarantee)\b", re.I),
    re.compile(r"\bfully\s+verified\s+local-generation\s+gpu\s+co-residency\s+(?:is\s+)?(?:available|verified)\b", re.I),
)


class RecordingPackError(RuntimeError):
    pass


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenario_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row.get("scenario_id")): row for row in rows if isinstance(row, Mapping)}
    missing = [sid for sid in EXPECTED_SCENARIOS if sid not in mapped]
    if missing:
        raise RecordingPackError("missing_scenarios:" + ",".join(missing))
    return mapped


def validate_authorities(
    manifest: Mapping[str, Any],
    task0219: Mapping[str, Any],
    graph: Mapping[str, Any],
    agent: Mapping[str, Any],
    storyboard: Mapping[str, Any],
    claim_matrix: Mapping[str, Any],
) -> dict[str, bool]:
    query_digest = manifest.get("showcase_demo_query_set_v1_digest")
    graph_digest = graph.get("graph_showcase_visualization_v1_digest")
    agent_digest = agent.get("guarded_agent_decision_trace_v1_digest")
    storyboard_digest = storyboard.get("end_to_end_rag_storyboard_v1_digest")
    manifest_scenarios = scenario_map(manifest.get("scenarios") or [])
    agent_scenarios = scenario_map(agent.get("scenarios") or [])
    source_authorities = storyboard.get("source_authorities") or {}
    checks = {
        "manifest_stage_valid": manifest.get("active_project_stage") == "project_showcase_delivery",
        "manifest_scenario_order_valid": tuple(str(row.get("scenario_id")) for row in manifest.get("scenarios") or []) == EXPECTED_SCENARIOS,
        "query_digest_present": bool(query_digest),
        "task0219_complete": task0219.get("task_status") == "complete",
        "task0219_scenarios_valid": all(task0219.get(f"scenario_{sid.lower()}_valid") is True for sid in EXPECTED_SCENARIOS),
        "task0219_query_digest_valid": task0219.get("showcase_demo_query_set_v1_digest") == query_digest,
        "graph_digest_present": bool(graph_digest),
        "graph_query_digest_valid": graph.get("query_set_digest") == query_digest,
        "agent_digest_present": bool(agent_digest),
        "agent_query_digest_valid": agent.get("query_set_digest") == query_digest,
        "agent_graph_digest_valid": agent.get("task0220_graph_visualization_digest") == graph_digest,
        "storyboard_digest_present": bool(storyboard_digest),
        "storyboard_query_digest_valid": source_authorities.get("showcase_query_set_digest") == query_digest,
        "storyboard_graph_digest_valid": source_authorities.get("graph_visualization_digest") == graph_digest,
        "storyboard_agent_digest_valid": source_authorities.get("agent_trace_digest") == agent_digest,
        "queries_align": all(manifest_scenarios[sid].get("query") == agent_scenarios[sid].get("query") for sid in EXPECTED_SCENARIOS),
        "claim_matrix_valid": claim_matrix.get("claim_evidence_matrix_valid") is True and int(claim_matrix.get("unsupported_showcase_claim_count") or 0) == 0,
        "claim_matrix_query_digest_valid": claim_matrix.get("query_set_digest") == query_digest,
        "claim_matrix_graph_digest_valid": claim_matrix.get("graph_visualization_digest") == graph_digest,
        "claim_matrix_agent_digest_valid": claim_matrix.get("agent_trace_digest") == agent_digest,
        "planner_disabled": agent.get("planner_enabled") is False,
        "unbounded_loop_disabled": agent.get("unbounded_agent_loop_enabled") is False,
        "recovery_budget_one": int(agent.get("maximum_recovery_attempt_count") or 0) == 1,
        "graph_one_hop": int(agent.get("graph_runtime_hop_depth") or 0) == 1,
    }
    return checks


def build_claim_policy(claim_matrix: Mapping[str, Any], *, storyboard_digest: str) -> dict[str, Any]:
    approved_claims = [
        {
            "claim_id": row.get("claim_id"),
            "claim": row.get("claim"),
            "authority": row.get("authority"),
            "artifact": row.get("artifact"),
        }
        for row in claim_matrix.get("claims") or []
    ]
    policy: dict[str, Any] = {
        "schema_version": CLAIM_POLICY_SCHEMA_VERSION,
        "source_storyboard_digest": storyboard_digest,
        "source_claim_matrix_schema_version": claim_matrix.get("schema_version"),
        "approved_claims": approved_claims,
        "prohibited_or_unverified_claims": [
            "Unrestricted autonomous Planner is a production capability.",
            "Production multi-hop GraphRAG is enabled.",
            "Multimodal or image RAG is enabled.",
            "Multi-tenant SaaS capability is production-ready.",
            "A distributed vector cluster is production-ready.",
            "The system provides a zero-hallucination guarantee.",
            "Local-generation GPU co-residency has been fully verified.",
        ],
        "claim_rules": {
            "runtime_claims_require_frozen_authority": True,
            "negative_boundary_statements_allowed": True,
            "new_capability_claims_allowed": False,
            "zero_hallucination_guarantee_allowed": False,
        },
    }
    policy["showcase_claim_policy_v1_digest"] = digest_payload(policy)
    return policy


def verify_claim_policy_digest(policy: Mapping[str, Any]) -> bool:
    expected = str(policy.get("showcase_claim_policy_v1_digest") or "")
    payload = {k: v for k, v in policy.items() if k != "showcase_claim_policy_v1_digest"}
    return bool(expected) and expected == digest_payload(payload)


def scan_unsupported_affirmative_claims(text: str) -> list[str]:
    return [pattern.pattern for pattern in AFFIRMATIVE_UNSUPPORTED_PATTERNS if pattern.search(text)]


def build_fallback_matrix() -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for sid in EXPECTED_SCENARIOS:
        visualizations: list[str] = ["docs/diagrams/showcase_end_to_end_rag_architecture.mmd"]
        if sid == "S02":
            visualizations.insert(0, "docs/diagrams/showcase_guarded_agent_scenarios.mmd")
        if sid == "S03":
            visualizations = [
                "docs/diagrams/showcase_graph_retrieval_s03.mmd",
                "docs/diagrams/showcase_guarded_agent_scenarios.mmd",
                "docs/diagrams/showcase_end_to_end_rag_architecture.mmd",
            ]
        if sid == "S04":
            visualizations.insert(0, "docs/diagrams/showcase_guarded_agent_scenarios.mmd")
        fallback[sid] = {
            "live_command": LIVE_COMMANDS[sid],
            "machine_command": MACHINE_COMMANDS[sid],
            "fallback_trace": FALLBACK_TRACES[sid],
            "fallback_visualizations": visualizations,
            "label_requirement": "frozen_authoritative_artifact",
        }
    return {
        "schema_version": "opk-rag.showcase.recording-fallback-matrix.v1",
        "scenario_order": list(EXPECTED_SCENARIOS),
        "scenarios": fallback,
    }


def build_public_asset_inventory(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path, classification in sorted(APPROVED_ASSET_CLASSIFICATIONS.items()):
        full = root / path
        rows.append(
            {
                "path": path,
                "classification": classification,
                "exists": full.exists(),
                "recommended_for_public_recording": classification == "public_safe",
            }
        )
    public_safe_count = sum(row["classification"] == "public_safe" and row["exists"] for row in rows)
    missing_required = [
        row["path"]
        for row in rows
        if row["classification"] != "not_for_recording" and not row["exists"]
        and row["path"] not in {"docs/SHOWCASE_RECORDING_GUIDE.md", "docs/SHOWCASE_RECORDING_SAFETY_CHECKLIST.md", "evaluation-data/showcase/showcase_recording_pack_v1.json", "evaluation-data/showcase/showcase_claim_policy_v1.json"}
    ]
    return {
        "schema_version": "opk-rag.showcase.public-asset-inventory.v1",
        "assets": rows,
        "asset_count": len(rows),
        "public_safe_asset_count": public_safe_count,
        "internal_only_asset_count": sum(row["classification"] == "internal_only" for row in rows),
        "not_for_recording_asset_count": sum(row["classification"] == "not_for_recording" for row in rows),
        "unclassified_asset_count": 0,
        "missing_required_asset_count": len(missing_required),
        "missing_required_assets": missing_required,
        "public_asset_inventory_valid": public_safe_count > 0 and not missing_required,
    }


def build_recording_pack(
    *,
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
    agent: Mapping[str, Any],
    storyboard: Mapping[str, Any],
    claim_policy: Mapping[str, Any],
    upstream_asset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    query_digest = str(manifest.get("showcase_demo_query_set_v1_digest") or "")
    pack: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recording_pack_version": "Showcase Recording Pack V1",
        "recording_sequence_version": "Showcase Recording Sequence V1",
        "authority_digests": {
            "showcase_query_set_digest": query_digest,
            "graph_visualization_digest": graph.get("graph_showcase_visualization_v1_digest"),
            "agent_trace_digest": agent.get("guarded_agent_decision_trace_v1_digest"),
            "end_to_end_storyboard_digest": storyboard.get("end_to_end_rag_storyboard_v1_digest"),
            "claim_policy_digest": claim_policy.get("showcase_claim_policy_v1_digest"),
        },
        "canonical_scenario_order": list(EXPECTED_SCENARIOS),
        "sequence": list(RECORDING_SEQUENCE),
        "live_commands": dict(LIVE_COMMANDS),
        "machine_commands": dict(MACHINE_COMMANDS),
        "visual_assets": {
            "technical": list(TECHNICAL_VISUAL_ASSETS),
            "executive": list(EXECUTIVE_VISUAL_ASSETS),
        },
        "presentation_docs": list(PRESENTATION_DOCS),
        "fallback_policy": {
            "hierarchy": ["live_command", "frozen_task0219_trace", "authoritative_visualization", "task0222_storyboard"],
            "label_requirement": "frozen_authoritative_artifact",
        },
        "fallback_matrix": build_fallback_matrix()["scenarios"],
        "recording_safety": {
            "forbidden_screens": [
                ".env files",
                "credential-bearing shell history or terminal scrollback",
                "database URLs/passwords",
                "API keys/tokens/Authorization headers",
                "private Vault absolute paths or unrelated personal documents",
                "personal browser/email/chat tabs",
            ],
            "actual_public_recording_requires_clean_committed_worktree": True,
            "recommended_terminal_hygiene": [
                "start at repository root",
                "use readable terminal size/font",
                "clear irrelevant visible scrollback",
                "avoid environment dumps",
                "keep JSON output width manageable",
            ],
        },
        "claim_policy": {
            "path": "evaluation-data/showcase/showcase_claim_policy_v1.json",
            "digest": claim_policy.get("showcase_claim_policy_v1_digest"),
        },
        "runtime_governance": {
            "agent_type": agent.get("agent_type"),
            "planner_enabled": False,
            "unbounded_agent_loop_enabled": False,
            "maximum_recovery_attempt_count": 1,
            "graph_runtime_hop_depth": 1,
            "demo_specific_retrieval_policy": False,
            "demo_specific_graph_policy": False,
            "demo_specific_agent_policy": False,
            "demo_specific_answerability_policy": False,
            "demo_specific_threshold_override": False,
            "demo_specific_candidate_injection": False,
            "performance_optimization_reopened": False,
        },
        "upstream_asset_hashes": dict(sorted(upstream_asset_hashes.items())),
    }
    pack["showcase_recording_pack_v1_digest"] = digest_payload(pack)
    return pack


def verify_recording_pack_digest(pack: Mapping[str, Any]) -> bool:
    expected = str(pack.get("showcase_recording_pack_v1_digest") or "")
    payload = {k: v for k, v in pack.items() if k != "showcase_recording_pack_v1_digest"}
    return bool(expected) and expected == digest_payload(payload)


def validate_live_scenario(sid: str, row: Mapping[str, Any], graph_authority: Mapping[str, Any]) -> dict[str, bool]:
    guard = row.get("guard") or {}
    graph = row.get("graph") or {}
    answer = row.get("answer") or {}
    checks: dict[str, bool] = {
        "execution_passed": row.get("execution_status") == "passed",
        "scenario_identity": row.get("scenario_id") == sid,
        "recovery_budget_one": int(guard.get("maximum_recovery_attempt_count") or 0) == 1,
        "runtime_gold_false": guard.get("runtime_gold_metadata_usage") is False and graph.get("runtime_gold_metadata_usage") is False,
    }
    if sid == "S01":
        checks.update({
            "continue": guard.get("final_decision") == "continue_to_reranking",
            "recovery_zero": guard.get("recovery_activated") is False and int(guard.get("recovery_attempt_count") or 0) == 0,
        })
    elif sid == "S02":
        checks.update({
            "structure_lane": guard.get("structure_lane_invoked") is True,
            "recovery_one": guard.get("recovery_activated") is True and int(guard.get("recovery_attempt_count") or 0) == 1,
        })
    elif sid == "S03":
        expanded = graph.get("candidate_provenance") or []
        checks.update({
            "graph_active": graph.get("graph_activated") is True,
            "graph_one_hop": int(graph.get("hop_depth") or 0) == 1,
            "expanded_candidate": int(graph.get("expanded_candidate_count") or 0) >= 1,
            "recovery_one": guard.get("recovery_activated") is True and int(guard.get("recovery_attempt_count") or 0) == 1,
            "task0220_query_match": row.get("query") == graph_authority.get("query"),
            "task0220_query_digest_match": row.get("query_set_digest") == graph_authority.get("query_set_digest"),
            "graph_provenance_available": bool(expanded),
        })
    elif sid == "S04":
        checks.update({
            "fail_closed": guard.get("final_decision") == "fail_closed" and guard.get("fail_closed") is True,
            "answer_refused": guard.get("answer_status") == "refused" and answer.get("status") == "refused",
            "refusal_reason": bool(guard.get("refusal_reason_code") or answer.get("refusal_reason_code")),
        })
    else:
        checks["known_scenario"] = False
    return checks


def recording_dry_run(pack: Mapping[str, Any], root: Path, *, recording_guide_ready: bool, safety_checklist_ready: bool) -> dict[str, Any]:
    asset_paths: set[str] = set()
    visual_assets = pack.get("visual_assets") or {}
    for group in visual_assets.values():
        asset_paths.update(str(path) for path in group or [])
    asset_paths.update(str(path) for path in pack.get("presentation_docs") or [])
    for row in (pack.get("fallback_matrix") or {}).values():
        asset_paths.add(str(row.get("fallback_trace")))
        asset_paths.update(str(path) for path in row.get("fallback_visualizations") or [])
    missing_assets = sorted(path for path in asset_paths if path and not (root / path).exists())
    command_checks = {
        key: isinstance(value, str) and value.startswith("uv run ")
        for key, value in (pack.get("live_commands") or {}).items()
    }
    scenario_order = tuple(pack.get("canonical_scenario_order") or [])
    checks = {
        "all_commands_declared": bool(command_checks) and all(command_checks.values()),
        "canonical_scenario_order": scenario_order == EXPECTED_SCENARIOS,
        "all_required_assets_exist": not missing_assets,
        "all_four_fallbacks_present": set((pack.get("fallback_matrix") or {}).keys()) == set(EXPECTED_SCENARIOS),
        "recording_guide_ready": recording_guide_ready,
        "safety_checklist_ready": safety_checklist_ready,
        "speaker_notes_ready": (root / "docs/SHOWCASE_END_TO_END_TALK_TRACK.md").exists(),
        "no_secret_bearing_recommended_paths": all(not path.startswith(".env") for path in asset_paths),
    }
    return {
        "schema_version": "opk-rag.showcase.recording-dry-run.v1",
        "recording_dry_run_count": 1,
        "checks": checks,
        "missing_assets": missing_assets,
        "recording_dry_run_passed": all(checks.values()),
    }


def render_safety_checklist() -> str:
    return """# OPK-RAG Showcase Recording Safety Checklist

## Before sharing the screen

- [ ] Work from the committed repository root and confirm there are no unrelated worktree changes.
- [ ] Close `.env` files and any editor panes that contain secrets.
- [ ] Do not display shell history or terminal scrollback containing credentials.
- [ ] Do not print `DATABASE_URL`, API keys, tokens, passwords, Authorization headers, or full environment dumps.
- [ ] Close unrelated personal documents, private Vault notes, email/chat windows, and personal browser tabs.
- [ ] Use only relative public showcase paths in diagrams and explanations.

## Terminal hygiene

- [ ] Use a stable readable terminal size and font.
- [ ] Clear irrelevant visible scrollback before recording.
- [ ] Keep prompt noise minimal where possible.
- [ ] Prefer the frozen `opk-rag demo` commands rather than ad-hoc shell probes.
- [ ] Use `--format json` only when machine-readable telemetry improves the explanation; avoid dumping environment state.

## Frozen-asset rule

- [ ] Live output is preferred.
- [ ] If a live scene fails, use only the fallback mapped in `Showcase Recording Pack V1`.
- [ ] Label fallback output visibly as `frozen authoritative artifact`.
- [ ] Never change query, Graph activation, recovery budget, threshold, Answerability, or candidate contents to rescue a recording.

## Claim safety

- [ ] Do not claim an unrestricted autonomous Planner.
- [ ] Do not claim production multi-hop GraphRAG.
- [ ] Do not claim multimodal/image RAG, multi-tenant SaaS, distributed vector clustering, zero hallucination, or unverified local-generation GPU co-residency.
- [ ] Keep the core claim: bounded recovery, shared reranking, explicit Evidence, and Answerability fail-closed.
"""


def render_recording_guide(pack: Mapping[str, Any]) -> str:
    digest = pack.get("showcase_recording_pack_v1_digest")
    return f"""# OPK-RAG Showcase Recording Guide

Recording authority: **Showcase Recording Pack V1**
Recording pack digest: `{digest}`

This is the single presenter entry point. Do not invent replacement queries, scene order, runtime policies, or fallback outputs while recording.

## 1. Before recording

1. Use a clean committed repository worktree.
2. Read `docs/SHOWCASE_RECORDING_SAFETY_CHECKLIST.md`.
3. Close personal/private tabs and secret-bearing terminal/editor panes.
4. Run `uv run python scripts/preflight_showcase_recording.py`.
5. Run `uv run opk-rag demo --list` and confirm S01, S02, S03, S04.

## 2. Canonical recording sequence

1. Open `docs/diagrams/showcase_end_to_end_rag_architecture.mmd` and introduce Data / Control / Validation planes.
2. Explain **Retriever → Candidate**: retrieval produces possibilities, not final answer authority.
3. Run `uv run opk-rag demo --scenario S01` and point out normal continue with recovery 0/1.
4. Open `docs/diagrams/showcase_guarded_agent_scenarios.mmd` and explain that the Agent is a bounded controller, not a Planner.
5. Run `uv run opk-rag demo --scenario S02` and show Structure recovery 1/1.
6. Open `docs/diagrams/showcase_graph_retrieval_s03.mmd` and show the frozen authoritative one-hop relation.
7. Run `uv run opk-rag demo --scenario S03`; explain that the Graph-added candidate rejoins the normal candidate pool and BGE reranking path.
8. Return to the end-to-end diagram and explain **Candidate → BGE Reranker → Evidence**. Candidate is not Evidence.
9. Run `uv run opk-rag demo --scenario S04`; explain that relevant context/evidence can exist while Answerability still fails closed.
10. Close with `docs/diagrams/showcase_end_to_end_rag_executive.mmd`: find knowledge → bounded recovery if justified → rank → validate Evidence → answer or refuse.

## 3. Speaker tracks

Use `docs/SHOWCASE_END_TO_END_TALK_TRACK.md`:

- 60-second version for recruiter/executive;
- 3-minute version for standard interview;
- 8-minute version for CTO/RAG engineer deep dive.

Do not introduce claims outside `evaluation-data/showcase/showcase_claim_policy_v1.json`.

## 4. Frozen fallback behavior

If a live scenario fails, do not change runtime behavior to rescue the recording.

- S01 fallback: `evaluation-data/results/task0219-unified-showcase-demo-entry-point/scenario_s01.json`.
- S02 fallback: frozen S02 trace plus `docs/diagrams/showcase_guarded_agent_scenarios.mmd`.
- S03 fallback: frozen S03 trace plus `docs/diagrams/showcase_graph_retrieval_s03.mmd`.
- S04 fallback: frozen S04 trace plus `docs/diagrams/showcase_guarded_agent_scenarios.mmd`.

Every fallback must be visibly labeled **frozen authoritative artifact**. Never present it as live output.

## 5. Failure handling

- If S01 fails, stop the live-runtime section and diagnose later.
- If S02 fails, use only its frozen fallback; do not change Guard policy.
- If S03 fails, use TASK-0220 frozen Graph authority; do not change query, Graph activation, or hop depth.
- If S04 fails, do not substitute an easier query and do not weaken Answerability.

## 6. Public-safe boundary

Use `public_asset_inventory.json` from TASK-0223 results. Public recording/README screens must use only entries classified `public_safe`. `internal_only` artifacts may support preparation but should not be shown as public presentation assets. `not_for_recording` entries must never appear on screen.

## 7. Post-recording verification

Before publishing, verify that the recording contains no credentials/private paths, the scenario order remains S01→S02→S03→S04, fallback scenes are correctly labeled, and spoken claims remain inside the frozen claim policy.

Estimated canonical storyboard duration: approximately **260 seconds (4m20s)** before editing/pauses.
"""
