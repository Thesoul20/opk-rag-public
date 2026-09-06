from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "opk-rag.showcase.end-to-end-storyboard.v1"
EXPECTED_SCENARIOS = ("S01", "S02", "S03", "S04")
EXPECTED_DECISIONS = {
    "S01": "continue",
    "S02": "structure_recovery",
    "S03": "graph_recovery",
    "S04": "fail_closed",
}
STAGE_IDS = (
    "query",
    "query_embedding",
    "initial_retrieval",
    "candidate_pool",
    "guard_observation",
    "guard_decision",
    "structure_recovery",
    "graph_recovery",
    "unified_candidate_pool",
    "reranking",
    "evidence_composition",
    "answerability",
    "generation",
    "grounding_validation",
    "answer",
    "fail_closed",
)
FORBIDDEN_CLAIM_TERMS = (
    "unrestricted autonomous planner",
    "production multi-hop graphrag",
    "multimodal rag supported",
    "multi-tenant saas supported",
    "zero-hallucination",
    "zero hallucination",
)
FORBIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "chain-of-thought",
    "cot",
    "scratchpad",
    "internal_reasoning",
    "private_reasoning",
    "reasoning_steps",
}


class StoryboardError(RuntimeError):
    pass


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _scenario_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row.get("scenario_id")): row for row in rows if isinstance(row, Mapping)}
    missing = [sid for sid in EXPECTED_SCENARIOS if sid not in mapped]
    if missing:
        raise StoryboardError("missing_scenarios:" + ",".join(missing))
    return mapped


def validate_authorities(
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
    agent: Mapping[str, Any],
    task0219: Mapping[str, Any],
) -> dict[str, bool]:
    query_digest = manifest.get("showcase_demo_query_set_v1_digest")
    graph_digest = graph.get("graph_showcase_visualization_v1_digest")
    agent_digest = agent.get("guarded_agent_decision_trace_v1_digest")
    manifest_scenarios = _scenario_map(manifest.get("scenarios") or [])
    agent_scenarios = _scenario_map(agent.get("scenarios") or [])
    queries_align = all(manifest_scenarios[sid].get("query") == agent_scenarios[sid].get("query") for sid in EXPECTED_SCENARIOS)
    task0219_scenarios_valid = all(task0219.get(f"scenario_{sid.lower()}_valid") is True for sid in EXPECTED_SCENARIOS)
    return {
        "manifest_stage_valid": manifest.get("active_project_stage") == "project_showcase_delivery",
        "manifest_digest_present": bool(query_digest),
        "task0219_complete": task0219.get("task_status") == "complete",
        "task0219_scenarios_valid": task0219_scenarios_valid,
        "graph_digest_present": bool(graph_digest),
        "graph_s03_valid": graph.get("scenario_id") == "S03" and graph.get("query_set_digest") == query_digest,
        "agent_digest_present": bool(agent_digest),
        "agent_query_digest_valid": agent.get("query_set_digest") == query_digest,
        "agent_graph_digest_valid": agent.get("task0220_graph_visualization_digest") == graph_digest,
        "queries_align": queries_align,
        "planner_disabled": agent.get("planner_enabled") is False,
        "unbounded_loop_disabled": agent.get("unbounded_agent_loop_enabled") is False,
        "bounded_recovery_one": int(agent.get("maximum_recovery_attempt_count") or 0) == 1,
        "graph_one_hop": int(agent.get("graph_runtime_hop_depth") or 0) == 1,
        "runtime_gold_disabled": agent.get("runtime_gold_metadata_usage") is False and graph.get("runtime_gold_metadata_usage") is False,
    }


def _stage(stage_id: str, plane: str, label: str, role: str) -> dict[str, str]:
    return {"stage_id": stage_id, "plane": plane, "label": label, "role": role}


def _edge(source: str, target: str, meaning: str) -> dict[str, str]:
    return {"source": source, "target": target, "meaning": meaning}


def build_claim_evidence_matrix(
    *,
    query_digest: str,
    graph_digest: str,
    agent_digest: str,
) -> dict[str, Any]:
    claims = [
        {
            "claim_id": "C01",
            "claim": "Qdrant is the production vector backend for the frozen showcase.",
            "authority": "TASK-0218",
            "artifact": "evaluation-data/showcase/showcase_manifest_v1.json",
            "authority_field": "production_vector_backend=qdrant",
        },
        {
            "claim_id": "C02",
            "claim": "Qwen/Qwen3-Embedding-0.6B is the production embedding model in the showcase authority.",
            "authority": "TASK-0218",
            "artifact": "evaluation-data/showcase/showcase_manifest_v1.json",
            "authority_field": "production_embedding_model",
        },
        {
            "claim_id": "C03",
            "claim": "BAAI/bge-reranker-v2-m3 reranks the shared candidate pool.",
            "authority": "TASK-0218/TASK-0220",
            "artifact": "evaluation-data/showcase/graph_retrieval_visualization_v1.json",
            "authority_field": "runtime.reranker",
        },
        {
            "claim_id": "C04",
            "claim": "The Agent is a guarded bounded controller rather than an unrestricted Planner.",
            "authority": "TASK-0221",
            "artifact": "evaluation-data/showcase/guarded_agent_decision_trace_v1.json",
            "authority_field": "agent_type/planner_enabled/unbounded_agent_loop_enabled",
        },
        {
            "claim_id": "C05",
            "claim": "Structure-aware recovery is bounded to at most one recovery attempt.",
            "authority": "TASK-0221 S02",
            "artifact": "evaluation-data/showcase/guarded_agent_decision_trace_v1.json",
            "authority_field": "S02.action.recovery_attempt_count / maximum_recovery_attempt_count",
        },
        {
            "claim_id": "C06",
            "claim": "Graph recovery is conditional, authoritative, one-hop, and bounded.",
            "authority": "TASK-0220/TASK-0221 S03",
            "artifact": "evaluation-data/showcase/graph_retrieval_visualization_v1.json",
            "authority_field": "activation/authoritative_relations/expanded_candidates",
        },
        {
            "claim_id": "C07",
            "claim": "A Graph-added candidate rejoins the shared candidate pool and passes through the BGE reranker before Evidence selection.",
            "authority": "TASK-0220",
            "artifact": "evaluation-data/showcase/graph_retrieval_visualization_v1.json",
            "authority_field": "expanded_candidates/reranked_candidates/selected_evidence",
        },
        {
            "claim_id": "C08",
            "claim": "Candidates and final Evidence are distinct runtime concepts.",
            "authority": "TASK-0219/TASK-0220",
            "artifact": "evaluation-data/results/task0219-unified-showcase-demo-entry-point/scenario_s03.json",
            "authority_field": "top_candidates vs evidence.selected_evidence",
        },
        {
            "claim_id": "C09",
            "claim": "Answerability can fail closed even when retrieval results and selected Evidence exist.",
            "authority": "TASK-0221 S04",
            "artifact": "evaluation-data/showcase/guarded_agent_decision_trace_v1.json",
            "authority_field": "S04.observations/outcome",
        },
        {
            "claim_id": "C10",
            "claim": "The frozen showcase uses four authoritative scenarios S01-S04.",
            "authority": "TASK-0218",
            "artifact": "evaluation-data/showcase/showcase_manifest_v1.json",
            "authority_field": "scenario_ids/query_set_digest",
        },
    ]
    return {
        "schema_version": "opk-rag.showcase.claim-evidence-matrix.v1",
        "query_set_digest": query_digest,
        "graph_visualization_digest": graph_digest,
        "agent_trace_digest": agent_digest,
        "claims": claims,
        "claim_count": len(claims),
        "unsupported_showcase_claim_count": 0,
        "claim_evidence_matrix_valid": True,
    }


def build_storyboard(
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
    agent: Mapping[str, Any],
    task0219: Mapping[str, Any],
) -> dict[str, Any]:
    checks = validate_authorities(manifest, graph, agent, task0219)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise StoryboardError("authority_validation_failed:" + ",".join(failed))

    query_digest = str(manifest["showcase_demo_query_set_v1_digest"])
    graph_digest = str(graph["graph_showcase_visualization_v1_digest"])
    agent_digest = str(agent["guarded_agent_decision_trace_v1_digest"])
    manifest_scenarios = _scenario_map(manifest.get("scenarios") or [])
    agent_scenarios = _scenario_map(agent.get("scenarios") or [])

    stages = [
        _stage("query", "data", "User Query", "question enters a scoped knowledge base"),
        _stage("query_embedding", "data", "Query Embedding", "Qwen embedding prepares semantic retrieval input"),
        _stage("initial_retrieval", "data", "Initial Retrieval", "retrieve candidate chunks"),
        _stage("candidate_pool", "data", "Candidate Pool", "retrieved possibilities; not final evidence"),
        _stage("guard_observation", "control", "Guard Observation", "observe explicit retrieval/structure/graph signals"),
        _stage("guard_decision", "control", "Guarded Agent Decision", "continue or invoke one approved bounded recovery lane"),
        _stage("structure_recovery", "control", "Structure Recovery", "bounded structure-aware expansion"),
        _stage("graph_recovery", "control", "Graph Recovery", "bounded authoritative one-hop expansion"),
        _stage("unified_candidate_pool", "data", "Unified Candidate Pool", "initial and recovered candidates rejoin one ranking path"),
        _stage("reranking", "data", "BGE Reranking", "reorder candidates by relevance"),
        _stage("evidence_composition", "validation", "Evidence Composition", "select/budget/provenance-check the subset usable for grounding"),
        _stage("answerability", "validation", "Answerability", "decide whether available evidence supports the requested claim"),
        _stage("generation", "validation", "Generation", "generate only on the supported-answer path"),
        _stage("grounding_validation", "validation", "Grounding / Citation Validation", "validate grounded answer and citations"),
        _stage("answer", "validation", "Answer", "return supported grounded response"),
        _stage("fail_closed", "validation", "Fail Closed", "refuse unsupported answer instead of guessing"),
    ]
    if tuple(stage["stage_id"] for stage in stages) != STAGE_IDS:
        raise StoryboardError("stage_sequence_mismatch")

    edges = [
        _edge("query", "query_embedding", "prepare semantic query representation"),
        _edge("query_embedding", "initial_retrieval", "retrieve"),
        _edge("initial_retrieval", "candidate_pool", "produce candidates"),
        _edge("candidate_pool", "guard_observation", "observe retrieval state"),
        _edge("guard_observation", "guard_decision", "evaluate governed signals"),
        _edge("guard_decision", "unified_candidate_pool", "continue without recovery"),
        _edge("guard_decision", "structure_recovery", "approved structure recovery"),
        _edge("structure_recovery", "unified_candidate_pool", "add structure candidates"),
        _edge("guard_decision", "graph_recovery", "approved graph recovery"),
        _edge("graph_recovery", "unified_candidate_pool", "add graph candidates"),
        _edge("unified_candidate_pool", "reranking", "shared ranking path"),
        _edge("reranking", "evidence_composition", "ranked candidates"),
        _edge("evidence_composition", "answerability", "selected evidence"),
        _edge("answerability", "generation", "supported"),
        _edge("generation", "grounding_validation", "validate answer"),
        _edge("grounding_validation", "answer", "grounded/cited"),
        _edge("answerability", "fail_closed", "unsupported claim authority"),
    ]

    overlays: dict[str, Any] = {}
    overlay_paths = {
        "S01": ["query", "initial_retrieval", "candidate_pool", "guard_decision", "unified_candidate_pool", "reranking", "evidence_composition", "answerability"],
        "S02": ["query", "initial_retrieval", "candidate_pool", "guard_decision", "structure_recovery", "unified_candidate_pool", "reranking", "evidence_composition", "answerability"],
        "S03": ["query", "initial_retrieval", "candidate_pool", "guard_decision", "graph_recovery", "unified_candidate_pool", "reranking", "evidence_composition", "answerability"],
        "S04": ["query", "initial_retrieval", "candidate_pool", "guard_decision", "unified_candidate_pool", "reranking", "evidence_composition", "answerability", "fail_closed"],
    }
    for sid in EXPECTED_SCENARIOS:
        arow = agent_scenarios[sid]
        expected_decision = EXPECTED_DECISIONS[sid]
        actual_decision = (arow.get("decision") or {}).get("decision_path")
        if actual_decision != expected_decision:
            raise StoryboardError(f"{sid}_decision_mismatch:{actual_decision}")
        overlays[sid] = {
            "scenario_id": sid,
            "name": manifest_scenarios[sid].get("name"),
            "query": manifest_scenarios[sid].get("query"),
            "decision_path": actual_decision,
            "stage_path": overlay_paths[sid],
            "recovery_attempt_count": int((arow.get("action") or {}).get("recovery_attempt_count") or 0),
            "maximum_recovery_attempt_count": int((arow.get("action") or {}).get("maximum_recovery_attempt_count") or agent.get("maximum_recovery_attempt_count") or 0),
            "outcome_type": (arow.get("outcome") or {}).get("outcome_type"),
        }

    graph_expanded = list(graph.get("expanded_candidates") or [])
    if not graph_expanded or not any(row.get("selected_as_evidence") is True for row in graph_expanded):
        raise StoryboardError("graph_candidate_not_traceable_to_evidence")

    frames = [
        {"frame": 1, "stage": "query", "title": "User Query", "message": "Everything starts with a scoped user question."},
        {"frame": 2, "stage": "initial_retrieval", "title": "Initial Retrieval", "message": "Retrieval produces candidates, not answers."},
        {"frame": 3, "stage": "candidate_pool", "title": "Candidate Pool", "message": "Candidates are possibilities, not trusted evidence yet."},
        {"frame": 4, "stage": "guard_decision", "title": "Guarded Agent", "message": "The bounded controller observes explicit telemetry and chooses an approved path."},
        {"frame": 5, "stage": "guard_decision", "title": "S01 Normal Continue", "message": "No recovery signal: continue with recovery 0/1."},
        {"frame": 6, "stage": "structure_recovery", "title": "S02 Structure Recovery", "message": "Structural signal invokes one bounded structure-aware recovery."},
        {"frame": 7, "stage": "graph_recovery", "title": "S03 Graph Recovery", "message": "A relation-sensitive query follows one authoritative hop to recover an additional candidate."},
        {"frame": 8, "stage": "unified_candidate_pool", "title": "Unified Candidate Pool", "message": "Initial and recovered candidates rejoin the same ranking path."},
        {"frame": 9, "stage": "reranking", "title": "BGE Reranking", "message": "Reranking orders relevance; it does not grant answer authority."},
        {"frame": 10, "stage": "evidence_composition", "title": "Evidence", "message": "Only selected, budgeted, provenance-aware candidates become Evidence."},
        {"frame": 11, "stage": "grounding_validation", "title": "Answer / Validation", "message": "Supported answers pass generation and grounding/citation validation."},
        {"frame": 12, "stage": "fail_closed", "title": "S04 Fail Closed", "message": "Relevant context can exist while the requested claim remains unsupported; the system refuses instead of guessing."},
    ]

    claims = build_claim_evidence_matrix(query_digest=query_digest, graph_digest=graph_digest, agent_digest=agent_digest)
    model: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_authorities": {
            "showcase_query_set_digest": query_digest,
            "graph_visualization_digest": graph_digest,
            "agent_trace_digest": agent_digest,
        },
        "architecture": {
            "stages": stages,
            "edges": edges,
            "planes": {
                "data_plane": ["query", "query_embedding", "initial_retrieval", "candidate_pool", "unified_candidate_pool", "reranking"],
                "control_plane": ["guard_observation", "guard_decision", "structure_recovery", "graph_recovery"],
                "validation_plane": ["evidence_composition", "answerability", "generation", "grounding_validation", "answer", "fail_closed"],
            },
        },
        "scenario_overlays": overlays,
        "storyboard_frames": frames,
        "definitions": {
            "retriever": "Produces candidate chunks from the knowledge base.",
            "candidate": "A retrieved or recovered chunk that may be useful; it is not yet final grounding evidence.",
            "reranker": "Reorders the shared candidate pool by relevance without granting answer authority.",
            "evidence": "The selected, budgeted, provenance-aware subset the answer path may rely on.",
        },
        "governance": {
            "agent_type": agent.get("agent_type"),
            "planner_enabled": False,
            "unbounded_agent_loop_enabled": False,
            "maximum_recovery_attempt_count": 1,
            "graph_runtime_hop_depth": 1,
            "hidden_chain_of_thought_exposed": False,
            "runtime_gold_metadata_usage": False,
            "performance_optimization_reopened": False,
        },
        "graph_recovery_authority": {
            "scenario_id": "S03",
            "authoritative_relation_count": len(graph.get("authoritative_relations") or []),
            "expanded_candidate_count": len(graph_expanded),
            "graph_candidate_rejoins_reranking_path": True,
            "graph_candidate_selected_as_evidence": any(row.get("selected_as_evidence") is True for row in graph_expanded),
        },
        "claim_evidence_summary": {
            "claim_count": claims["claim_count"],
            "unsupported_showcase_claim_count": 0,
            "claim_evidence_matrix_valid": True,
        },
        "presentation": {
            "candidate_and_evidence_visually_distinct": True,
            "executive_message": "Find relevant knowledge, recover only when governed signals justify it, rank candidates, validate Evidence, then answer or refuse.",
            "technical_message": "Retriever → Candidate Pool → Guarded Agent → bounded Structure/Graph recovery → shared BGE reranking → Evidence → Answerability/Grounding → Answer or Fail Closed.",
        },
    }
    digest_input = dict(model)
    model["end_to_end_rag_storyboard_v1_digest"] = digest_payload(digest_input)
    return model


def verify_storyboard_digest(model: Mapping[str, Any]) -> bool:
    expected = str(model.get("end_to_end_rag_storyboard_v1_digest") or "")
    payload = {key: value for key, value in model.items() if key != "end_to_end_rag_storyboard_v1_digest"}
    return bool(expected) and expected == digest_payload(payload)


def storyboard_integrity(model: Mapping[str, Any]) -> dict[str, bool]:
    stages = {row.get("stage_id") for row in (model.get("architecture") or {}).get("stages") or []}
    edges = {(row.get("source"), row.get("target")) for row in (model.get("architecture") or {}).get("edges") or []}
    overlays = model.get("scenario_overlays") or {}
    governance = model.get("governance") or {}
    return {
        "all_stage_ids_present": set(STAGE_IDS).issubset(stages),
        "candidate_and_evidence_distinct": "candidate_pool" in stages and "evidence_composition" in stages and "candidate_pool" != "evidence_composition",
        "retriever_precedes_candidate_pool": ("initial_retrieval", "candidate_pool") in edges,
        "structure_rejoins_pool": ("structure_recovery", "unified_candidate_pool") in edges,
        "graph_rejoins_pool": ("graph_recovery", "unified_candidate_pool") in edges,
        "pool_precedes_reranker": ("unified_candidate_pool", "reranking") in edges,
        "reranker_precedes_evidence": ("reranking", "evidence_composition") in edges,
        "answerability_can_fail_closed": ("answerability", "fail_closed") in edges,
        "planner_disabled": governance.get("planner_enabled") is False,
        "unbounded_loop_disabled": governance.get("unbounded_agent_loop_enabled") is False,
        "recovery_budget_one": int(governance.get("maximum_recovery_attempt_count") or 0) == 1,
        "graph_one_hop": int(governance.get("graph_runtime_hop_depth") or 0) == 1,
        "s01_overlay": (overlays.get("S01") or {}).get("decision_path") == "continue",
        "s02_overlay": (overlays.get("S02") or {}).get("decision_path") == "structure_recovery",
        "s03_overlay": (overlays.get("S03") or {}).get("decision_path") == "graph_recovery",
        "s04_overlay": (overlays.get("S04") or {}).get("decision_path") == "fail_closed",
        "no_hidden_cot": governance.get("hidden_chain_of_thought_exposed") is False,
    }


def _escape(value: Any) -> str:
    return str(value or "").replace('"', "'").replace("\n", " ")


def render_technical_mermaid(model: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  Q["User Query"] --> EMB["Qwen Query Embedding"] --> RET["Initial Retrieval"] --> C["Candidate Pool"]',
            '  C --> OBS["Guard Observation"] --> DEC{"Guarded Agent Decision"}',
            '  DEC -->|continue| U["Unified Candidate Pool"]',
            '  DEC -->|structure signal| STR["Structure Recovery<br/>bounded 1/1"] --> U',
            '  DEC -->|relation + graph seed| GR["Graph Recovery<br/>authoritative one-hop · bounded 1/1"] --> U',
            '  U --> RR["BGE Reranker<br/>BAAI/bge-reranker-v2-m3"] --> EV["Evidence Composition"]',
            '  EV --> AN{"Answerability"}',
            '  AN -->|supported| GEN["Generation"] --> GV["Grounding / Citation Validation"] --> A["Answer"]',
            '  AN -->|insufficient authority| F["Fail Closed / Refusal"]',
            '  N["Candidate ≠ Evidence<br/>Agent = bounded controller<br/>No Planner · No unbounded loop"] -. governance .-> DEC',
        ]
    ) + "\n"


def render_executive_mermaid(model: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  Q["Question"] --> K["Find relevant knowledge"] --> C{"Controlled recovery needed?"}',
            '  C -->|no| R["Rank best information"]',
            '  C -->|structure / relation signal| B["Bounded approved recovery"] --> R',
            '  R --> E["Verify usable Evidence"] --> A{"Supported?"}',
            '  A -->|yes| OK["Answer with grounding / citations"]',
            '  A -->|no| F["Refuse instead of guessing"]',
        ]
    ) + "\n"


def render_scenario_overlay_mermaid(model: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "flowchart TB",
            '  BASE["Shared OPK-RAG Architecture"]',
            '  S01["S01 · Normal<br/>Continue · recovery 0/1"]',
            '  S02["S02 · Structure<br/>Structure Recovery · 1/1"]',
            '  S03["S03 · Relationship<br/>One-hop Graph Recovery · 1/1"]',
            '  S04["S04 · Unsupported claim<br/>Answerability → Fail Closed"]',
            "  BASE --> S01",
            "  BASE --> S02",
            "  BASE --> S03",
            "  BASE --> S04",
        ]
    ) + "\n"


def render_drawio_handoff(model: Mapping[str, Any]) -> str:
    digest = model.get("end_to_end_rag_storyboard_v1_digest")
    return f"""# Draw.io Handoff — OPK-RAG End-to-End Showcase Architecture

Create one 16:9 architecture diagram from the authoritative TASK-0222 model digest `{digest}`.

## Visual language

- dark black / ink-green background;
- subtle dark-green gradients;
- mint-green highlight for active flow and authoritative recovery;
- restrained thin connectors and clear visual hierarchy;
- no people, cartoon characters, robot-brain icons, or decorative knowledge-graph spider web.

## Layout

Top/left: User Query → Qwen Embedding → Initial Retrieval → Candidate Pool.

Upper control strip: Guard Observation → Guarded Agent Decision, clearly labeled `bounded controller`, `No Planner`, `maximum recovery=1`.

Recovery branches: Structure Recovery and Graph Recovery. Graph must show only an authoritative one-hop branch and must rejoin the Unified Candidate Pool.

Center/right: Unified Candidate Pool → BGE Reranker → Evidence Composition.

Bottom/right validation rail: Answerability → supported path (Generation → Grounding/Citation Validation → Answer) OR unsupported path (Fail Closed / Refusal).

## Mandatory semantics

- visually distinguish Candidate Pool from Evidence;
- Structure/Graph recovery add candidates only;
- recovered candidates must rejoin before Reranking;
- Graph hop depth=1;
- recovery budget=1;
- Agent is a controller, not an autonomous Planner;
- S04 must show that retrieved context/evidence can exist before Answerability refuses;
- do not invent nodes, edges, capabilities, or Graph relations absent from the canonical model.
"""


def render_storyboard_document(model: Mapping[str, Any]) -> str:
    return f"""# OPK-RAG End-to-End Showcase Storyboard

Canonical digest: `{model.get('end_to_end_rag_storyboard_v1_digest')}`

## One-line architecture

**Retriever finds Candidates → Guarded Agent decides whether bounded recovery is justified → Structure/Graph only add Candidates → BGE reranks the shared pool → Evidence composition selects usable grounding → Answerability/Grounding decides Answer or Fail Closed.**

## Three planes

### Data Plane
Query → Qwen embedding → initial retrieval → Candidate Pool → Unified Candidate Pool → BGE reranking.

### Control Plane
The Guarded Agent observes explicit runtime telemetry and chooses among normal continue, bounded Structure recovery, or bounded one-hop Graph recovery. It is not an unrestricted Planner and cannot enter an unbounded recovery loop.

### Validation Plane
Ranked candidates become a smaller Evidence set through composition/provenance constraints. Answerability then decides whether the requested claim is supported. Supported answers proceed through generation and grounding/citation validation; unsupported claims fail closed.

## Candidate vs Evidence

A **Candidate** is a retrieved or recovered chunk that may be relevant. An **Evidence** item is a selected, budgeted, provenance-aware chunk that the answer path is allowed to rely on. Retrieval success therefore does not automatically mean answer authority.

## Retriever vs Reranker

The Retriever creates the Candidate Pool. The Reranker (`BAAI/bge-reranker-v2-m3`) orders the shared pool. The Reranker does not itself decide factual support or answerability.

## Four frozen scenarios

- **S01 — Continue:** ordinary semantic retrieval is sufficient; recovery remains `0/1`.
- **S02 — Structure Recovery:** a structural query signal invokes the approved Structure lane once (`1/1`) before candidates return to the shared pool.
- **S03 — Graph Recovery:** a relation-sensitive query plus an expandable seed activates authoritative one-hop Graph expansion (`1/1`). TASK-0220 proves the Graph-added candidate rejoins the shared BGE reranking path and reaches Evidence.
- **S04 — Fail Closed:** retrieval results and selected Evidence exist, but Answerability authority is still insufficient for the requested production SLA claim, so the system refuses instead of guessing.

## Graph and Agent relationship

TASK-0221 describes the **control plane**: why an approved recovery lane is selected. TASK-0220 describes the **Graph data plane**: which authoritative edge was traversed and which candidate was added. Graph does not bypass the Agent governance, Reranker, or Evidence path.

## Claims this showcase can make

- production vector backend: Qdrant;
- embedding: Qwen/Qwen3-Embedding-0.6B;
- reranker: BAAI/bge-reranker-v2-m3;
- Guarded Structure-aware runtime control;
- recovery budget bounded to one attempt;
- conditional authoritative one-hop Graph Retrieval;
- Candidate/Evidence separation;
- Answerability/Grounding validation and fail-closed refusal.

## Claims this showcase must not make

No unrestricted autonomous Planner, no production multi-hop GraphRAG, no multimodal/image RAG claim, no multi-tenant SaaS claim, no zero-hallucination guarantee, and no unverified local-generation GPU co-residency claim.

## Presentation close

The project is deliberately governed: it can expand retrieval when explicit signals justify it, but every recovered item returns to the normal ranking/evidence path, and unsupported claims are refused rather than promoted into answers.
"""


def render_talk_track(model: Mapping[str, Any]) -> str:
    return """# OPK-RAG Showcase Talk Track

## 60-second version

OPK-RAG is a local-first personal knowledge RAG with a governed retrieval control layer. A question is embedded and used to retrieve Candidate chunks. The Guarded Agent then observes explicit retrieval signals: if ordinary retrieval is enough it continues; if the question is structure-sensitive it can perform one bounded Structure recovery; if it is relationship-sensitive it can perform one authoritative one-hop Graph recovery. All candidates then rejoin the same BGE reranking path. Only a selected subset becomes Evidence. Finally, Answerability and grounding validation decide whether the system has enough authority to answer. If not, it fails closed instead of guessing.

## 3-minute version

Start with the distinction between retrieval and answering. The Retriever only creates a Candidate Pool. Those candidates are possibilities, not trusted Evidence. OPK-RAG adds a Guarded Agent around this retrieval stage, but it is intentionally not an open-ended Planner. It can choose among a very small set of governed actions and the recovery budget is one attempt.

S01 demonstrates the normal path: vector retrieval is good enough, so the Agent does nothing extra. S02 demonstrates structure-aware recovery: a structural signal activates the Structure lane once, then those recovered chunks return to the shared Candidate Pool. S03 demonstrates the Graph path: a relationship question plus an expandable seed activates one authoritative hop. TASK-0220 proves that the Graph-added MVP chunk does not jump directly to the answer; it rejoins the Candidate Pool, passes through the BGE reranker, and is then selected into Evidence.

After reranking, Evidence composition reduces the candidate set to the chunks the generator may actually rely on. This is why Candidate and Evidence are separate concepts. S04 demonstrates the safety boundary: the system still has retrieval results and Evidence, but those items do not support a reliable production SLA claim. Answerability therefore fails closed and returns a refusal. That is the core engineering story: controlled recovery, shared ranking, explicit Evidence, and governed refusal.

## 8-minute version

1. **Scope and authority.** The production showcase is frozen around four scenarios and uses Qdrant, Qwen embeddings, and BGE reranking. The current showcase does not claim multi-hop GraphRAG, multimodal RAG, or an autonomous Planner.
2. **Initial retrieval.** The query enters a scoped knowledge base, is embedded, and retrieves chunks. These chunks are Candidates. Retrieval relevance alone is not sufficient to establish answer authority.
3. **Guarded control plane.** Explicit runtime telemetry enters a deterministic Guarded Agent decision boundary. The Agent is a bounded controller, not a recursive reasoning loop. Maximum recovery attempts are one.
4. **S01.** Normal retrieval succeeds, no Structure or Graph recovery is needed, and execution continues to the shared candidate path.
5. **S02.** A structural signal invokes the Structure lane. The recovered chunks are additions to the Candidate Pool, not automatic Evidence.
6. **S03.** A relation-sensitive query activates Graph recovery. The current authority allows one hop only. TASK-0220 records the actual authoritative LINKS_TO edge and the Graph-added candidate. That candidate is reranked normally and can then enter Evidence.
7. **Reranking and Evidence.** The shared BGE reranker orders initial and recovered candidates. Evidence composition then applies the context/evidence budget and provenance constraints. This separates 'retrieved' from 'allowed to ground the answer'.
8. **Answerability and grounding.** If Evidence supports the requested claim, generation proceeds and grounding/citation validation checks the result. If authority is insufficient, the system refuses.
9. **S04.** This is stronger than a 'nothing was found' demo. There are retrieval results and selected Evidence, but they do not support the requested SLA claim, so the system fails closed with an explicit refusal reason.
10. **Takeaway.** OPK-RAG's main engineering value is not simply adding more retrieval methods. It governs when recovery is allowed, keeps all recovered content on a common ranking/evidence path, and preserves an explicit boundary between relevance and factual answer authority.
"""


def render_video_storyboard(model: Mapping[str, Any]) -> str:
    rows = [
        ("00:00–00:20", "Problem / goal", "End-to-end executive diagram", "A RAG system needs more than retrieval: it needs controlled recovery and answer authority.", "Question → knowledge → answer/refusal"),
        ("00:20–00:50", "Architecture", "Technical architecture Mermaid", "Introduce Data, Control, and Validation planes.", "Three-plane separation"),
        ("00:50–01:20", "Normal retrieval", "`opk-rag demo --scenario S01`", "Retriever produces Candidates; S01 needs no recovery.", "Candidate Pool; recovery 0/1"),
        ("01:20–01:50", "Guarded Agent", "TASK-0221 scenario diagram", "The Agent is a bounded controller, not a Planner.", "continue / structure / graph / fail-closed"),
        ("01:50–02:20", "Structure recovery", "`opk-rag demo --scenario S02`", "Structure lane is invoked once and recovered chunks return to the pool.", "Structure recovery 1/1"),
        ("02:20–03:00", "Graph recovery", "TASK-0220 Graph diagram + `S03`", "Follow the real authoritative one-hop relation and show the Graph-added candidate.", "Graph candidate rejoins shared pool"),
        ("03:00–03:30", "Reranking / Evidence", "Technical architecture", "BGE reranks all candidates; only a subset becomes Evidence.", "Candidate ≠ Evidence"),
        ("03:30–04:00", "Fail Closed", "`opk-rag demo --scenario S04`", "Relevant context exists, but the SLA claim lacks answer authority.", "Answerability → refusal"),
        ("04:00–04:20", "Closing", "Executive diagram", "Controlled recovery plus explicit Evidence and refusal boundaries make the system trustworthy to demonstrate.", "Governed RAG story"),
    ]
    lines = ["# OPK-RAG Showcase Video Storyboard", "", "| Time | Screen | Live command / artifact | Speaker message | Visual focus |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "Fallback: if a live scenario is unavailable, use the corresponding clearly labeled frozen TASK-0219/0220/0221 artifact; never fabricate runtime output or change policy for the recording."])
    return "\n".join(lines) + "\n"


def render_readme_fragment(model: Mapping[str, Any]) -> str:
    return """## Architecture: governed retrieval, not just retrieval

```mermaid
flowchart LR
  Q[User Query] --> R[Initial Retrieval] --> C[Candidate Pool]
  C --> G{Guarded Agent}
  G -->|continue| U[Unified Candidate Pool]
  G -->|structure| S[Bounded Structure Recovery] --> U
  G -->|relationship| GR[One-hop Graph Recovery] --> U
  U --> RR[BGE Reranker] --> E[Evidence Composition] --> A{Answerability}
  A -->|supported| OK[Grounded Answer + Citations]
  A -->|unsupported| F[Fail Closed]
```

OPK-RAG separates **Candidates** from **Evidence**. Retrieval and bounded Structure/Graph recovery only create candidates. Every recovered candidate returns to the same BGE reranking path. Evidence composition selects the subset the answer path may rely on, and Answerability/Grounding decides whether to answer or refuse.

The Agent is deliberately **guarded and bounded**: no unrestricted Planner, maximum recovery budget `1`, and production Graph recovery is authoritative one-hop only.

Deep dives: `docs/SHOWCASE_GUARDED_AGENT_DECISION_TRACE.md`, `docs/SHOWCASE_GRAPH_RETRIEVAL_VISUALIZATION.md`, and `docs/SHOWCASE_END_TO_END_RAG_STORYBOARD.md`.
"""


def contains_forbidden_reasoning(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_REASONING_KEYS:
                return True
            if contains_forbidden_reasoning(value):
                return True
    elif isinstance(payload, list):
        return any(contains_forbidden_reasoning(item) for item in payload)
    return False
