from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "opk-rag.showcase.guarded-agent-trace.v1"
AGENT_TYPE = "guarded_agent"
EXPECTED_SCENARIOS = ("S01", "S02", "S03", "S04")
DECISION_PATHS = {
    "S01": "continue",
    "S02": "structure_recovery",
    "S03": "graph_recovery",
    "S04": "fail_closed",
}
FORBIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "chain-of-thought",
    "cot",
    "thoughts",
    "scratchpad",
    "internal_reasoning",
    "private_reasoning",
    "reasoning_steps",
}


class AgentTraceError(RuntimeError):
    pass


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _scenario_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row.get("scenario_id")): row for row in rows if isinstance(row, Mapping)}
    missing = [sid for sid in EXPECTED_SCENARIOS if sid not in mapped]
    if missing:
        raise AgentTraceError("missing_scenarios:" + ",".join(missing))
    return mapped


def validate_runtime_scenario(row: Mapping[str, Any]) -> dict[str, bool]:
    sid = str(row.get("scenario_id"))
    guard = row.get("guard") or {}
    graph = row.get("graph") or {}
    answer = row.get("answer") or {}
    base = {
        "execution_passed": row.get("execution_status") == "passed",
        "guarded_agent": guard.get("agent_type") == AGENT_TYPE,
        "guard_evaluated": guard.get("guard_evaluated") is True,
        "recovery_budget_one": int(guard.get("maximum_recovery_attempt_count") or 0) == 1,
        "runtime_gold_metadata_usage_false": guard.get("runtime_gold_metadata_usage") is False and graph.get("runtime_gold_metadata_usage") is False,
    }
    if sid == "S01":
        base.update(
            {
                "no_structure_lane": guard.get("structure_lane_invoked") is False,
                "no_recovery": guard.get("recovery_activated") is False and int(guard.get("recovery_attempt_count") or 0) == 0,
                "graph_inactive": graph.get("graph_activation_evaluated") is True and graph.get("graph_activated") is False,
                "continue_decision": guard.get("final_decision") == "continue_to_reranking",
            }
        )
    elif sid == "S02":
        base.update(
            {
                "guard_triggered": guard.get("guard_triggered") is True,
                "structure_lane": guard.get("structure_lane_invoked") is True,
                "recovery_active": guard.get("recovery_activated") is True,
                "recovery_bounded": 0 < int(guard.get("recovery_attempt_count") or 0) <= 1,
                "graph_inactive": graph.get("graph_activated") is False,
                "continue_decision": guard.get("final_decision") == "continue_to_reranking",
            }
        )
    elif sid == "S03":
        base.update(
            {
                "graph_evaluated": graph.get("graph_activation_evaluated") is True,
                "graph_active": graph.get("graph_activated") is True,
                "graph_one_hop": int(graph.get("hop_depth") or 0) == 1,
                "graph_expansion_present": int(graph.get("expanded_candidate_count") or 0) >= 1,
                "recovery_active": guard.get("recovery_activated") is True,
                "recovery_bounded": 0 < int(guard.get("recovery_attempt_count") or 0) <= 1,
                "continue_decision": guard.get("final_decision") == "continue_to_reranking",
            }
        )
    elif sid == "S04":
        base.update(
            {
                "fail_closed": guard.get("final_decision") == "fail_closed" and guard.get("fail_closed") is True,
                "answer_refused": answer.get("status") == "refused" and guard.get("answer_status") == "refused",
                "refusal_reason_present": bool(answer.get("refusal_reason_code") or guard.get("refusal_reason_code")),
            }
        )
    else:
        base["known_scenario"] = False
    return base


def _decision_action_outcome(sid: str, row: Mapping[str, Any], graph_authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    guard = row.get("guard") or {}
    graph = row.get("graph") or {}
    evidence = row.get("evidence") or {}
    answer = row.get("answer") or {}
    path = DECISION_PATHS[sid]

    if sid == "S01":
        if guard.get("recovery_activated") is not False or graph.get("graph_activated") is not False:
            raise AgentTraceError("S01_runtime_does_not_support_continue_path")
        decision = {
            "decision_path": path,
            "runtime_final_decision": guard.get("final_decision"),
            "recovery_decision": guard.get("recovery_decision"),
            "decision_reason": guard.get("guard_reason"),
        }
        action = {"action_type": "no_recovery", "recovery_attempt_count": 0, "maximum_recovery_attempt_count": 1}
        outcome = {"outcome_type": "continue_to_reranking", "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0)}
    elif sid == "S02":
        if guard.get("structure_lane_invoked") is not True or guard.get("recovery_activated") is not True:
            raise AgentTraceError("S02_runtime_does_not_support_structure_recovery_path")
        decision = {
            "decision_path": path,
            "runtime_final_decision": guard.get("final_decision"),
            "recovery_decision": guard.get("recovery_decision"),
            "decision_reason": guard.get("guard_reason"),
        }
        action = {
            "action_type": "invoke_structure_lane",
            "structure_candidate_count": int(guard.get("structure_candidate_count") or 0),
            "recovery_attempt_count": int(guard.get("recovery_attempt_count") or 0),
            "maximum_recovery_attempt_count": int(guard.get("maximum_recovery_attempt_count") or 0),
        }
        outcome = {"outcome_type": "continue_to_reranking", "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0)}
    elif sid == "S03":
        if graph.get("graph_activated") is not True or guard.get("recovery_activated") is not True:
            raise AgentTraceError("S03_runtime_does_not_support_graph_recovery_path")
        if graph_authority.get("scenario_id") != "S03" or graph_authority.get("query") != row.get("query"):
            raise AgentTraceError("TASK0220_graph_authority_mismatch")
        decision = {
            "decision_path": path,
            "runtime_final_decision": guard.get("final_decision"),
            "recovery_decision": guard.get("recovery_decision"),
            "graph_activation_reason": graph.get("graph_activation_reason"),
        }
        action = {
            "action_type": "invoke_one_hop_graph_expansion",
            "graph_hop_depth": int(graph.get("hop_depth") or 0),
            "expanded_candidate_count": int(graph.get("expanded_candidate_count") or 0),
            "recovery_attempt_count": int(guard.get("recovery_attempt_count") or 0),
            "maximum_recovery_attempt_count": int(guard.get("maximum_recovery_attempt_count") or 0),
            "task0220_graph_visualization_digest": graph_authority.get("graph_showcase_visualization_v1_digest"),
        }
        outcome = {
            "outcome_type": "continue_to_reranking",
            "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0),
            "graph_enabled_evidence_present": any("graph" in (item.get("retrieval_sources") or []) for item in evidence.get("selected_evidence") or []),
        }
    elif sid == "S04":
        if guard.get("fail_closed") is not True or answer.get("status") != "refused":
            raise AgentTraceError("S04_runtime_does_not_support_fail_closed_path")
        decision = {
            "decision_path": path,
            "runtime_final_decision": guard.get("final_decision"),
            "recovery_decision": guard.get("recovery_decision"),
            "decision_reason": "insufficient_answer_authority",
        }
        action = {
            "action_type": "refuse_unsupported_answer",
            "recovery_attempt_count": int(guard.get("recovery_attempt_count") or 0),
            "maximum_recovery_attempt_count": int(guard.get("maximum_recovery_attempt_count") or 0),
        }
        outcome = {
            "outcome_type": "fail_closed_refusal",
            "answer_status": answer.get("status"),
            "refusal_reason_code": answer.get("refusal_reason_code") or guard.get("refusal_reason_code"),
            "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0),
        }
    else:  # pragma: no cover
        raise AgentTraceError(f"unsupported_scenario:{sid}")
    return decision, action, outcome


def normalize_agent_trace(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    query_set_digest: str,
    graph_authority: Mapping[str, Any],
    source_trace: str,
) -> dict[str, Any]:
    mapped = _scenario_map(scenarios)
    normalized: list[dict[str, Any]] = []
    max_recovery_values: set[int] = set()
    hop_values: set[int] = set()
    runtime_gold = False

    for sid in EXPECTED_SCENARIOS:
        row = mapped[sid]
        checks = validate_runtime_scenario(row)
        if not all(checks.values()):
            failed = [name for name, ok in checks.items() if not ok]
            raise AgentTraceError(f"{sid}_runtime_validation_failed:" + ",".join(failed))
        guard = row.get("guard") or {}
        graph = row.get("graph") or {}
        retrieval = row.get("retrieval") or {}
        evidence = row.get("evidence") or {}
        answer = row.get("answer") or {}
        max_recovery_values.add(int(guard.get("maximum_recovery_attempt_count") or 0))
        hop_values.add(int(graph.get("hop_depth") or 0))
        runtime_gold = runtime_gold or guard.get("runtime_gold_metadata_usage") is True or graph.get("runtime_gold_metadata_usage") is True
        decision, action, outcome = _decision_action_outcome(sid, row, graph_authority)
        observations = {
            "retrieval_result_count": int(retrieval.get("result_count") or 0),
            "reranker_active": retrieval.get("reranker_active") is True,
            "guard_evaluated": guard.get("guard_evaluated") is True,
            "guard_triggered": guard.get("guard_triggered") is True,
            "guard_reason": guard.get("guard_reason"),
            "structure_lane_invoked": guard.get("structure_lane_invoked") is True,
            "structure_candidate_count": int(guard.get("structure_candidate_count") or 0),
            "graph_activation_evaluated": graph.get("graph_activation_evaluated") is True,
            "graph_activated": graph.get("graph_activated") is True,
            "graph_activation_reason": graph.get("graph_activation_reason"),
            "graph_relation_intent": (graph.get("graph_activation_features") or {}).get("relation_intent") is True,
            "graph_expandable_seed_count": int((graph.get("graph_activation_features") or {}).get("graph_expandable_seed_count") or 0),
            "selected_evidence_count": int(evidence.get("selected_evidence_count") or 0),
            "answerable": answer.get("answerable"),
        }
        normalized.append(
            {
                "scenario_id": sid,
                "scenario_name": row.get("scenario_name"),
                "query": row.get("query"),
                "command": row.get("command"),
                "observations": observations,
                "decision": decision,
                "action": action,
                "outcome": outcome,
            }
        )

    if max_recovery_values != {1}:
        raise AgentTraceError(f"recovery_budget_authority_mismatch:{sorted(max_recovery_values)}")
    if hop_values != {1}:
        raise AgentTraceError(f"graph_hop_authority_mismatch:{sorted(hop_values)}")

    model: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "query_set_digest": query_set_digest,
        "agent_type": AGENT_TYPE,
        "planner_enabled": False,
        "unbounded_agent_loop_enabled": False,
        "maximum_recovery_attempt_count": 1,
        "graph_runtime_hop_depth": 1,
        "runtime_gold_metadata_usage": runtime_gold,
        "hidden_chain_of_thought_exposed": False,
        "source_trace": source_trace,
        "task0220_graph_visualization_digest": graph_authority.get("graph_showcase_visualization_v1_digest"),
        "scenarios": normalized,
        "presentation": {
            "scope": "explicit_runtime_decision_telemetry_only",
            "not_chain_of_thought": True,
            "controller_description": "guarded bounded runtime controller",
        },
    }
    digest_payload = dict(model)
    model["guarded_agent_decision_trace_v1_digest"] = _digest(digest_payload)
    return model


def verify_agent_trace_digest(model: Mapping[str, Any]) -> bool:
    expected = str(model.get("guarded_agent_decision_trace_v1_digest") or "")
    payload = dict(model)
    payload.pop("guarded_agent_decision_trace_v1_digest", None)
    return bool(expected) and _digest(payload) == expected


def _esc(value: Any) -> str:
    return str(value or "").replace('"', "'").replace("\n", " ")


def render_architecture_mermaid(model: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  Q["Query"] --> IR["Initial Retrieval"] --> O["Observation<br/>explicit runtime telemetry"] --> D{"Guarded Decision"}',
            '  D -->|sufficient| C["Continue"]',
            '  D -->|structure signal| S["Structure Recovery<br/>bounded 1/1"]',
            '  D -->|relation + graph seed| G["Graph Recovery<br/>one hop · bounded 1/1"]',
            '  D -->|answer authority insufficient| F["Fail Closed"]',
            '  C --> R["Shared Reranking"]',
            '  S --> R',
            '  G --> R',
            '  R --> E["Evidence / Answerability"]',
            '  E --> A["Answer or Controlled Refusal"]',
            '  F --> A',
            '  N["No Planner · No unbounded loop"] -. governance .-> D',
        ]
    ) + "\n"


def render_scenario_mermaid(model: Mapping[str, Any]) -> str:
    rows = {row["scenario_id"]: row for row in model.get("scenarios") or []}
    def label(sid: str) -> str:
        row = rows[sid]
        return f"{sid}<br/>{_esc(row['decision']['decision_path'])}<br/>{_esc(row['action']['action_type'])}"
    return "\n".join(
        [
            "flowchart LR",
            '  CTRL{"Guarded Agent<br/>bounded controller"}',
            f'  S01["{label("S01")}"]',
            f'  S02["{label("S02")}"]',
            f'  S03["{label("S03")}<br/>TASK-0220 Graph authority"]',
            f'  S04["{label("S04")}"]',
            "  CTRL --> S01",
            "  CTRL --> S02",
            "  CTRL --> S03",
            "  CTRL --> S04",
        ]
    ) + "\n"


def render_executive_mermaid(model: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  Q["Question"] --> C{"Controlled decision"}',
            '  C -->|normal question| N["Use normal retrieval"]',
            '  C -->|structure-sensitive| S["Expand approved document structure"]',
            '  C -->|relationship question| G["Follow approved one-hop relation"]',
            '  C -->|unsupported claim| F["Refuse instead of guessing"]',
            '  B["Recovery budget: at most 1"] -.-> C',
        ]
    ) + "\n"


def render_documentation(model: Mapping[str, Any]) -> str:
    rows = list(model.get("scenarios") or [])
    table = [
        "| Scenario | Guard | Recovery | Lane / Action | Attempts | Final |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        obs, action, decision, outcome = row["observations"], row["action"], row["decision"], row["outcome"]
        attempts = f"{action.get('recovery_attempt_count', 0)} / {action.get('maximum_recovery_attempt_count', 1)}"
        table.append(
            f"| {row['scenario_id']} | evaluated={str(obs['guard_evaluated']).lower()} | {decision['decision_path']} | {action['action_type']} | {attempts} | {outcome['outcome_type']} |"
        )
    return f"""# Showcase Guarded Agent Decision Trace

## What this proves

OPK-RAG contains a **Guarded Agent control plane**, not an open-ended Planner. The controller observes explicit Search/Ask telemetry, selects from governed actions, enforces a recovery budget of **1**, preserves Graph hop depth **1**, and can fail closed when answer authority is insufficient.

Canonical trace digest: `{model['guarded_agent_decision_trace_v1_digest']}`.

## Decision telemetry, not chain-of-thought

This artifact contains only explicit runtime fields such as `guard_reason`, `recovery_decision`, `graph_activation_reason`, `structure_lane_invoked`, recovery counters, answer status, and refusal codes. It does **not** expose or infer hidden chain-of-thought, scratchpads, or private model reasoning.

## Four frozen paths

{chr(10).join(table)}

### S01 — Continue

Normal semantic retrieval is sufficient. Guard evaluation occurs, recovery remains inactive (`0 / 1`), Graph remains inactive, and execution continues to shared reranking/evidence selection.

### S02 — Structure recovery

The explicit guard reason identifies a structural query signal. The governed structure lane is invoked, exactly one bounded recovery attempt is used, then the merged candidate pool continues through normal reranking/evidence selection.

### S03 — Graph recovery

Graph activation is driven by explicit relation/query signals plus graph-expandable seed availability. The control plane uses one bounded recovery attempt and one-hop Graph expansion. Graph edge/candidate details are owned by TASK-0220 (`{model.get('task0220_graph_visualization_digest')}`), avoiding an independent reconstruction here.

### S04 — Fail closed

Retrieval can still return context, but answerability authority is insufficient. The final explicit runtime decision is `fail_closed`, answer status is `refused`, and the system refuses rather than inventing an SLA.

## Why this is an Agent

The Agent role is control, not free-form generation: observe state → select an approved lane → enforce budget → continue or fail closed. It can choose normal continuation, structure-aware recovery, Graph recovery, or controlled refusal.

## Why this is not a Planner

`planner_enabled=false` and `unbounded_agent_loop_enabled=false`. There is no recursive task decomposition, arbitrary tool selection, unlimited retry loop, or unrestricted multi-hop exploration in this showcase authority.
"""


def render_drawio_handoff(model: Mapping[str, Any]) -> str:
    return f"""# Draw.io Handoff — Guarded Agent Decision Trace

Create a 16:9 presentation diagram from the canonical authority `evaluation-data/showcase/guarded_agent_decision_trace_v1.json`.

## Layout

Use a left-to-right control-plane flow with four visually separated bands: Observation → Decision → Permitted Action → Outcome. Put one central **Guarded Agent Controller** decision node in the middle and branch to four frozen scenario paths:

- S01: Continue / no recovery / 0 of 1.
- S02: Structure Recovery / invoke structure lane / 1 of 1.
- S03: Graph Recovery / one-hop expansion / 1 of 1. Reference TASK-0220 Graph visualization digest `{model.get('task0220_graph_visualization_digest')}` for Graph data-plane details.
- S04: Fail Closed / controlled refusal.

Add governance callouts: **No Planner**, **No unbounded loop**, **Maximum recovery attempts = 1**, **Graph hop depth = 1**, **Decision telemetry only — no chain-of-thought**.

Do not invent decisions, actions, graph relations, or hidden reasoning. Preserve the exact scenario meanings from the canonical trace.
"""


def contains_hidden_reasoning_fields(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_REASONING_KEYS:
                return True
            if contains_hidden_reasoning_fields(value):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(contains_hidden_reasoning_fields(item) for item in payload)
    return False
