from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.budget import AgentBudget
from opk_rag.agentic_v2.decision import AgentDecision
from opk_rag.agentic_v2.guard_contracts import AgentGuardDecision
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.termination import AgentTermination

SCHEMAS: dict[str, type] = {
    "state": AgentState,
    "observation": AgentObservation,
    "decision": AgentDecision,
    "action": AgentAction,
    "budget": AgentBudget,
    "guard_decision": AgentGuardDecision,
    "termination": AgentTermination,
}


def contract_schemas() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in SCHEMAS.items()}


def contract_digest() -> str:
    return stable_digest({
        "schemas": contract_schemas(),
        "action_space": ["hybrid_search", "structure_search", "graph_search", "rewrite_query", "inspect_evidence", "finish", "abstain"],
        "budget_authority": AgentBudget().model_dump(mode="json"),
    })


def export_contract_schemas(root: Path) -> dict[str, str]:
    target = root / "evaluation-data" / "contracts"
    target.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, schema in contract_schemas().items():
        path = target / f"agentic_v2_{name}_schema.json"
        text = json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        digests[name] = stable_digest(schema)
    return digests
