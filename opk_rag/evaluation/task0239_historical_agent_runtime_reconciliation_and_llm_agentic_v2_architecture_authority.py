from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "evaluation-data/results/task0239-historical-agent-runtime-reconciliation-and-llm-agentic-v2-architecture-authority"
ARCH = RESULT_DIR / "architecture.json"
SUMMARY = RESULT_DIR / "summary.json"


def build_summary() -> dict[str, Any]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def verify() -> dict[str, Any]:
    summary = build_summary()
    architecture = json.loads(ARCH.read_text(encoding="utf-8"))
    required = {
        "task_status": "complete",
        "historical_agent_asset_reconciliation_complete": True,
        "agentic_v2_architecture_frozen": True,
        "production_agentic_v2_active": False,
        "llm_agent_policy_active": False,
        "llm_agent_loop_active": False,
        "agent_framework_runtime_used": False,
        "graph_hop_contract_max": 1,
        "blocking_failure_count": 0,
    }
    mismatches={k:{"expected":v,"actual":summary.get(k)} for k,v in required.items() if summary.get(k)!=v}
    expected_actions={"hybrid_search","structure_search","graph_search","rewrite_query","inspect_evidence","finish","abstain"}
    if set(architecture.get("action_space",[])) != expected_actions:
        mismatches["action_space"]={"expected":sorted(expected_actions),"actual":architecture.get("action_space")}
    files=[
        ROOT/"tasks/TASK-0239_historical_agent_runtime_reconciliation_and_llm_agentic_v2_architecture_authority.md",
        ROOT/"docs/LLM_AGENTIC_RAG_V2_ARCHITECTURE_AUTHORITY.md",
        ROOT/"docs/TASK0239_HISTORICAL_AGENT_RUNTIME_RECONCILIATION_AND_LLM_AGENTIC_V2_ARCHITECTURE_AUTHORITY_REPORT.md",
    ]
    missing=[str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    return {"task_id":"TASK-0239","verification_passed":not mismatches and not missing,"mismatches":mismatches,"missing_files":missing}
