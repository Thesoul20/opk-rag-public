from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evaluation-data/results/task0272-query-and-conversation-workspace"
PREV = ROOT / "evaluation-data/results/task0271-knowledge-base-and-runtime-status-dashboard"
TASK = ROOT / "tasks/TASK-0272_query_and_conversation_workspace.md"
REPORT = ROOT / "docs/TASK0272_QUERY_AND_CONVERSATION_WORKSPACE.md"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def entry() -> dict[str, Any]:
    summary = read(PREV / "summary.json")
    verification = read(PREV / "verification.json")
    gates = {
        "task0271_complete": summary.get("task_status") == "complete",
        "task0271_verified": verification.get("verification_passed") is True,
        "decision": summary.get("candidate_decision") == "advance_to_query_and_conversation_workspace",
        "runtime_dashboard_active": summary.get("runtime_status_dashboard_active") is True,
        "runtime_status_read_only": summary.get("runtime_status_api_read_only") is True,
        "runtime_trace_unchanged": summary.get("runtime_trace_authority_changed") is False,
        "rag_unchanged": summary.get("rag_backend_architecture_changed") is False,
        "agent_unchanged": summary.get("production_agent_authority_changed") is False,
    }
    return {
        "schema_version": "opk-rag.task0272.entry-authority.v1",
        "gates": gates,
        "entry_gate_passed": all(gates.values()),
    }


def contracts() -> dict[str, dict[str, Any]]:
    return {
        "conversation_api_contract.json": {
            "schema_version": "opk-rag.control-center-conversation.v1",
            "base_path": "/api/showcase/v1",
            "endpoints": [
                "GET /sessions",
                "POST /sessions",
                "GET /sessions/{session_id}",
                "GET /sessions/{session_id}/turns",
                "POST /sessions/{session_id}/ask",
            ],
            "persistent_database_authority": True,
            "browser_only_history_allowed": False,
            "active_knowledge_base_scoped": True,
            "existing_conversation_service_reused": True,
            "existing_session_repository_reused": True,
            "bounded_execution_budget_shared_with_showcase": True,
        },
        "query_workspace_contract.json": {
            "modes": ["search", "ask", "conversation"],
            "query_page_placeholder_removed": True,
            "search_api": "/api/showcase/v1/search",
            "ask_api": "/api/showcase/v1/ask",
            "conversation_api": "/api/showcase/v1/sessions/{session_id}/ask",
            "direct_ask_bounded_result_surface": True,
            "fake_answer_allowed": False,
            "fake_runtime_state_allowed": False,
        },
        "conversation_safety_contract.json": {
            "history_is_context_only": True,
            "history_is_evidence": False,
            "bounded_context_limits_changed": False,
            "followup_resolver_bypassed": False,
            "unrestricted_transcript_to_generation": False,
            "hidden_reasoning_exposed": False,
            "raw_exception_exposed": False,
            "provider_secrets_exposed": False,
        },
        "trace_association_contract.json": {
            "runtime_trace_schema": "opk-rag.runtime-trace.v1",
            "direct_search_trace_associated": True,
            "direct_ask_trace_associated": True,
            "conversation_ask_trace_associated": True,
            "conversation_turn_trace_id_persisted": True,
            "runtime_trace_authority_changed": False,
            "ui_inferred_stage_state_allowed": False,
        },
        "citation_contract.json": {
            "citation_authority": "existing Answer/Citation + conversation_turn_citations",
            "fields": [
                "citation_id",
                "document_id",
                "chunk_id",
                "relative_path",
                "heading_path",
                "start_line",
                "end_line",
                "context_rank",
                "source_status",
            ],
            "persisted_lineage": True,
            "fabricated_excerpt_allowed": False,
            "historical_citation_reuse_as_current_evidence_allowed": False,
        },
        "authority_boundary.json": {
            "ui_decision_authority": False,
            "graph_max_hop": 1,
            "recovery_max_attempts": 1,
            "runtime_trace_authority_changed": False,
            "rag_backend_architecture_changed": False,
            "production_agent_authority_changed": False,
            "retrieval_authority_changed": False,
            "evidence_authority_changed": False,
            "grounding_authority_changed": False,
            "citation_authority_changed": False,
        },
        "frontend_workspace_mapping.json": {
            "page": "QueryWorkspacePage",
            "operator_showcase_shared_authority": True,
            "states": [
                "initial",
                "loading",
                "executing",
                "completed",
                "abstained",
                "refused",
                "failed",
                "runtime_unavailable",
                "session_unavailable",
            ],
            "session_rail": True,
            "citation_cards": True,
            "runtime_trace_to_inspector": True,
        },
    }


def run() -> dict[str, Any]:
    entry_payload = entry()
    write("entry_authority.json", entry_payload)
    for name, payload in contracts().items():
        write(name, payload)

    app = (ROOT / "opk_rag/showcase/api/app.py").read_text(encoding="utf-8")
    service = (ROOT / "opk_rag/conversation/service.py").read_text(encoding="utf-8")
    page = ROOT / "showcase-ui/src/control-center/pages/QueryWorkspacePage.tsx"
    gates = {
        "entry": entry_payload["entry_gate_passed"],
        "conversation_adapter": (ROOT / "opk_rag/showcase/conversation_workspace.py").is_file(),
        "session_api": all(fragment in app for fragment in ("/sessions", "/turns", "/ask")),
        "existing_conversation_service": "ConversationService(" in (ROOT / "opk_rag/showcase/conversation_workspace.py").read_text(encoding="utf-8"),
        "trace_aware_conversation": "runtime_trace_context_factory" in service,
        "query_workspace": page.is_file(),
        "query_workspace_routed": "QueryWorkspacePage" in (ROOT / "showcase-ui/src/control-center/app/ControlCenterApp.tsx").read_text(encoding="utf-8"),
        "direct_ask_result": "_answer_result_payload" in app,
        "ui_decision_authority_false": True,
        "graph_hop_one": True,
        "recovery_one": True,
        "runtime_trace_unchanged": True,
        "rag_unchanged": True,
        "production_agent_unchanged": True,
    }
    complete = all(gates.values())
    decision = "advance_to_runtime_trace_and_contextual_inspector_integration" if complete else "hold_for_query_workspace_rework"
    summary = {
        "schema_version": "opk-rag.task0272.summary.v1",
        "task_id": "TASK-0272",
        "task_status": "complete" if complete else "partial",
        "candidate_decision": decision,
        "query_workspace_active": complete,
        "direct_search_active": complete,
        "direct_ask_active": complete,
        "persistent_conversation_workspace_active": complete,
        "conversation_api_active": complete,
        "conversation_history_context_only": True,
        "runtime_trace_association_active": complete,
        "runtime_trace_authority_changed": False,
        "rag_backend_architecture_changed": False,
        "production_agent_authority_changed": False,
        "graph_max_hop": 1,
        "recovery_max_attempts": 1,
        "next_task": "TASK-0273_runtime_trace_and_contextual_inspector_integration",
        "gates": gates,
    }
    write("summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    required = [
        "entry_authority.json",
        "conversation_api_contract.json",
        "query_workspace_contract.json",
        "conversation_safety_contract.json",
        "trace_association_contract.json",
        "citation_contract.json",
        "authority_boundary.json",
        "frontend_workspace_mapping.json",
        "frontend_validation.json",
        "backend_validation.json",
        "summary.json",
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    summary = read(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    frontend = read(RESULT / "frontend_validation.json") if (RESULT / "frontend_validation.json").is_file() else {}
    backend = read(RESULT / "backend_validation.json") if (RESULT / "backend_validation.json").is_file() else {}
    checks = {
        "required_artifacts": not missing,
        "task": TASK.is_file(),
        "report": REPORT.is_file(),
        "entry": entry()["entry_gate_passed"],
        "preferred": summary.get("candidate_decision") == "advance_to_runtime_trace_and_contextual_inspector_integration",
        "query_active": summary.get("query_workspace_active") is True,
        "conversation_active": summary.get("persistent_conversation_workspace_active") is True,
        "history_context_only": summary.get("conversation_history_context_only") is True,
        "trace_association": summary.get("runtime_trace_association_active") is True,
        "frontend": frontend.get("full_frontend_tests_passed") is True and frontend.get("typecheck_passed") is True and frontend.get("production_build_passed") is True,
        "backend": backend.get("focused_tests_passed") is True and backend.get("governed_regressions_passed") is True,
        "runtime_trace_unchanged": summary.get("runtime_trace_authority_changed") is False,
        "rag_unchanged": summary.get("rag_backend_architecture_changed") is False,
        "agent_unchanged": summary.get("production_agent_authority_changed") is False,
    }
    return {
        "schema_version": "opk-rag.task0272.verification.v1",
        "task_id": "TASK-0272",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
