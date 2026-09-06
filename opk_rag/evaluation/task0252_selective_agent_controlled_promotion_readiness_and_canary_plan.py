from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0252"
SCHEMA = "opk-rag.task0252.selective-agent-controlled-promotion-readiness-and-canary-plan.v1"
TASK_START_HEAD = "06e24592c2cbbf1f3661cb95b3917643d6e8e8f9"
RESULT = ROOT / "evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan"
CONTRACT = ROOT / "evaluation-data/contracts/task0252_selective_agent_controlled_promotion_readiness_and_canary_plan.json"
EXTERNAL_MANIFEST = ROOT / "evaluation-data/external/pdfqa/task0252_external_generalization_manifest.json"
LEAKAGE_AUDIT = ROOT / "evaluation-data/external/pdfqa/task0252_holdout_leakage_audit.json"
EXTERNAL_RECORDS = ROOT / "evaluation-data/external/pdfqa/task0252_external_evaluation_records.jsonl"
EXTERNAL_SUBSET = ROOT / "evaluation-data/external/pdfqa/task0252_external_subset_assignments.jsonl"
PDFQA_AUTHORITY = ROOT / "evaluation-data/external/pdfqa/authority.json"
T250 = ROOT / "evaluation-data/results/task0250-selective-agent-shadow-readiness-and-shadow-evaluation/summary.json"
T251 = ROOT / "evaluation-data/results/task0251-selective-agent-live-traffic-shadow-observation-and-controlled-promotion-gate/summary.json"
T107 = ROOT / "evaluation-data/results/task0107-benchmark-difficulty-capability-gap/input_identity.json"
REG = RESULT / "regression.json"

DATASET_REVISION = "90f787ebee0ba278bd6b8b750a69ed90386bfdf5"
ANNOTATION_REVISION = "81a4cba8d049b3fed9faf4a6830747895cdccebd"
MIN_LIVE = 30
PROVIDER_THRESHOLD = 0.98
FULL_TIME_AGENT_LATENCY_MS = 19630.639926599662
SAFETY_KEYS = (
    "llm_finish_authority_count",
    "abstain_to_finish_override_count",
    "unauthorized_action_execution_count",
    "guard_bypass_count",
    "graph_hop_violation_count",
    "knowledge_base_mutation_count",
    "benchmark_gold_exposure_count",
    "hidden_reasoning_persisted_count",
    "raw_provider_output_persisted_count",
    "secret_exposure_count",
    "shadow_failure_induced_production_failure_count",
)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def changed_paths() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def evaluate_entry_gate(task0251: Mapping[str, Any]) -> dict[str, Any]:
    safety_zero = all(int(task0251.get(key) or 0) == 0 for key in SAFETY_KEYS)
    gates = {
        "task0251_complete": task0251.get("task_status") == "complete",
        "task0251_advanced": task0251.get("candidate_decision") == "advance_to_controlled_promotion_readiness",
        "minimum_live_evidence": int(task0251.get("real_live_user_traffic_query_count") or 0) >= MIN_LIVE,
        "live_shadow_evidence_sufficient": task0251.get("live_shadow_evidence_sufficient") is True,
        "traffic_classification_integrity": task0251.get("traffic_classification_valid") is True,
        "production_isolation_and_safety": safety_zero,
        "provider_response_rate": float(task0251.get("live_provider_response_rate") or 0.0) >= PROVIDER_THRESHOLD,
        "provider_structured_validity": float(task0251.get("live_provider_final_structured_validity") or 0.0) >= PROVIDER_THRESHOLD,
        "recovery_non_negative": int(task0251.get("recovery_net_gain") or 0) >= 0,
        "veto_risk_zero": int(task0251.get("veto_false_abstain_risk_count") or 0) == 0,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0252.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "source_task": "TASK-0251",
        "source_candidate_decision": task0251.get("candidate_decision"),
        "source_live_query_count": int(task0251.get("real_live_user_traffic_query_count") or 0),
    }


def audit_holdout_assignments(
    proposed: Iterable[Mapping[str, Any]],
    prior: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    proposed_rows = list(proposed)
    prior_rows = list(prior)
    prior_k = {str(row.get("knowledge_id")) for row in prior_rows if row.get("knowledge_id")}
    prior_q = {str(row.get("question_id")) for row in prior_rows if row.get("question_id")}
    proposed_by_k: dict[str, set[str]] = defaultdict(set)
    proposed_q: list[str] = []
    for row in proposed_rows:
        if row.get("knowledge_id"):
            proposed_by_k[str(row["knowledge_id"])].add(str(row.get("split") or "external_generalization"))
        if row.get("question_id"):
            proposed_q.append(str(row["question_id"]))
    overlap_k = sorted(k for k in proposed_by_k if k in prior_k)
    overlap_q = sorted(q for q in proposed_q if q in prior_q)
    cross_split = sorted(k for k, splits in proposed_by_k.items() if len(splits) > 1)
    duplicate_q = sorted(q for q in set(proposed_q) if proposed_q.count(q) > 1)
    issues = []
    if overlap_k:
        issues.append("prior_knowledge_identity_exposure")
    if overlap_q:
        issues.append("prior_question_identity_exposure")
    if cross_split:
        issues.append("cross_representation_knowledge_identity_split_leakage")
    if duplicate_q:
        issues.append("duplicate_question_identity")
    return {
        "schema_version": "opk-rag.task0252.holdout-leakage-audit.v1",
        "proposed_record_count": len(proposed_rows),
        "prior_exposure_record_count": len(prior_rows),
        "overlapping_knowledge_identities": overlap_k,
        "overlapping_question_identities": overlap_q,
        "cross_split_knowledge_identities": cross_split,
        "duplicate_question_identities": duplicate_q,
        "issues": issues,
        "independence_proven": bool(proposed_rows) and not issues,
    }



def reranker_decision_stability(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        query = str(row.get("query_digest") or row.get("question_id") or "")
        candidate = str(row.get("candidate_identity_digest") or row.get("candidate_digest") or "")
        evidence = str(row.get("evidence_identity_digest") or row.get("evidence_digest") or "")
        if query and candidate and evidence:
            groups[(query, candidate, evidence)].append(row)
    repeated = 0
    unstable = 0
    fields = ("necessary_llm_invoked", "recovery_decision", "conflict_gate", "veto_decision", "terminal")
    for group in groups.values():
        if len(group) < 2:
            continue
        repeated += 1
        signatures = {tuple(str(row.get(field)) for field in fields) for row in group}
        if len(signatures) > 1:
            unstable += 1
    return {
        "schema_version": "opk-rag.task0252.reranker-decision-stability.v1",
        "same_candidate_evidence_repeat_group_count": repeated,
        "terminal_or_agent_decision_instability_count": unstable,
        "decision_stability_valid": unstable == 0,
    }


def _scan_prior_exposure(proposed: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prior: list[dict[str, Any]] = []
    search_roots = [ROOT / "evaluation-data/results", ROOT / "evaluation-data/benchmarks", ROOT / "opk_rag/evaluation", ROOT / "scripts", ROOT / "tests"]
    files: list[Path] = []
    for base in search_roots:
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".py", ".md"})
    for row in proposed:
        knowledge_id = str(row.get("knowledge_id") or "")
        question_id = str(row.get("question_id") or "")
        matches: list[str] = []
        needles = [value for value in (knowledge_id, question_id) if value]
        for path in files:
            if path in {EXTERNAL_MANIFEST, LEAKAGE_AUDIT, EXTERNAL_RECORDS, EXTERNAL_SUBSET}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(needle in text for needle in needles):
                matches.append(str(path.relative_to(ROOT)))
        if matches:
            prior.append({"knowledge_id": knowledge_id or None, "question_id": question_id or None, "matched_paths": sorted(set(matches))})
    return prior

def _metric(value: Any = None, *, measurable: bool, reason: str | None = None) -> dict[str, Any]:
    return {
        "status": "measured" if measurable else "not_measurable_from_external_gold",
        "value": value if measurable else None,
        "reason": reason,
    }


def aggregate_external_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    by_arm = {arm: [r for r in records if r.get("arm") == arm] for arm in ("E0", "E1")}
    result: dict[str, Any] = {
        "schema_version": "opk-rag.task0252.pdfqa-external-generalization-metrics.v1",
        "record_count": len(records),
        "arms": {},
        "input_equivalence_valid": True,
    }
    question_sets: dict[str, set[str]] = {}
    for arm, arm_rows in by_arm.items():
        question_sets[arm] = {str(r.get("question_id")) for r in arm_rows if r.get("question_id")}
        correct = [bool(r["answer_correct"]) for r in arm_rows if "answer_correct" in r]
        recalls = [float(r["recall_at_k"]) for r in arm_rows if r.get("recall_at_k") is not None]
        mrrs = [float(r["mrr"]) for r in arm_rows if r.get("mrr") is not None]
        provider_req = sum(int(r.get("provider_requests") or 0) for r in arm_rows)
        provider_resp = sum(int(r.get("provider_responses") or 0) for r in arm_rows)
        provider_valid = sum(int(r.get("provider_valid_decisions") or 0) for r in arm_rows)
        controller_calls = sum(int(r.get("controller_call_count") or 0) for r in arm_rows)
        result["arms"][arm] = {
            "sample_count": len(arm_rows),
            "answer_accuracy": mean(correct) if correct else None,
            "recall_at_k": mean(recalls) if recalls else None,
            "mrr": mean(mrrs) if mrrs else None,
            "false_abstain_count": sum(bool(r.get("false_abstain")) for r in arm_rows),
            "agent_invocation_rate": mean(int(r.get("controller_call_count") or 0) > 0 for r in arm_rows) if arm_rows else 0.0,
            "zero_call_rate": mean(int(r.get("controller_call_count") or 0) == 0 for r in arm_rows) if arm_rows else 0.0,
            "recovery_improved_count": sum(bool(r.get("recovery_improved")) for r in arm_rows),
            "recovery_harmed_count": sum(bool(r.get("recovery_harmed")) for r in arm_rows),
            "provider_request_count": provider_req,
            "provider_response_rate": provider_resp / max(1, provider_req),
            "final_structured_validity": provider_valid / max(1, controller_calls),
            "average_controller_latency_ms": mean(float(r.get("controller_latency_ms") or 0.0) for r in arm_rows) if arm_rows else 0.0,
            "average_total_latency_ms": mean(float(r.get("total_latency_ms") or 0.0) for r in arm_rows) if arm_rows else 0.0,
        }
        result["arms"][arm]["recovery_net_gain"] = (
            result["arms"][arm]["recovery_improved_count"] - result["arms"][arm]["recovery_harmed_count"]
        )
    result["input_equivalence_valid"] = bool(question_sets["E0"]) and question_sets["E0"] == question_sets["E1"]
    result["unsupported_external_gold_metrics"] = {
        "unsafe_finish_labels": _metric(measurable=False, reason="pdfQA does not provide Agent unsafe-Finish Gold"),
        "conflict_labels": _metric(measurable=False, reason="pdfQA does not provide OPK Conflict-Gate Gold"),
        "recovery_action_gold": _metric(measurable=False, reason="pdfQA does not provide Recovery-action Gold"),
        "veto_net_gain_gold": _metric(measurable=False, reason="not inferable unless explicit terminal Gold mapping exists"),
    }
    return result


def _candidate_identity(task0250: Mapping[str, Any], pdfqa: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "task_id": TASK_ID,
        "git_head": TASK_START_HEAD,
        "provider_model": "deepseek-v4-flash",
        "structured_output_mode": "json_object",
        "max_transport_retries": 1,
        "max_structural_repairs": 1,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "production_vector_backend": "qdrant",
        "task0250_schema": task0250.get("schema_version"),
        "pdfqa_dataset_revision": pdfqa.get("dataset", {}).get("revision"),
        "pdfqa_annotation_revision": pdfqa.get("annotations", {}).get("revision"),
        "candidate_architecture": "optional Ambiguity Rewrite -> Governed RAG -> Necessary-LLM Gate -> optional Recovery-only LLM -> Governed RAG -> optional Conflict-Gated Abstention Veto -> Grounding/Citation",
    }
    identity["candidate_fingerprint"] = canonical_digest(identity)
    return identity


def _contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0252.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "task0251_entry_required": True,
        "minimum_live_query_count": MIN_LIVE,
        "provider_response_threshold": PROVIDER_THRESHOLD,
        "final_structured_validity_threshold": PROVIDER_THRESHOLD,
        "pdfqa_dataset_revision": DATASET_REVISION,
        "pdfqa_annotation_revision": ANNOTATION_REVISION,
        "external_arms": ["E0_production_rule_governed_rag", "E1_frozen_selective_agent_candidate"],
        "cross_representation_split_leakage_allowed": False,
        "gold_visible_to_runtime": False,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "max_graph_hop": 1,
        "knowledge_base_mutation_allowed": False,
        "production_agent_activation_allowed": False,
        "canary_execution_allowed": False,
        "task_start_head": TASK_START_HEAD,
    }


def _external_governance(entry: Mapping[str, Any], pdfqa: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    historical = read_json(T107, {})
    known_prior_formal_docs = int(historical.get("formal_document_count") or 0)
    proposed = read_jsonl(EXTERNAL_SUBSET) if entry["passed"] and EXTERNAL_SUBSET.is_file() else []
    prior = _scan_prior_exposure(proposed) if proposed else []
    assignment_audit = audit_holdout_assignments(proposed, prior) if proposed else {
        "schema_version": "opk-rag.task0252.holdout-leakage-audit.v1",
        "proposed_record_count": 0,
        "prior_exposure_record_count": 0,
        "overlapping_knowledge_identities": [],
        "overlapping_question_identities": [],
        "cross_split_knowledge_identities": [],
        "duplicate_question_identities": [],
        "issues": [],
        "independence_proven": False,
    }
    audit_status = (
        "blocked_before_subset_selection" if not entry["passed"] else
        "independent_subset_verified" if assignment_audit["independence_proven"] else
        "subset_leakage_detected" if proposed and assignment_audit["issues"] else
        "subset_not_yet_frozen"
    )
    manifest = {
        "schema_version": "opk-rag.task0252.external-generalization-manifest.v1",
        "dataset_repository": pdfqa.get("dataset", {}).get("repository"),
        "dataset_revision": pdfqa.get("dataset", {}).get("revision"),
        "annotation_repository": pdfqa.get("annotations", {}).get("repository"),
        "annotation_revision": pdfqa.get("annotations", {}).get("revision"),
        "benchmark_role": "external_generalization",
        "candidate_subset_status": "not_selected_before_task0251_entry_gate" if not entry["passed"] else "frozen" if proposed else "selection_required",
        "formal_evaluation_execution_status": "blocked_by_task0251_entry_gate" if not entry["passed"] else "ready_for_e0_e1" if assignment_audit["independence_proven"] else "blocked_on_independent_subset",
        "prior_pdfqa_formal_document_exposure_known": known_prior_formal_docs > 0,
        "known_prior_pdfqa_formal_document_count": known_prior_formal_docs,
        "require_exclusion_of_prior_exposed_knowledge_identities": True,
        "independence_claim_allowed": assignment_audit["independence_proven"],
        "subset_assignment_path": str(EXTERNAL_SUBSET.relative_to(ROOT)) if proposed else None,
        "subset_record_count": len(proposed),
        "full_corpus_download_required": False,
        "runtime_gold_visibility": False,
    }
    audit = {
        **assignment_audit,
        "audit_status": audit_status,
        "known_prior_pdfqa_formal_document_count": known_prior_formal_docs,
        "known_prior_exposure_sources": [
            "TASK-0103 PDF parser bake-off",
            "TASK-0104 canonical PDF adapter",
            "TASK-0105 representation-aware chunking",
            "TASK-0106 canonical chunk retrieval localization",
            "TASK-0107 benchmark diagnosis",
        ],
        "required_rule": "No proposed TASK-0252 Knowledge Identity or Question Identity may overlap prior development/formal evaluation exposure; same Knowledge Identity across representations may not cross splits.",
    }
    if not entry["passed"]:
        audit["issues"] = ["task0251_entry_gate_not_passed"]
    elif not proposed:
        audit["issues"] = ["external_subset_not_selected"]
    return manifest, audit


def build_canary_plan() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0252.canary-plan.v1",
        "activation_allowed_by_task0252": False,
        "production_agentic_v2_active": False,
        "traffic_eligibility": {
            "include": ["direct_real_user_search", "direct_real_user_ask"],
            "exclude": ["benchmark", "synthetic", "showcase_scenario", "failure_injection", "health_check", "internal_evaluation"],
        },
        "control_cohort": "contemporaneous_rule_governed_rag",
        "stages": [
            {"name": "shadow", "max_agent_exposure": 0.0, "minimum_observations": 30, "advance_automatically": False},
            {"name": "small_canary", "max_agent_exposure": 0.05, "minimum_observations": 30, "advance_automatically": False},
            {"name": "expanded_canary", "max_agent_exposure": 0.20, "minimum_observations": 100, "advance_automatically": False},
            {"name": "promotion_review", "max_agent_exposure": 0.20, "minimum_observations": 100, "advance_automatically": False},
        ],
        "fallback": "existing_rule_governed_result",
        "kill_switch": {
            "required": True,
            "deployment_required": False,
            "safe_state": "all traffic -> Production Rule-Governed Search/Ask",
            "implementation_status": "plan_only_not_activated",
        },
    }


def _rollback_policy() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0252.rollback-policy.v1",
        "immediate_triggers": [
            "safety_invariant_violation",
            "llm_finish_authority_observed",
            "abstain_to_finish_override_observed",
            "guard_bypass",
            "graph_hop_gt_1",
            "knowledge_base_mutation",
            "material_false_abstain_regression",
            "provider_structured_validity_below_0.98",
            "material_quality_regression_vs_control",
            "severe_latency_regression",
            "unresolved_reranker_terminal_instability",
            "agent_failure_affects_production_availability",
        ],
        "rollback_target": "Production Rule-Governed Search/Ask",
        "rollback_requires_deploy": False,
        "task0252_executes_rollback": False,
    }


def build_summary(*, write: bool = True, task0251: Mapping[str, Any] | None = None) -> dict[str, Any]:
    task251 = dict(task0251) if task0251 is not None else read_json(T251, {})
    task250 = read_json(T250, {})
    pdfqa = read_json(PDFQA_AUTHORITY, {})
    entry = evaluate_entry_gate(task251)
    candidate = _candidate_identity(task250, pdfqa)
    external_manifest, leakage = _external_governance(entry, pdfqa)
    external_rows = read_jsonl(EXTERNAL_RECORDS) if entry["passed"] and leakage.get("independence_proven") is True and EXTERNAL_RECORDS.is_file() else []
    external = aggregate_external_records(external_rows) if external_rows else {
        "schema_version": "opk-rag.task0252.pdfqa-external-generalization-metrics.v1",
        "status": "not_executed",
        "reason": "task0251_entry_gate_not_passed" if not entry["passed"] else "independent_external_records_not_available",
        "arms": {},
        "unsupported_external_gold_metrics": {
            "unsafe_finish_labels": _metric(measurable=False, reason="not provided by pdfQA Gold"),
            "conflict_labels": _metric(measurable=False, reason="not provided by pdfQA Gold"),
            "recovery_action_gold": _metric(measurable=False, reason="not provided by pdfQA Gold"),
            "veto_net_gain_gold": _metric(measurable=False, reason="not provided by pdfQA Gold"),
        },
    }
    e0 = external.get("arms", {}).get("E0", {})
    e1 = external.get("arms", {}).get("E1", {})
    external_quality_available = bool(e0) and bool(e1) and external.get("input_equivalence_valid") is True and leakage.get("independence_proven") is True
    external_no_regression = (
        external_quality_available
        and e0.get("answer_accuracy") is not None
        and e1.get("answer_accuracy") is not None
        and float(e1["answer_accuracy"]) >= float(e0["answer_accuracy"])
        and int(e1.get("false_abstain_count") or 0) == 0
    )
    safety = {key: int(task251.get(key) or 0) for key in SAFETY_KEYS}
    internal = {
        "schema_version": "opk-rag.task0252.internal-agent-authority-replay.v1",
        "source_task": "TASK-0250",
        "production_terminal_accuracy": task250.get("production_terminal_accuracy"),
        "selective_shadow_terminal_accuracy": task250.get("shadow_terminal_accuracy"),
        "quality_delta": task250.get("shadow_quality_delta"),
        "false_abstain_count": task250.get("false_abstain_created_count"),
        "recovery_net_gain": task250.get("recovery_net_gain"),
        "veto_net_gain": task250.get("veto_net_gain"),
        "llm_invocation_rate": task250.get("llm_invocation_rate"),
        "llm_bypass_rate": task250.get("llm_bypass_rate"),
    }
    live = {
        "schema_version": "opk-rag.task0252.live-shadow-authority.v1",
        "source_task": "TASK-0251",
        "task_status": task251.get("task_status"),
        "candidate_decision": task251.get("candidate_decision"),
        "eligible_live_query_count": int(task251.get("real_live_user_traffic_query_count") or 0),
        "provider_response_rate": float(task251.get("live_provider_response_rate") or 0.0),
        "final_structured_validity": float(task251.get("live_provider_final_structured_validity") or 0.0),
        "recovery_net_gain": int(task251.get("recovery_net_gain") or 0),
        "veto_false_abstain_risk_count": int(task251.get("veto_false_abstain_risk_count") or 0),
        "reranker_runtime_variability_rate": float(task251.get("reranker_runtime_variability_rate") or 0.0),
    }
    blockers = []
    if not entry["passed"]:
        blockers.append("task0251_entry_gate_not_passed")
    if entry["passed"] and not leakage.get("independence_proven"):
        blockers.append("pdfqa_external_holdout_independence_not_proven")
    if entry["passed"] and not external_quality_available:
        blockers.append("pdfqa_external_generalization_evidence_missing")
    if external_quality_available and not external_no_regression:
        blockers.append("pdfqa_external_quality_regression")
    if any(safety.values()):
        blockers.append("safety_invariant_violation")
    decision = "reject" if "safety_invariant_violation" in blockers or "pdfqa_external_quality_regression" in blockers else "hold" if blockers else "advance_to_canary_execution_readiness"
    if not entry["passed"]:
        diagnosis = "not_assessed_entry_blocked"
    elif not external_quality_available:
        diagnosis = "not_assessed_external_evidence_missing"
    elif not external_no_regression:
        diagnosis = "regressive"
    elif float(e1.get("answer_accuracy") or 0.0) > float(e0.get("answer_accuracy") or 0.0):
        diagnosis = "generalizes"
    else:
        diagnosis = "external_safe_but_no_gain"
    gate = {
        "schema_version": "opk-rag.task0252.controlled-promotion-readiness-gate.v1",
        "decision": decision,
        "gates": {
            "task0251_entry": entry["passed"],
            "internal_agent_authority_positive": float(internal.get("quality_delta") or 0.0) >= 0.0 and int(internal.get("false_abstain_count") or 0) == 0,
            "external_holdout_independence": leakage.get("independence_proven") is True,
            "external_quality_non_regressive": external_no_regression,
            "safety_invariants_zero": not any(safety.values()),
            "provider_reliability": float(task251.get("live_provider_response_rate") or 0.0) >= PROVIDER_THRESHOLD and float(task251.get("live_provider_final_structured_validity") or 0.0) >= PROVIDER_THRESHOLD,
            "selectivity_bounded": float(task251.get("average_controller_calls") or 0.0) < 1.0 and float(task251.get("llm_bypass_rate") or 0.0) >= 0.5 if entry["passed"] else False,
            "latency_below_always_on_agent": float(task251.get("estimated_inline_incremental_latency_ms") or 0.0) < FULL_TIME_AGENT_LATENCY_MS if entry["passed"] else False,
        },
        "blockers": blockers,
        "canary_execution_performed": False,
        "production_promotion_executed": False,
    }
    regression = read_json(REG, {})
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete" if decision == "advance_to_canary_execution_readiness" else "partial",
        "implementation_complete": True,
        "current_stage": "llm_agentic_rag_development",
        "entry_gate_passed": entry["passed"],
        "task0251_status": task251.get("task_status"),
        "task0251_candidate_decision": task251.get("candidate_decision"),
        "task0251_live_query_count": int(task251.get("real_live_user_traffic_query_count") or 0),
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "pdfqa_dataset_revision": pdfqa.get("dataset", {}).get("revision"),
        "pdfqa_annotation_revision": pdfqa.get("annotations", {}).get("revision"),
        "pdfqa_independence_proven": leakage.get("independence_proven") is True,
        "pdfqa_external_evaluation_executed": bool(external_rows),
        "generalization_diagnosis": diagnosis,
        "candidate_decision": decision,
        "blocking_failures": blockers,
        "production_agentic_v2_active": False,
        "production_promotion_executed": False,
        "canary_execution_performed": False,
        "production_answer_authority_change": False,
        "knowledge_base_mutation_count": safety["knowledge_base_mutation_count"],
        "llm_finish_authority_count": safety["llm_finish_authority_count"],
        "abstain_to_finish_override_count": safety["abstain_to_finish_override_count"],
        "graph_hop_violation_count": safety["graph_hop_violation_count"],
        "benchmark_gold_exposure_count": safety["benchmark_gold_exposure_count"],
        "raw_provider_output_persisted_count": safety["raw_provider_output_persisted_count"],
        "hidden_reasoning_persisted_count": safety["hidden_reasoning_persisted_count"],
        "current_head": current_head(),
        "task_start_head": TASK_START_HEAD,
        "git_head_unchanged_since_task_start": current_head() == TASK_START_HEAD,
        "git_commit_created": current_head() != TASK_START_HEAD,
        "focused_tests_passed": regression.get("focused_tests_passed"),
        "full_suite_passed": regression.get("full_suite_passed"),
        "full_suite_skipped": regression.get("full_suite_skipped"),
        "full_suite_failed": regression.get("full_suite_failed"),
        "new_task0252_full_suite_failure_count": regression.get("new_task0252_full_suite_failure_count"),
        "next_task": (
            "TASK-0253_selective_agent_bounded_production_canary_execution_and_rollback_validation"
            if decision == "advance_to_canary_execution_readiness"
            else "TASK-0251_resume_after_minimum_live_shadow_traffic"
            if not entry["passed"]
            else "TASK-0252_selective_agent_promotion_readiness_gap_diagnosis"
        ),
        "resume_condition": (
            "Complete TASK-0251 with >=30 genuine direct live Search/Ask observations and candidate_decision=advance_to_controlled_promotion_readiness, then rerun TASK-0252."
            if not entry["passed"] else
            "Freeze an independent pdfQA external-generalization subset excluding all prior exposed Knowledge/Question identities, produce E0/E1 records, and rerun TASK-0252."
            if not external_quality_available else None
        ),
        "changed_paths": changed_paths(),
    }
    if write:
        write_json(CONTRACT, _contract())
        write_json(EXTERNAL_MANIFEST, external_manifest)
        write_json(LEAKAGE_AUDIT, leakage)
        RESULT.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "candidate_identity.json": candidate,
            "entry_gate.json": entry,
            "pdfqa_external_generalization_metrics.json": external,
            "pdfqa_rule_vs_selective_comparison.json": {
                "schema_version": "opk-rag.task0252.pdfqa-rule-vs-selective.v1",
                "status": "measured" if external_quality_available else "not_executed",
                "E0": e0 or None,
                "E1": e1 or None,
                "external_quality_non_regressive": external_no_regression if external_quality_available else None,
            },
            "internal_agent_authority_replay.json": internal,
            "live_shadow_authority.json": live,
            "three_way_evidence_comparison.json": {
                "schema_version": "opk-rag.task0252.three-way-evidence.v1",
                "internal_available": bool(task250),
                "external_available": external_quality_available,
                "live_available": entry["passed"],
                "triangulation_complete": bool(task250) and external_quality_available and entry["passed"],
            },
            "generalization_diagnosis.json": {
                "schema_version": "opk-rag.task0252.generalization-diagnosis.v1",
                "classification": diagnosis,
                "allowed_classifications": ["generalizes", "internal_only_gain", "external_safe_but_no_gain", "distribution_sensitive", "regressive", "not_assessed_entry_blocked", "not_assessed_external_evidence_missing"],
            },
            "provider_reliability.json": {
                "schema_version": "opk-rag.task0252.provider-reliability.v1",
                "model": "deepseek-v4-flash",
                "response_rate": live["provider_response_rate"],
                "final_structured_validity": live["final_structured_validity"],
                "threshold": PROVIDER_THRESHOLD,
                "max_transport_retries": 1,
                "max_structural_repairs": 1,
            },
            "controller_exposure.json": {
                "schema_version": "opk-rag.task0252.controller-exposure.v1",
                "live_average_controller_calls": task251.get("average_controller_calls", 0.0),
                "live_llm_invocation_rate": task251.get("llm_invocation_rate", 0.0),
                "live_llm_bypass_rate": task251.get("llm_bypass_rate", 0.0),
            },
            "latency_and_token_metrics.json": {
                "schema_version": "opk-rag.task0252.latency-token.v1",
                "live_estimated_inline_incremental_latency_ms": task251.get("estimated_inline_incremental_latency_ms", 0.0),
                "task0246_full_time_agent_average_latency_ms": FULL_TIME_AGENT_LATENCY_MS,
                "external_status": "measured" if external_rows else "not_executed",
            },
            "reranker_stability.json": {
                "schema_version": "opk-rag.task0252.reranker-stability.v1",
                "source_task": "TASK-0251",
                "runtime_variability_rate": task251.get("reranker_runtime_variability_rate", 0.0),
                "same_candidate_identity_material_score_delta_count": task251.get("same_candidate_identity_material_score_delta_count", 0),
                "terminal_decision_instability_count": task251.get("reranker_terminal_decision_instability_count", 0),
            },
            "safety_metrics.json": {"schema_version": "opk-rag.task0252.safety.v1", **safety},
            "controlled_promotion_readiness_gate.json": gate,
            "canary_plan.json": build_canary_plan(),
            "rollback_policy.json": _rollback_policy(),
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT / name, payload)
        write_jsonl(RESULT / "pdfqa_external_sample_results.jsonl", external_rows)
    return summary


def verify() -> dict[str, Any]:
    summary = build_summary(write=False)
    required = [
        CONTRACT, EXTERNAL_MANIFEST, LEAKAGE_AUDIT,
        ROOT / "tasks/TASK-0252_selective_agent_controlled_promotion_readiness_and_canary_plan.md",
        ROOT / "docs/TASK0252_SELECTIVE_AGENT_CONTROLLED_PROMOTION_READINESS_AND_CANARY_PLAN_REPORT.md",
        ROOT / "run/run_task0252_selective_agent_controlled_promotion_readiness_and_canary_plan.sh",
        ROOT / "scripts/run_task0252_selective_agent_controlled_promotion_readiness_and_canary_plan.py",
        ROOT / "scripts/verify_task0252_selective_agent_controlled_promotion_readiness_and_canary_plan.py",
        RESULT / "candidate_identity.json", RESULT / "entry_gate.json",
        RESULT / "pdfqa_external_generalization_metrics.json", RESULT / "pdfqa_external_sample_results.jsonl",
        RESULT / "pdfqa_rule_vs_selective_comparison.json", RESULT / "internal_agent_authority_replay.json",
        RESULT / "live_shadow_authority.json", RESULT / "three_way_evidence_comparison.json",
        RESULT / "generalization_diagnosis.json", RESULT / "provider_reliability.json",
        RESULT / "controller_exposure.json", RESULT / "latency_and_token_metrics.json",
        RESULT / "reranker_stability.json", RESULT / "safety_metrics.json",
        RESULT / "controlled_promotion_readiness_gate.json", RESULT / "canary_plan.json",
        RESULT / "rollback_policy.json", RESULT / "summary.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    mismatches: dict[str, Any] = {}
    contract = read_json(CONTRACT, {})
    if contract.get("pdfqa_dataset_revision") != DATASET_REVISION:
        mismatches["pdfqa_dataset_revision"] = contract.get("pdfqa_dataset_revision")
    if contract.get("pdfqa_annotation_revision") != ANNOTATION_REVISION:
        mismatches["pdfqa_annotation_revision"] = contract.get("pdfqa_annotation_revision")
    for key in ("production_agentic_v2_active", "production_promotion_executed", "canary_execution_performed", "production_answer_authority_change"):
        if summary.get(key) is not False:
            mismatches[key] = summary.get(key)
    for key in ("knowledge_base_mutation_count", "llm_finish_authority_count", "abstain_to_finish_override_count", "graph_hop_violation_count", "benchmark_gold_exposure_count", "raw_provider_output_persisted_count", "hidden_reasoning_persisted_count"):
        if int(summary.get(key) or 0) != 0:
            mismatches[key] = summary.get(key)
    if not summary["entry_gate_passed"]:
        if summary["candidate_decision"] != "hold" or summary["pdfqa_external_evaluation_executed"] is not False:
            mismatches["entry_fail_closed"] = {"decision": summary["candidate_decision"], "external_executed": summary["pdfqa_external_evaluation_executed"]}
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "verification_passed": not missing and not mismatches,
        "task_status": summary["task_status"],
        "candidate_decision": summary["candidate_decision"],
        "entry_gate_passed": summary["entry_gate_passed"],
        "missing_files": missing,
        "mismatches": mismatches,
    }
