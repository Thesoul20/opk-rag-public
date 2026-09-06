import type { RuntimeTraceV1, StageState } from "../../types/runtimeTrace";
import type { ControlCenterNavItem, RuntimeVisualStatus } from "../architecture";
import type { InspectorEntityKind } from "../state/ControlCenterState";

export type IntegratedStageId =
  | "conversation_resolution" | "initial_retrieval" | "guard_agent_decision" | "optional_recovery"
  | "candidate_pool" | "reranking" | "evidence_composition" | "answerability" | "generation"
  | "grounding" | "citation" | "outcome";

export interface IntegratedStage {
  id: IntegratedStageId;
  label: string;
  state: string;
  detail: string;
  nav: ControlCenterNavItem;
  inspectorKind?: Exclude<InspectorEntityKind, null>;
}

const labels: Record<IntegratedStageId,string> = {
  conversation_resolution:"Conversation Resolution", initial_retrieval:"Initial Retrieval", guard_agent_decision:"Guard / Agent",
  optional_recovery:"Recovery", candidate_pool:"Candidate Pool", reranking:"BGE Reranking", evidence_composition:"Evidence",
  answerability:"Answerability", generation:"Generation", grounding:"Grounding", citation:"Citation", outcome:"Outcome",
};

function s(value: StageState | string | null | undefined): string { return value || "unavailable"; }
function count(value: number | null | undefined, suffix: string): string { return value == null ? "Unavailable" : `${value} ${suffix}`; }

export function recoveryState(trace: RuntimeTraceV1): { state: string; detail: string; inspectorKind: "structure_recovery"|"graph_recovery" } {
  if (trace.graph_recovery.graph_activated) return { state:s(trace.graph_recovery.stage_state), detail:`Graph · hop ${trace.graph_recovery.hop_depth ?? "?"} · +${trace.graph_recovery.recovered_candidate_count ?? "?"}`, inspectorKind:"graph_recovery" };
  if (trace.structure_recovery.triggered) return { state:s(trace.structure_recovery.stage_state), detail:`Structure · +${trace.structure_recovery.expanded_candidate_count ?? "?"}`, inspectorKind:"structure_recovery" };
  return { state: trace.structure_recovery.stage_state === "skipped" && trace.graph_recovery.stage_state === "skipped" ? "skipped" : s(trace.guard.stage_state), detail:"No bounded recovery", inspectorKind:"graph_recovery" };
}

export function buildIntegratedStages(trace: RuntimeTraceV1): IntegratedStage[] {
  const recovery = recoveryState(trace);
  const finalCandidates = trace.retrieval.final_result_count ?? trace.rerank.output_candidate_count ?? trace.outcome.final_candidate_count;
  const stages: IntegratedStage[] = [
    { id:"conversation_resolution", label:labels.conversation_resolution, state:"unavailable", detail:"No dedicated trace stage", nav:"query" },
    { id:"initial_retrieval", label:labels.initial_retrieval, state:s(trace.retrieval.stage_state), detail:trace.retrieval.retrieval_mode || trace.retrieval.executed_policy || "Unavailable", nav:"query" },
    { id:"guard_agent_decision", label:labels.guard_agent_decision, state:s(trace.guard.stage_state), detail:trace.guard.final_decision || trace.guard.initial_decision || "Unavailable", nav:"runtime", inspectorKind:"guard" },
    { id:"optional_recovery", label:labels.optional_recovery, state:recovery.state, detail:recovery.detail, nav:trace.graph_recovery.graph_activated ? "graph" : "runtime", inspectorKind:recovery.inspectorKind },
    { id:"candidate_pool", label:labels.candidate_pool, state:s(trace.retrieval.stage_state), detail:count(finalCandidates,"final"), nav:"query" },
    { id:"reranking", label:labels.reranking, state:s(trace.rerank.stage_state), detail:count(trace.rerank.output_candidate_count,"ranked"), nav:"query" },
    { id:"evidence_composition", label:labels.evidence_composition, state:s(trace.evidence.stage_state), detail:count(trace.evidence.evidence_count,"items"), nav:"evidence", inspectorKind:"evidence_stage" },
    { id:"answerability", label:labels.answerability, state:s(trace.answerability.stage_state), detail:trace.answerability.answerability_state || trace.answerability.reason_code || "Unavailable", nav:"evidence", inspectorKind:"answerability" },
    { id:"generation", label:labels.generation, state:s(trace.generation.stage_state), detail:trace.generation.generation_abstained ? "abstained" : trace.generation.generation_completed ? "completed" : "Unavailable", nav:"evidence", inspectorKind:"generation" },
    { id:"grounding", label:labels.grounding, state:s(trace.grounding.stage_state), detail:trace.grounding.status || trace.grounding.reason_code || "Unavailable", nav:"evidence", inspectorKind:"grounding" },
    { id:"citation", label:labels.citation, state:s(trace.citation.stage_state), detail:count(trace.citation.citation_count,"citations"), nav:"evidence", inspectorKind:"citation" },
    { id:"outcome", label:labels.outcome, state:trace.outcome.status || trace.trace.status || "unavailable", detail:trace.outcome.status === "refused" && !trace.outcome.failure_stage ? "Safe refusal" : trace.outcome.answer_status || trace.outcome.final_decision || "Unavailable", nav:"evidence", inspectorKind:"outcome" },
  ];
  return stages;
}

export function visualStatus(state: string): RuntimeVisualStatus {
  if (state === "failed") return "failed";
  if (state === "refused") return "refused";
  if (state === "partial") return "partial";
  if (state === "active" || state === "running" || state === "started") return "running";
  if (state === "completed") return "completed";
  if (state === "skipped" || state === "not_applicable" || state === "unavailable" || state === "not_started") return "unavailable";
  return "ready";
}

export function activeExecutionSummary(trace: RuntimeTraceV1) {
  const scope=(trace.query.execution_scope || "unknown").toUpperCase();
  const status=trace.outcome.status || trace.trace.status || "unavailable";
  return {
    traceId: trace.trace.trace_id,
    scope,
    status,
    query: trace.query.query_text || trace.query.normalized_query || "Query unavailable",
    graph: trace.graph_recovery.graph_activated ? `Graph ${trace.graph_recovery.hop_depth ?? "?"}-hop` : "Graph skipped",
    evidence: trace.evidence.evidence_count == null ? "Evidence unavailable" : `Evidence ${trace.evidence.evidence_count}`,
    grounding: trace.grounding.grounding_passed === true ? "Grounded" : trace.grounding.status === "not_applicable" ? "Grounding N/A" : trace.grounding.grounding_passed === false ? "Not grounded" : "Grounding unavailable",
    citations: trace.citation.citation_count == null ? "Citations unavailable" : `Citations ${trace.citation.citation_count}`,
    safeRefusal: status === "refused" && !trace.outcome.failure_stage,
  };
}
