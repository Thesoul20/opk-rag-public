import type { RuntimeTraceV1 } from "../types/runtimeTrace";

export const SHOWCASE_SCENARIO_ORDER = ["S01", "S02", "S03", "S04"] as const;

export type RecoveryKind = "none" | "structure" | "graph" | "fail_closed";

export interface EndToEndModel {
  scenarioId: string | null;
  executionScope: string | null;
  recovery: RecoveryKind;
  graphHop: number | null;
  rerankInput: number | null;
  rerankOutput: number | null;
  evidenceCount: number | null;
  outcomeStatus: string;
  refused: boolean;
  stages: Array<{ id: string; state: string; value?: string | number | null }>;
}

export function buildEndToEndModel(trace?: RuntimeTraceV1): EndToEndModel | null {
  if (!trace) return null;
  const recovery: RecoveryKind = trace.outcome.refused && trace.guard.fail_closed
    ? "fail_closed"
    : trace.graph_recovery.graph_activated
      ? "graph"
      : trace.structure_recovery.triggered
        ? "structure"
        : "none";
  const answerStage = trace.query.execution_scope === "ask" ? trace.answerability.stage_state || "unavailable" : "not_applicable";
  return {
    scenarioId: trace.query.scenario_id || null,
    executionScope: trace.query.execution_scope || null,
    recovery,
    graphHop: trace.graph_recovery.hop_depth ?? null,
    rerankInput: trace.rerank.input_candidate_count ?? null,
    rerankOutput: trace.rerank.output_candidate_count ?? null,
    evidenceCount: trace.evidence.evidence_count ?? null,
    outcomeStatus: trace.trace.status,
    refused: Boolean(trace.outcome.refused || trace.trace.status === "refused"),
    stages: [
      { id: "query", state: trace.query.stage_state || "unavailable", value: trace.query.execution_scope },
      { id: "retrieval", state: trace.retrieval.stage_state || "unavailable" },
      { id: "guard", state: trace.guard.stage_state || "unavailable", value: trace.guard.recovery_action || trace.guard.final_decision },
      { id: "recovery", state: recovery === "none" ? "skipped" : "completed", value: recovery },
      { id: "rerank", state: trace.rerank.stage_state || "unavailable", value: trace.rerank.output_candidate_count },
      { id: "evidence", state: trace.evidence.stage_state || "unavailable", value: trace.evidence.evidence_count },
      { id: "answer", state: answerStage, value: trace.query.execution_scope },
      { id: "outcome", state: trace.trace.status, value: trace.trace.status },
    ],
  };
}
